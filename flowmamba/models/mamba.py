"""A self-contained, CPU-runnable Mamba encoder.

This is a pure-PyTorch implementation of the selective state-space (Mamba) block.
It deliberately does *not* depend on the CUDA `mamba-ssm` selective-scan kernel,
so it runs on a laptop CPU or a Raspberry-Pi-class gateway, which is the
deployment target. The selective scan is written as an explicit sequential
recurrence -- O(L) in sequence length, which is exactly Mamba's advantage over
the O(L^2) attention of a transformer. For the short flow sequences used here
(K <= 32) the Python-level loop is perfectly fast; swap in the fused kernel for
long-sequence training on GPU if available.

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
Spaces" (2023). Input representation and the pre-train/fine-tune recipe follow
NetMamba (Wang et al., 2024).
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "flowmamba.models requires PyTorch. Install with `pip install \"torch>=2.1\"`."
    ) from exc

from flowmamba.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MambaBlock(nn.Module):
    """A single Mamba block: gated selective state-space mixer.

    Parameters follow the paper's naming:
      d_model -- model width D
      d_inner -- expanded inner width (expand * D)
      d_state -- SSM state size N
      d_conv  -- depthwise causal conv width
      dt_rank -- rank of the input-dependent delta projection
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.d_inner = cfg.expand * cfg.d_model
        self.d_state = cfg.d_state
        self.d_conv = cfg.d_conv
        self.dt_rank = cfg.resolved_dt_rank()

        # Input projection -> [x branch, gate (z) branch]
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Depthwise causal 1-D convolution over the sequence
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )

        # x -> (delta, B, C) projection
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        # delta low-rank -> per-channel delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is stored as log for positivity of -A (HIPPO-style init: A = 1..N)
        a = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a).repeat(self.d_inner, 1))  # (d_inner, N)
        self.D = nn.Parameter(torch.ones(self.d_inner))                  # skip connection

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self._init_dt_bias()

    def _init_dt_bias(self):
        # Initialise dt bias so softplus(dt) starts in a sensible range.
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    # ------------------------------------------------------------------ #
    def _selective_scan(self, u, delta, A, B, C):
        """Sequential selective scan.

        Shapes:
          u, delta : (b, d_inner, L)
          A        : (d_inner, N)
          B, C     : (b, N, L)
        Returns y  : (b, d_inner, L)
        """
        b, d_inner, L = u.shape
        N = A.shape[1]

        # Discretise: deltaA = exp(delta * A), deltaB_u = delta * B * u
        deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))          # (b,d,L,N)
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)           # (b,d,L,N)

        h = torch.zeros(b, d_inner, N, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, :, t] * h + deltaB_u[:, :, t]                     # (b,d,N)
            y = torch.einsum("bdn,bn->bd", h, C[:, :, t])                   # (b,d)
            ys.append(y)
        y = torch.stack(ys, dim=2)                                         # (b,d,L)
        return y + u * self.D.unsqueeze(0).unsqueeze(-1)

    def forward(self, x, mask: Optional["torch.Tensor"] = None):
        """x: (b, L, d_model); mask: (b, L) bool, True for valid positions."""
        _, L, _ = x.shape
        xz = self.in_proj(x)                                # (b, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                       # each (b, L, d_inner)

        # Causal depthwise conv along the sequence axis
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L]  # trim right padding
        x_conv = x_conv.transpose(1, 2)
        x_act = F.silu(x_conv)                              # (b, L, d_inner)

        # Project to delta, B, C
        x_dbl = self.x_proj(x_act)                          # (b, L, dt_rank + 2N)
        delta, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))             # (b, L, d_inner)

        A = -torch.exp(self.A_log.float())                  # (d_inner, N)

        y = self._selective_scan(
            x_act.transpose(1, 2),                          # (b, d_inner, L)
            delta.transpose(1, 2),                          # (b, d_inner, L)
            A,
            B.transpose(1, 2),                              # (b, N, L)
            C.transpose(1, 2),                              # (b, N, L)
        )
        y = y.transpose(1, 2)                               # (b, L, d_inner)
        y = y * F.silu(z)                                   # gating
        out = self.out_proj(y)                              # (b, L, d_model)

        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = MambaBlock(cfg)

    def forward(self, x, mask=None):
        return x + self.mixer(self.norm(x), mask)


class MambaEncoder(nn.Module):
    """Embeds per-packet features and runs a stack of Mamba blocks.

    forward() returns the per-position hidden states (used for masked-pattern
    SSL). :meth:`encode` mean-pools over valid positions to a single flow
    embedding z (used by both heads).
    """

    def __init__(self, cfg: ModelConfig, n_features: int):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(n_features, cfg.d_model)
        self.layers = nn.ModuleList([ResidualBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)

    def forward(self, flow, mask=None):
        """flow: (b, L, F) -> hidden (b, L, d_model)."""
        h = self.embed(flow)
        for layer in self.layers:
            h = layer(h, mask)
        return self.norm_f(h)

    def encode(self, flow, mask=None):
        """Pool to a single embedding z per flow: (b, d_model)."""
        h = self.forward(flow, mask)
        if mask is None:
            return h.mean(dim=1)
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class ReconstructionHead(nn.Module):
    """Stage A SSL head: predict masked per-packet feature vectors from context."""

    def __init__(self, cfg: ModelConfig, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, n_features),
        )

    def forward(self, hidden):
        return self.net(hidden)
