"""The two detector heads plus the anomaly projection.

  * ClassifierHead -- supervised softmax over benign + 7 attack categories.
  * ProjectionHead -- learnable R^d_model -> R^proj_dim map that gives the
    one-class scorer a lower-dimensional space (distances concentrate in high
    dimensions; projecting sharpens the anomaly score).
  * DeepSVDDHead  -- packs benign projected embeddings into a minimum-volume
    hypersphere around a centre c; the anomaly score is distance to c.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "flowmamba.models requires PyTorch. Install with `pip install \"torch>=2.1\"`."
    ) from exc

from flowmamba.config import ModelConfig


class ClassifierHead(nn.Module):
    """Small MLP -> logits over n_classes. Trained with focal loss (Stage B)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.head_hidden, cfg.n_classes),
        )

    def forward(self, z):
        return self.net(z)

    def expand_classes(self, n_new: int = 1) -> None:
        """Grow the softmax by ``n_new`` classes for progressive learning.

        Copies the existing output weights so known classes keep their learned
        mapping; new rows are initialised small. The encoder is untouched -- this
        is the cheap "head-only" update from the proposal's continual-learning
        loop.
        """
        old: nn.Linear = self.net[-1]
        in_features, old_out = old.in_features, old.out_features
        new = nn.Linear(in_features, old_out + n_new)
        with torch.no_grad():
            new.weight[:old_out] = old.weight
            new.bias[:old_out] = old.bias
            nn.init.normal_(new.weight[old_out:], std=1e-3)
            nn.init.zeros_(new.bias[old_out:])
        self.net[-1] = new


class ProjectionHead(nn.Module):
    """R^d_model -> R^proj_dim projection, trained jointly with the SVDD objective."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.proj_dim, bias=False),  # bias-free: SVDD convention
        )

    def forward(self, z):
        return self.net(z)


class DeepSVDDHead(nn.Module):
    """One-class deep SVDD scorer.

    Holds the hypersphere centre ``c`` and the calibrated radius ``R``. The
    centre is set to the mean projected benign embedding (standard Deep SVDD
    initialisation); the projection is then trained to minimise the mean squared
    distance to ``c``. At inference, score = ||proj(z) - c||^2 and a flow is
    flagged when score > R^2 (R calibrated at the benign 99th percentile).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj_dim = cfg.proj_dim
        self.register_buffer("center", torch.zeros(cfg.proj_dim))
        self.register_buffer("radius", torch.tensor(0.0))   # threshold (distance, not squared)
        self.center_set = False

    @torch.no_grad()
    def set_center(self, projected: "torch.Tensor", eps: float = 0.1) -> None:
        """Initialise c as the mean projected embedding, nudging away from 0."""
        c = projected.mean(dim=0)
        # Avoid a centre at exactly 0, which admits the trivial all-zero solution.
        c[(c.abs() < eps) & (c >= 0)] = eps
        c[(c.abs() < eps) & (c < 0)] = -eps
        self.center.copy_(c)
        self.center_set = True

    def scores(self, projected: "torch.Tensor") -> "torch.Tensor":
        """Squared distance to the centre. Higher == more anomalous."""
        return torch.sum((projected - self.center) ** 2, dim=1)

    @torch.no_grad()
    def calibrate(self, benign_scores: "torch.Tensor", percentile: float) -> float:
        """Set the radius at the given percentile of benign squared-distance."""
        thr = torch.quantile(benign_scores, percentile / 100.0)
        self.radius.copy_(thr)
        return float(thr)

    def is_anomaly(self, projected: "torch.Tensor") -> "torch.Tensor":
        return self.scores(projected) > self.radius
