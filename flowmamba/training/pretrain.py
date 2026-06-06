"""Stage A -- masked-pattern self-supervised pre-training.

Take a large set of *benign* flows, randomly mask ~15% of packet positions, and
train the Mamba encoder to in-fill the masked per-packet feature vectors from
context. The output is an encoder theta_0 that has learned what normal traffic
looks like -- the ET-BERT recipe applied to continuous per-packet features.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import torch
from torch.utils.data import DataLoader

from flowmamba.config import TrainConfig
from flowmamba.data.dataset import FlowDataset, MaskedPatternCollator
from flowmamba.models.detector import Detector
from flowmamba.utils import resolve_device


def pretrain_encoder(
    model: Detector,
    benign_flows: np.ndarray,
    benign_lengths: Optional[np.ndarray],
    cfg: TrainConfig,
    mask_prob: float = 0.15,
    log_every: int = 50,
) -> Detector:
    """Train ``model.encoder`` (and the reconstruction head) on benign flows.

    Only masked positions contribute to the loss, so the encoder must use
    surrounding context rather than copying the (zeroed) input.
    """
    device = resolve_device(cfg.device)
    model.to(device)
    model.train()

    ds = FlowDataset(benign_flows, labels=None, lengths=benign_lengths)
    collate = MaskedPatternCollator(mask_prob=mask_prob, seed=cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)

    params = list(model.encoder.parameters()) + list(model.recon.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs_pretrain):
        running = 0.0
        for step, batch in enumerate(loader):
            flow = batch["flow"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            length = batch["length"].to(device)

            pred = model.reconstruct(flow, length)            # (b, K, F)
            # MSE on masked positions only.
            diff = (pred - target) ** 2                        # (b, K, F)
            loss = diff[mask].mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            running += loss.item()
            if log_every and step % log_every == 0:
                print(f"[pretrain] epoch {epoch+1}/{cfg.epochs_pretrain} "
                      f"step {step} loss {loss.item():.4f}")
        print(f"[pretrain] epoch {epoch+1} mean loss {running/max(1,len(loader)):.4f}")

    model.eval()
    return model
