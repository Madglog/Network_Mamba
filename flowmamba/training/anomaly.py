"""Stage C -- fit the one-class anomaly head.

Freeze the fine-tuned encoder theta_*, push a held-out *benign* set through it to
get embeddings, train the projection MLP under the deep-SVDD objective, then
calibrate the radius at the 99th percentile of benign scores (1% FPR target).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import torch
from torch.utils.data import DataLoader

from flowmamba.config import ModelConfig, TrainConfig
from flowmamba.data.dataset import FlowDataset, standard_collate
from flowmamba.models.detector import Detector
from flowmamba.training.losses import deep_svdd_loss
from flowmamba.utils import resolve_device


@torch.no_grad()
def _all_projected(model: Detector, loader: DataLoader, device) -> "torch.Tensor":
    chunks = []
    for batch in loader:
        flow = batch["flow"].to(device)
        length = batch["length"].to(device)
        z = model.embed(flow, length)
        chunks.append(model.project(z).cpu())
    return torch.cat(chunks, dim=0)


def fit_anomaly_head(
    model: Detector,
    benign_flows: np.ndarray,
    benign_lengths: Optional[np.ndarray],
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    log_every: int = 50,
) -> Detector:
    """Train projection + deep SVDD on benign embeddings; calibrate the radius.

    The encoder is frozen throughout -- "what's normal" lives in the encoder,
    "how far from normal" lives in this cheap head.
    """
    device = resolve_device(cfg.device)
    model.to(device)
    model.freeze_encoder(True)
    model.encoder.eval()       # keep encoder in eval (no dropout/BN drift)

    ds = FlowDataset(benign_flows, labels=None, lengths=benign_lengths)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=standard_collate)

    # Initialise the SVDD centre from the mean projected benign embedding.
    init_proj = _all_projected(model, loader, device).to(device)
    model.anomaly.set_center(init_proj)

    params = list(model.projection.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.projection.train()
    for epoch in range(cfg.epochs_anomaly):
        running = 0.0
        for step, batch in enumerate(loader):
            flow = batch["flow"].to(device)
            length = batch["length"].to(device)
            with torch.no_grad():
                z = model.embed(flow, length)
            proj = model.project(z)
            loss = deep_svdd_loss(proj, model.anomaly.center)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            running += loss.item()
            if log_every and step % log_every == 0:
                print(f"[anomaly] epoch {epoch+1}/{cfg.epochs_anomaly} "
                      f"step {step} svdd {loss.item():.4f}")
        print(f"[anomaly] epoch {epoch+1} mean svdd {running/max(1,len(loader)):.4f}")

    # Calibrate the threshold on all benign scores.
    model.projection.eval()
    final_proj = _all_projected(model, loader, device).to(device)
    scores = model.anomaly.scores(final_proj)
    thr = model.anomaly.calibrate(scores, model_cfg.svdd_percentile)
    print(f"[anomaly] calibrated radius^2 = {thr:.4f} at p{model_cfg.svdd_percentile}")

    model.freeze_encoder(False)
    model.eval()
    return model
