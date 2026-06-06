"""Stage B -- supervised fine-tuning.

Attach a fresh classifier head on the pre-trained encoder theta_0 and jointly
fine-tune the encoder and head on labelled attack data with focal loss. Output:
the fine-tuned encoder theta_* and the trained classifier.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import torch
from torch.utils.data import DataLoader

from flowmamba.config import TrainConfig
from flowmamba.data.dataset import FlowDataset, standard_collate
from flowmamba.models.detector import Detector
from flowmamba.training.losses import focal_loss
from flowmamba.utils import resolve_device


def finetune_classifier(
    model: Detector,
    flows: np.ndarray,
    labels: np.ndarray,
    lengths: Optional[np.ndarray],
    cfg: TrainConfig,
    freeze_encoder: bool = False,
    log_every: int = 50,
) -> Detector:
    """Jointly fine-tune encoder + classifier with focal loss.

    Set ``freeze_encoder=True`` to train the head alone (used by the
    continual-learning update, where the expensive encoder must stay put).
    """
    device = resolve_device(cfg.device)
    model.to(device)
    model.train()
    model.freeze_encoder(freeze_encoder)

    alpha = None
    if cfg.focal_alpha is not None:
        alpha = torch.tensor(cfg.focal_alpha, dtype=torch.float32, device=device)

    ds = FlowDataset(flows, labels=labels, lengths=lengths)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=standard_collate)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs_finetune):
        running, correct, total = 0.0, 0, 0
        for step, batch in enumerate(loader):
            flow = batch["flow"].to(device)
            y = batch["label"].to(device)
            length = batch["length"].to(device)

            z = model.embed(flow, length)
            logits = model.classifier(z)
            loss = focal_loss(logits, y, gamma=cfg.focal_gamma, alpha=alpha)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            running += loss.item()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
            if log_every and step % log_every == 0:
                print(f"[finetune] epoch {epoch+1}/{cfg.epochs_finetune} "
                      f"step {step} loss {loss.item():.4f} acc {correct/max(1,total):.3f}")
        print(f"[finetune] epoch {epoch+1} mean loss {running/max(1,len(loader)):.4f} "
              f"train acc {correct/max(1,total):.3f}")

    model.freeze_encoder(False)
    model.eval()
    return model
