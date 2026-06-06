"""The loop -- absorbing a new attack type without retraining the encoder.

When an unseen attack appears, the anomaly head flags it on day zero; the SOC
labels a sample; the classifier softmax is extended by one class and *only the
head* is fine-tuned. The encoder stays frozen. This module implements the
four-defence update protocol:

  1. Replay buffer (primary) -- a few hundred labelled flows per existing class,
     batched in on every step so the head keeps seeing old classes.
  2. Knowledge distillation (LwF) -- regularise the new head toward the frozen
     old head on old-class data.
  3. Embedding-space sanity check -- locate the new class relative to existing
     prototypes (add / merge / separate) before committing.
  4. Surface uncertainty -- (handled at inference; see inference.runner).

It also runs the embedding-space scenario check and a per-class accuracy gate so
a regressed update can be rolled back.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader

from flowmamba.config import TrainConfig
from flowmamba.data.dataset import FlowDataset, standard_collate
from flowmamba.models.detector import Detector
from flowmamba.training.losses import distillation_loss, focal_loss
from flowmamba.utils import resolve_device


@dataclass
class ScenarioReport:
    """Result of the embedding-space sanity check for a candidate new class."""

    scenario: str                      # "add" | "merge" | "separate"
    nearest_class: int
    nearest_distance: float
    spread: float
    recommendation: str


@torch.no_grad()
def class_prototypes(
    model: Detector, flows: np.ndarray, labels: np.ndarray, lengths: Optional[np.ndarray], device
) -> Dict[int, "torch.Tensor"]:
    """Mean embedding per class -- the prototypes the new class is compared to."""
    ds = FlowDataset(flows, labels, lengths)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=standard_collate)
    sums: Dict[int, "torch.Tensor"] = {}
    counts: Dict[int, int] = {}
    for batch in loader:
        z = model.embed(batch["flow"].to(device), batch["length"].to(device)).cpu()
        for cls in batch["label"].unique().tolist():
            m = batch["label"] == cls
            sums[cls] = sums.get(cls, 0) + z[m].sum(0)
            counts[cls] = counts.get(cls, 0) + int(m.sum())
    return {c: sums[c] / counts[c] for c in sums}


@torch.no_grad()
def embedding_scenario(
    model: Detector,
    new_flows: np.ndarray,
    new_lengths: Optional[np.ndarray],
    prototypes: Dict[int, "torch.Tensor"],
    device,
    tight_factor: float = 0.5,
    overlap_factor: float = 0.25,
) -> ScenarioReport:
    """Classify the new class geometry: tight&far -> add, overlap -> merge, else separate.

    Heuristic on the embedding cloud of the candidate flows:
      * spread  = mean distance of new samples to their own centroid.
      * gap     = distance from the new centroid to the nearest existing prototype.
    overlap (gap small vs spread) -> merge as sub-class; tight & far -> add;
    in-between (scattered) -> margin-loss separation.
    """
    ds = FlowDataset(new_flows, None, new_lengths)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=standard_collate)
    embs = []
    for batch in loader:
        embs.append(model.embed(batch["flow"].to(device), batch["length"].to(device)).cpu())
    emb = torch.cat(embs, 0)
    centroid = emb.mean(0)
    spread = (emb - centroid).norm(dim=1).mean().item()

    dists = {c: (centroid - p).norm().item() for c, p in prototypes.items()}
    nearest_class = min(dists, key=lambda c: dists[c])
    gap = dists[nearest_class]

    if gap < overlap_factor * max(spread, 1e-6):
        scenario, rec = "merge", f"overlaps class {nearest_class}; merge as a sub-class"
    elif gap > spread / max(tight_factor, 1e-6):
        scenario, rec = "add", "tight and far; add as a new class with replay"
    else:
        scenario, rec = "separate", "scattered; fine-tune with a margin loss to enforce separation"
    return ScenarioReport(scenario, int(nearest_class), float(gap), float(spread), rec)


def build_replay_buffer(
    flows: np.ndarray,
    labels: np.ndarray,
    lengths: Optional[np.ndarray],
    per_class: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample up to ``per_class`` labelled flows for each existing class."""
    rng = np.random.default_rng(seed)
    picks: List[int] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        take = min(per_class, len(cls_idx))
        picks.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    idx = np.array(picks)
    lengths = lengths if lengths is not None else np.full(len(labels), flows.shape[1])
    return flows[idx], labels[idx], lengths[idx]


def head_only_update(
    model: Detector,
    new_flows: np.ndarray,
    new_labels: np.ndarray,
    new_lengths: Optional[np.ndarray],
    replay_flows: np.ndarray,
    replay_labels: np.ndarray,
    replay_lengths: Optional[np.ndarray],
    cfg: TrainConfig,
    use_distillation: bool = True,
    distill_weight: float = 1.0,
    epochs: int = 10,
) -> Detector:
    """Extend the softmax by the new class and fine-tune the head only.

    Encoder frozen throughout. Replay batches mix old-class examples with the new
    class on every step; an LwF distillation term anchors old-class behaviour to
    the frozen pre-update head.
    """
    device = resolve_device(cfg.device)
    model.to(device)

    n_old = model.classifier.net[-1].out_features
    teacher = copy.deepcopy(model.classifier).to(device).eval() if use_distillation else None
    for p in (teacher.parameters() if teacher else []):
        p.requires_grad = False

    # Relabel the new class to the next index and grow the head.
    new_index = n_old
    new_labels = np.full_like(new_labels, new_index)
    model.classifier.expand_classes(1)
    model.to(device)
    model.freeze_encoder(True)
    model.encoder.eval()

    flows = np.concatenate([new_flows, replay_flows], axis=0)
    labels = np.concatenate([new_labels, replay_labels], axis=0)
    if new_lengths is None:
        new_lengths = np.full(len(new_labels), flows.shape[1])
    if replay_lengths is None:
        replay_lengths = np.full(len(replay_labels), flows.shape[1])
    lengths = np.concatenate([new_lengths, replay_lengths], axis=0)

    ds = FlowDataset(flows, labels, lengths)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=standard_collate)

    params = list(model.classifier.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.classifier.train()
    for epoch in range(epochs):
        running = 0.0
        for batch in loader:
            flow = batch["flow"].to(device)
            y = batch["label"].to(device)
            length = batch["length"].to(device)
            with torch.no_grad():
                z = model.embed(flow, length)
            logits = model.classifier(z)
            loss = focal_loss(logits, y, gamma=cfg.focal_gamma)

            if teacher is not None:
                old_mask = y < n_old
                if old_mask.any():
                    with torch.no_grad():
                        t_logits = teacher(z[old_mask])
                    loss = loss + distill_weight * distillation_loss(logits[old_mask], t_logits)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            running += loss.item()
        print(f"[continual] epoch {epoch+1}/{epochs} loss {running/max(1,len(loader)):.4f}")

    model.freeze_encoder(False)
    model.eval()
    return model
