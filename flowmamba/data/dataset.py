"""PyTorch datasets and collators over flow-pattern tensors.

Importing this module requires PyTorch. The rest of :mod:`flowmamba.data`
(features, synthetic generation, preprocessing) does not.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - torch is an optional heavy dep
    raise ImportError(
        "flowmamba.data.dataset requires PyTorch. Install it with "
        "`pip install \"torch>=2.1\"` (Python 3.10-3.12)."
    ) from exc


class FlowDataset(Dataset):
    """Wraps standardised flow tensors and (optional) labels.

    Parameters
    ----------
    flows   : float array (N, K, F) -- already preprocessed.
    labels  : int array (N,), optional. Omit for unlabelled (SSL) data.
    lengths : int array (N,), optional -- true packet counts; used to build the
              padding mask the encoder honours.
    """

    def __init__(
        self,
        flows: np.ndarray,
        labels: Optional[np.ndarray] = None,
        lengths: Optional[np.ndarray] = None,
    ):
        self.flows = torch.as_tensor(np.asarray(flows), dtype=torch.float32)
        self.labels = (
            torch.as_tensor(np.asarray(labels), dtype=torch.long)
            if labels is not None
            else None
        )
        _, k, _ = self.flows.shape
        if lengths is None:
            # infer length from non-zero rows
            valid = (self.flows.abs().sum(dim=-1) > 0).sum(dim=1)
            lengths = valid.numpy()
        self.lengths = torch.as_tensor(np.asarray(lengths), dtype=torch.long).clamp(1, k)

    def __len__(self) -> int:
        return self.flows.shape[0]

    def __getitem__(self, idx: int):
        item = {"flow": self.flows[idx], "length": self.lengths[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def padding_mask(lengths: "torch.Tensor", max_len: int) -> "torch.Tensor":
    """Boolean mask, True for valid (non-padding) positions. Shape (B, max_len)."""
    ar = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ar < lengths.unsqueeze(1)


class MaskedPatternCollator:
    """Collator for Stage A masked-pattern SSL.

    Randomly masks a fraction of *valid* packet positions per flow and returns
    the original values as the reconstruction target. Masked positions are zeroed
    in the input (the encoder learns to in-fill them from context), mirroring the
    masked-token recipe of ET-BERT but on continuous per-packet features.
    """

    def __init__(self, mask_prob: float = 0.15, seed: int = 0):
        self.mask_prob = mask_prob
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, batch):
        flows = torch.stack([b["flow"] for b in batch])      # (B, K, F)
        lengths = torch.stack([b["length"] for b in batch])  # (B,)
        b, k, _ = flows.shape

        valid = padding_mask(lengths, k)                     # (B, K)
        rand = torch.rand(b, k, generator=self.generator)
        mask = (rand < self.mask_prob) & valid               # (B, K) positions to predict

        # Guarantee at least one masked position per flow so the loss is defined.
        no_mask = mask.sum(dim=1) == 0
        if no_mask.any():
            first_valid = valid.float().argmax(dim=1)
            mask[no_mask, first_valid[no_mask]] = True

        target = flows.clone()
        corrupted = flows.clone()
        corrupted[mask] = 0.0
        return {
            "flow": corrupted,
            "target": target,
            "mask": mask,
            "length": lengths,
        }


def standard_collate(batch):
    """Default collator for supervised / inference batches."""
    out = {
        "flow": torch.stack([b["flow"] for b in batch]),
        "length": torch.stack([b["length"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


def numpy_to_loaders(
    flows: np.ndarray,
    labels: Optional[np.ndarray],
    lengths: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    collate=None,
) -> Tuple:
    from torch.utils.data import DataLoader

    ds = FlowDataset(flows, labels, lengths)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate or standard_collate
    )
    return ds, loader
