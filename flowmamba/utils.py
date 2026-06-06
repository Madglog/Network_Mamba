"""Small cross-cutting helpers (device selection, seeding, timing)."""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if present) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(name: str = "auto"):
    """Return a torch.device, picking the best available when name == 'auto'."""
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@contextmanager
def timer() -> Iterator:
    """Context manager yielding a callable that returns elapsed seconds."""
    start = time.perf_counter()
    elapsed = {"value": 0.0}

    def read() -> float:
        return time.perf_counter() - start

    try:
        yield read
    finally:
        elapsed["value"] = time.perf_counter() - start


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
