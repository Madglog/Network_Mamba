"""Zero-day proxy -- leave-one-class-out evaluation.

The concrete answer to "how do you evaluate something you haven't trained on":
hide one attack class from the supervised head during Stage B, then measure how
well the *anomaly head* (which never trains on attacks at all) detects the
held-out class at test time. Repeat per class.

This module provides the data-splitting helper; the per-class training loop is
driven from the CLI / a notebook so the heavy three-stage pipeline stays explicit.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def leave_one_class_out_splits(
    labels: np.ndarray,
    attack_classes: Tuple[int, ...] | None = None,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(held_out_class, train_idx, test_idx)`` for each attack class.

    ``train_idx`` excludes every flow of the held-out class (the supervised head
    never sees it); ``test_idx`` is exactly the held-out class plus benign, so the
    anomaly head's detection rate on a genuinely unseen attack can be measured.

    Benign (class 0) is never held out -- it defines "normal".
    """
    classes = np.unique(labels)
    if attack_classes is None:
        attack_classes = tuple(int(c) for c in classes if c != 0)

    for held in attack_classes:
        train_idx = np.where(labels != held)[0]
        test_idx = np.where((labels == held) | (labels == 0))[0]
        yield int(held), train_idx, test_idx


def detection_rate_on_held_out(
    is_attack: np.ndarray,
    flagged: np.ndarray,
) -> float:
    """Fraction of held-out-class flows the anomaly head flags (recall)."""
    is_attack = is_attack.astype(bool)
    if is_attack.sum() == 0:
        return float("nan")
    return float(np.sum(flagged.astype(bool) & is_attack) / is_attack.sum())
