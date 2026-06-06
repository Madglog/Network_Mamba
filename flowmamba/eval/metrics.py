"""Evaluation metrics. Pure numpy / scikit-learn -- no PyTorch dependency.

Reports the metrics named in the proposal: per-class precision/recall/F1, PR-AUC
(the honest metric under heavy class imbalance), inference-latency percentiles,
and one-class anomaly-detection scores (ROC-AUC / PR-AUC) for the zero-day proxy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> Dict:
    """Per-class precision/recall/F1 plus macro / weighted aggregates."""
    from sklearn.metrics import classification_report, confusion_matrix

    labels = list(range(len(class_names))) if class_names else None
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names) if class_names else None,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"report": report, "confusion_matrix": cm}


def pr_auc_per_class(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Average precision (area under PR curve), one-vs-rest, per class + macro.

    Parameters
    ----------
    y_true  : (N,) int labels.
    y_score : (N, C) class probabilities / scores.
    """
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize

    n_classes = y_score.shape[1]
    classes = list(range(n_classes))
    y_bin = label_binarize(y_true, classes=classes)
    if y_bin.shape[1] == 1:  # binary edge case
        y_bin = np.hstack([1 - y_bin, y_bin])

    out: Dict[str, float] = {}
    aps: List[float] = []
    for c in range(n_classes):
        if y_bin[:, c].sum() == 0:
            continue
        ap = float(average_precision_score(y_bin[:, c], y_score[:, c]))
        name = class_names[c] if class_names else str(c)
        out[name] = ap
        aps.append(ap)
    out["macro"] = float(np.mean(aps)) if aps else 0.0
    return out


def anomaly_detection_metrics(
    is_attack: np.ndarray,
    anomaly_score: np.ndarray,
) -> Dict[str, float]:
    """ROC-AUC and PR-AUC of the one-class scorer treating attacks as positive.

    Threshold-free, so it measures the anomaly head's ranking quality directly --
    the right lens for the zero-day proxy where a fixed radius is beside the point.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {}
    if len(np.unique(is_attack)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    out["roc_auc"] = float(roc_auc_score(is_attack, anomaly_score))
    out["pr_auc"] = float(average_precision_score(is_attack, anomaly_score))
    return out


def binary_rates(is_attack: np.ndarray, flagged: np.ndarray) -> Dict[str, float]:
    """Detection rate (recall on attacks) and false-positive rate on benign."""
    is_attack = is_attack.astype(bool)
    flagged = flagged.astype(bool)
    tp = np.sum(flagged & is_attack)
    fn = np.sum(~flagged & is_attack)
    fp = np.sum(flagged & ~is_attack)
    tn = np.sum(~flagged & ~is_attack)
    return {
        "detection_rate": float(tp / max(1, tp + fn)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "precision": float(tp / max(1, tp + fp)),
    }


def latency_percentiles(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """p50 / p95 / p99 latency in milliseconds, as required for the gateway target."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "n": int(arr.size),
    }
