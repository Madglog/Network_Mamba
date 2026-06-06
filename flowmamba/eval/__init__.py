"""Evaluation: metrics, classical baselines, adversarial study, zero-day proxy.

`metrics` and `baselines` are numpy/sklearn/xgboost only. `adversarial` requires
PyTorch and imports it lazily.
"""

from flowmamba.eval.metrics import (
    classification_metrics,
    pr_auc_per_class,
    latency_percentiles,
    anomaly_detection_metrics,
)
from flowmamba.eval.baselines import (
    xgboost_baseline,
    isolation_forest_baseline,
)

__all__ = [
    "classification_metrics",
    "pr_auc_per_class",
    "latency_percentiles",
    "anomaly_detection_metrics",
    "xgboost_baseline",
    "isolation_forest_baseline",
]
