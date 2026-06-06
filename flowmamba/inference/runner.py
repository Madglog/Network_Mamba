"""Gateway inference runner.

Ties the trained detector to the two operating modes and produces SOC-ready
alert payloads. In default mode only the anomaly head runs; in strong mode both
heads run and (optionally) a SHAP attribution is attached to every alert. When
the classifier is genuinely uncertain it surfaces the split rather than forcing a
hard label -- the fourth continual-learning defence.

Requires PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

import torch

from flowmamba.data.synthetic import CLASS_NAMES
from flowmamba.models.detector import Detector
from flowmamba.utils import resolve_device, timer


@dataclass
class Alert:
    """A single SOC alert payload."""

    index: int
    anomaly_score: float
    is_anomaly: bool
    predicted_class: Optional[str] = None
    class_confidence: Optional[float] = None
    uncertain: bool = False
    class_distribution: Optional[Dict[str, float]] = None
    shap_top_features: Optional[List] = None


@dataclass
class RunSummary:
    n_flows: int
    n_alerts: int
    mode: str
    latencies_ms: List[float] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)


class GatewayRunner:
    """Runs a trained detector over batches of flows in a given mode."""

    def __init__(
        self,
        model: Detector,
        mode: str = "strong",
        class_names: Optional[List[str]] = None,
        uncertainty_margin: float = 0.15,
        device: Optional[str] = None,
    ):
        # device=None -> keep the model where it already is (avoids surprise moves
        # between training device and inference device).
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = resolve_device(device)
            model = model.to(self.device)
        self.model = model.eval()
        self.mode = mode
        self.class_names = class_names or CLASS_NAMES
        self.uncertainty_margin = uncertainty_margin

    def _length_tensor(self, lengths):
        if lengths is None:
            return None
        return torch.as_tensor(lengths, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def run(
        self,
        flows: np.ndarray,
        lengths: Optional[np.ndarray] = None,
        explainer=None,
    ) -> RunSummary:
        flow_t = torch.as_tensor(flows, dtype=torch.float32, device=self.device)
        n, _, _ = flow_t.shape
        len_t = self._length_tensor(lengths)

        # Per-flow latency timing (the metric the gateway target is judged on).
        latencies: List[float] = []
        out = self.model.predict(flow_t, len_t, mode=self.mode)
        # Re-time individually for an honest p50/p95/p99 distribution.
        for i in range(n):
            single = flow_t[i : i + 1]
            single_len = len_t[i : i + 1] if len_t is not None else None
            with timer() as elapsed:
                self.model.predict(single, single_len, mode=self.mode)
                dt = elapsed()
            latencies.append(dt * 1e3)

        probs = (
            torch.softmax(out.class_logits, dim=-1)
            if out.class_logits is not None
            else None
        )

        alerts: List[Alert] = []
        alert_mask = out.alert.cpu().numpy()
        for i in range(n):
            if not alert_mask[i]:
                continue
            alert = Alert(
                index=i,
                anomaly_score=float(out.anomaly_score[i]),
                is_anomaly=bool(out.is_anomaly[i]),
            )
            if probs is not None:
                p = probs[i]
                top2 = torch.topk(p, k=min(2, p.numel()))
                top_idx = int(top2.indices[0])
                alert.predicted_class = self.class_names[top_idx]
                alert.class_confidence = float(top2.values[0])
                # Surface uncertainty instead of forcing a label (defence #4).
                if p.numel() > 1:
                    margin = float(top2.values[0] - top2.values[1])
                    alert.uncertain = margin < self.uncertainty_margin
                alert.class_distribution = {
                    self.class_names[c]: float(p[c]) for c in range(p.numel())
                }
            if explainer is not None and self.mode == "strong":
                try:
                    # SHAP's GradientExplainer needs autograd, so step out of the
                    # surrounding no_grad context for the attribution.
                    with torch.enable_grad():
                        alert.shap_top_features = explainer.global_importance(
                            flows[i : i + 1]
                        )[:5]
                except Exception as exc:  # pragma: no cover - SHAP is best-effort
                    alert.shap_top_features = [("shap_failed", str(exc))]
            alerts.append(alert)

        return RunSummary(
            n_flows=n,
            n_alerts=len(alerts),
            mode=self.mode,
            latencies_ms=latencies,
            alerts=alerts,
        )
