"""SHAP attributions for flagged flows.

Every alert in strong mode carries an attribution of which per-packet features
drove the decision, so an analyst can act on it. SHAP values also double as a
*global* feature-importance ranking when averaged (|attribution|) across flows --
the same numbers feed the feature-significance analysis.

Requires both PyTorch and the `shap` package.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class FlowExplainer:
    """Wraps a SHAP explainer around a detector head.

    By default it explains the classifier's predicted-class logit. A background
    set of benign flows is required to define the SHAP baseline (the "expected"
    input). GradientExplainer is used because the Mamba encoder is differentiable
    and it is far cheaper than KernelExplainer on sequence input.
    """

    def __init__(self, model, background: np.ndarray, head: str = "classifier"):
        import torch  # noqa: WPS433

        try:
            import shap  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "FlowExplainer requires the `shap` package: pip install shap"
            ) from exc

        self._torch = torch
        self._shap = shap
        self.model = model.eval()
        self.head = head
        self.background = torch.as_tensor(np.asarray(background), dtype=torch.float32)

        # A flat, differentiable wrapper: flow -> scalar-per-class score.
        class _Wrapper(torch.nn.Module):
            def __init__(self, det, head):
                super().__init__()
                self.det = det
                self.head = head

            def forward(self, flow):
                z = self.det.encoder.encode(flow, None)
                if self.head == "classifier":
                    return self.det.classifier(z)
                # anomaly: return the (negative) squared distance as a score
                return -self.det.anomaly.scores(self.det.project(z)).unsqueeze(-1)

        self._wrapper = _Wrapper(model, head)
        self._explainer = shap.GradientExplainer(self._wrapper, self.background)

    def explain(self, flows: np.ndarray, nsamples: int = 64):
        """Return SHAP values, shape matching ``(N, K, F)`` per output class."""
        torch = self._torch
        x = torch.as_tensor(np.asarray(flows), dtype=torch.float32)
        return self._explainer.shap_values(x, nsamples=nsamples)

    def global_importance(
        self, flows: np.ndarray, feature_names: Optional[list] = None, nsamples: int = 64
    ):
        """Mean |SHAP| per per-packet feature, aggregated over packets and flows.

        Returns a list of ``(feature_name, importance)`` sorted descending -- a
        global feature ranking usable alongside mutual information.
        """
        shap_values = self.explain(flows, nsamples=nsamples)
        # shap_values: list (per class) of arrays (N, K, F) -- stack and average.
        arr = np.abs(np.stack(shap_values)) if isinstance(shap_values, list) else np.abs(shap_values)
        # collapse class, sample and packet axes -> per-feature importance
        importance = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        names = feature_names or [f"f{i}" for i in range(importance.shape[0])]
        ranking = sorted(zip(names, importance.tolist()), key=lambda kv: kv[1], reverse=True)
        return ranking
