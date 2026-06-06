"""Preprocessing transforms applied before the model sees a flow.

Traffic features are heavy right-tailed: a few flows carry megabytes while most
carry a few hundred bytes. Left raw, that tail dominates every distance and
swamps the one-class anomaly score. The pipeline therefore:

  1. compacts heavy-tailed features with a power transform
     (Yeo-Johnson by default -- learns the exponent per feature and handles
     zeros/negatives; log1p is the transparent fallback), and
  2. standardises every feature to zero mean / unit variance.

The transform is *fit on the training split only* and then frozen, so no test
information leaks. It serialises to a single file for deployment on the gateway.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from flowmamba.data.features import (
    FLAG_FEATURES,
    HEAVY_TAILED,
    PACKET_FEATURES,
)


class FlowPreprocessor:
    """Fit/transform power + standardisation over ``(N, K, F)`` flow tensors.

    Transforms are applied per *feature channel* (the F axis), pooling across all
    packets and flows so a single set of statistics describes each feature.
    Padding rows (all-zero) are excluded when fitting so they do not bias the
    learned statistics.
    """

    def __init__(self, method: str = "yeo-johnson"):
        if method not in ("yeo-johnson", "log1p", "none"):
            raise ValueError("method must be 'yeo-johnson', 'log1p' or 'none'")
        self.method = method
        self.heavy_idx: List[int] = [
            i for i, name in enumerate(PACKET_FEATURES) if name in HEAVY_TAILED
        ]
        self.flag_idx: List[int] = [
            i for i, name in enumerate(PACKET_FEATURES) if name in FLAG_FEATURES
        ]
        self._power = None          # sklearn PowerTransformer, fit on heavy cols
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.fitted_ = False

    # ------------------------------------------------------------------ #
    @staticmethod
    def _valid_rows(flows: np.ndarray) -> np.ndarray:
        """Mask of non-padding packets: any non-zero feature in the row."""
        return np.any(flows != 0.0, axis=-1)

    def fit(self, flows: np.ndarray) -> "FlowPreprocessor":
        flows = np.asarray(flows, dtype=np.float64)
        valid = self._valid_rows(flows)             # (N, K)
        packets = flows[valid]                       # (P, F) -- real packets only

        work = packets.copy()
        if self.method == "log1p" and self.heavy_idx:
            work[:, self.heavy_idx] = np.log1p(np.clip(work[:, self.heavy_idx], 0, None))
        elif self.method == "yeo-johnson" and self.heavy_idx:
            from sklearn.preprocessing import PowerTransformer

            self._power = PowerTransformer(method="yeo-johnson", standardize=False)
            work[:, self.heavy_idx] = self._power.fit_transform(work[:, self.heavy_idx])

        self.mean_ = work.mean(axis=0)
        self.std_ = work.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0           # avoid divide-by-zero on constants
        self.fitted_ = True
        return self

    def transform(self, flows: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("FlowPreprocessor must be fit before transform")
        flows = np.asarray(flows, dtype=np.float64)
        out = flows.copy()
        valid = self._valid_rows(flows)

        packets = out[valid]
        if self.method == "log1p" and self.heavy_idx:
            packets[:, self.heavy_idx] = np.log1p(np.clip(packets[:, self.heavy_idx], 0, None))
        elif self.method == "yeo-johnson" and self._power is not None:
            packets[:, self.heavy_idx] = self._power.transform(packets[:, self.heavy_idx])

        packets = (packets - self.mean_) / self.std_
        out[valid] = packets
        out[~valid] = 0.0                            # keep padding at exactly zero
        return out.astype(np.float32)

    def fit_transform(self, flows: np.ndarray) -> np.ndarray:
        return self.fit(flows).transform(flows)

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FlowPreprocessor":
        import joblib

        return joblib.load(path)
