"""Per-packet feature schema and flow aggregation.

The detector's primary input is a *flow pattern*: an ordered sequence of the
first K packets in a connection, each described by the payload-free feature
vector below. Because the features are header/metadata only, the representation
works on encrypted traffic.

The same schema feeds the classical baselines (XGBoost, Isolation Forest) after
:func:`aggregate_flows` collapses each sequence into the conventional
~80-statistic vector -- this is deliberately the *representation control* in the
evaluation: it isolates how much accuracy comes from keeping packet order.
"""

from __future__ import annotations

from typing import List

import numpy as np

# Ordered per-packet feature names. Index positions are stable -- other modules
# (preprocessing, synthetic generation, adversarial clipping) refer to them.
PACKET_FEATURES: List[str] = [
    "size",       # 0  payload+header bytes        (heavy-tailed)
    "direction",  # 1  +1 upstream / -1 downstream
    "iat",        # 2  inter-arrival time, seconds (heavy-tailed)
    "ttl",        # 3  IP time-to-live             (heavy-tailed-ish)
    "win",        # 4  TCP window size             (heavy-tailed)
    "syn",        # 5  TCP flag
    "ack",        # 6  TCP flag
    "fin",        # 7  TCP flag
    "rst",        # 8  TCP flag
    "psh",        # 9  TCP flag
    "urg",        # 10 TCP flag
    "is_tcp",     # 11 1 if TCP else 0 (UDP/other)
    "header_len", # 12 L4 header length, bytes
]

# Features whose raw distribution is heavy right-tailed and benefits from a
# log1p / Yeo-Johnson compaction before standardisation.
HEAVY_TAILED = {"size", "iat", "win", "header_len"}

# Binary / categorical features that should pass through power transforms unchanged.
FLAG_FEATURES = {"direction", "syn", "ack", "fin", "rst", "psh", "urg", "is_tcp"}


def n_features() -> int:
    return len(PACKET_FEATURES)


def feature_index(name: str) -> int:
    return PACKET_FEATURES.index(name)


def heavy_tailed_indices() -> List[int]:
    return [PACKET_FEATURES.index(name) for name in PACKET_FEATURES if name in HEAVY_TAILED]


# --------------------------------------------------------------------------- #
# Aggregation -> conventional summary-statistic vector (for classical baselines)
# --------------------------------------------------------------------------- #
_STATS = ("mean", "std", "min", "max")


def aggregated_feature_names() -> List[str]:
    names: List[str] = []
    for feat in PACKET_FEATURES:
        for stat in _STATS:
            names.append(f"{feat}_{stat}")
    names += [
        "n_packets",
        "duration",
        "total_bytes",
        "up_bytes",
        "down_bytes",
        "up_down_ratio",
        "mean_iat",
    ]
    return names


def aggregate_flows(flows: np.ndarray, lengths: np.ndarray | None = None) -> np.ndarray:
    """Collapse ``(N, K, F)`` flow sequences into ``(N, M)`` summary statistics.

    Parameters
    ----------
    flows : array, shape (N, K, F)
        Padded flow-pattern tensor (raw, pre-standardisation).
    lengths : array, shape (N,), optional
        True (unpadded) packet count per flow. If omitted, all K packets are
        treated as valid.

    Returns
    -------
    array, shape (N, M) where M == len(aggregated_feature_names()).
    """
    flows = np.asarray(flows, dtype=np.float64)
    n, k, f = flows.shape
    if lengths is None:
        lengths = np.full(n, k, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64).clip(1, k)

    out = np.zeros((n, f * len(_STATS) + 7), dtype=np.float64)
    dir_idx = feature_index("direction")
    size_idx = feature_index("size")
    iat_idx = feature_index("iat")

    for i in range(n):
        valid = flows[i, : lengths[i]]              # (Li, F)
        col = 0
        for j in range(f):
            v = valid[:, j]
            out[i, col + 0] = v.mean()
            out[i, col + 1] = v.std()
            out[i, col + 2] = v.min()
            out[i, col + 3] = v.max()
            col += 4
        sizes = valid[:, size_idx]
        dirs = valid[:, dir_idx]
        iats = valid[:, iat_idx]
        up_mask = dirs > 0
        out[i, col + 0] = lengths[i]                       # n_packets
        out[i, col + 1] = iats[1:].sum() if lengths[i] > 1 else 0.0  # duration
        out[i, col + 2] = sizes.sum()                      # total_bytes
        out[i, col + 3] = sizes[up_mask].sum()             # up_bytes
        out[i, col + 4] = sizes[~up_mask].sum()            # down_bytes
        out[i, col + 5] = (sizes[up_mask].sum() + 1.0) / (sizes[~up_mask].sum() + 1.0)
        out[i, col + 6] = iats.mean()                      # mean_iat
    return out
