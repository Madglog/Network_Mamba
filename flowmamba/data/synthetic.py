"""Synthetic flow-pattern generator.

Real training uses CICIoT2023 / TON_IoT / IoT-23 (see :mod:`flowmamba.data.ciciot`).
This module fabricates flows with *class-distinct temporal signatures* so the
whole pipeline -- preprocessing, encoder, both heads, baselines, evaluation --
is runnable end-to-end with no dataset download. It is a development harness, not
a substitute for the real data.

Each class is given a deliberately different packet-order pattern so that a
sequence model can beat an order-agnostic aggregate baseline on it, mirroring the
hypothesis the real experiment tests.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from flowmamba.data.features import feature_index, n_features

# Class 0 is benign; classes 1..7 are the seven CICIoT2023 attack categories.
CLASS_NAMES: List[str] = [
    "Benign",
    "DDoS",
    "DoS",
    "Recon",
    "WebBased",
    "BruteForce",
    "Spoofing",
    "Mirai",
]

_SIZE = feature_index("size")
_DIR = feature_index("direction")
_IAT = feature_index("iat")
_TTL = feature_index("ttl")
_WIN = feature_index("win")
_SYN = feature_index("syn")
_ACK = feature_index("ack")
_FIN = feature_index("fin")
_RST = feature_index("rst")
_FIN = feature_index("fin")
_PSH = feature_index("psh")
_IS_TCP = feature_index("is_tcp")
_HLEN = feature_index("header_len")


def _new_flow(k: int) -> np.ndarray:
    flow = np.zeros((k, n_features()), dtype=np.float64)
    flow[:, _TTL] = 64.0
    flow[:, _WIN] = 8192.0
    flow[:, _IS_TCP] = 1.0
    flow[:, _HLEN] = 20.0
    return flow


def _benign(rng: np.random.Generator, k: int) -> np.ndarray:
    """Request/response handshake then steady alternating exchange."""
    flow = _new_flow(k)
    flow[:, _DIR] = np.where(np.arange(k) % 2 == 0, 1.0, -1.0)
    flow[:, _SIZE] = rng.lognormal(mean=5.5, sigma=0.6, size=k)
    flow[:, _IAT] = rng.lognormal(mean=-3.0, sigma=0.7, size=k)
    flow[0, _SYN] = 1.0
    flow[1, _SYN] = 1.0
    flow[1, _ACK] = 1.0
    flow[2:, _ACK] = 1.0
    flow[3:, _PSH] = (rng.random(k - 3) > 0.4).astype(float)
    return flow


def _ddos(rng: np.random.Generator, k: int) -> np.ndarray:
    """High-rate same-direction SYN flood: tiny IAT, all upstream, SYN set."""
    flow = _new_flow(k)
    flow[:, _DIR] = 1.0
    flow[:, _SIZE] = rng.normal(60.0, 6.0, size=k).clip(40, None)
    flow[:, _IAT] = rng.lognormal(mean=-7.0, sigma=0.4, size=k)
    flow[:, _SYN] = 1.0
    flow[:, _WIN] = rng.normal(1024.0, 64.0, size=k)
    return flow


def _dos(rng: np.random.Generator, k: int) -> np.ndarray:
    """Slow-loris style: upstream PSH dribble, long IATs, half-open."""
    flow = _new_flow(k)
    flow[:, _DIR] = 1.0
    flow[:, _SIZE] = rng.normal(120.0, 20.0, size=k).clip(40, None)
    flow[:, _IAT] = rng.lognormal(mean=-1.0, sigma=0.5, size=k)
    flow[:, _PSH] = 1.0
    flow[0, _SYN] = 1.0
    return flow


def _recon(rng: np.random.Generator, k: int) -> np.ndarray:
    """Port/host scan: many short upstream probes, RST replies, no payload."""
    flow = _new_flow(k)
    flow[:, _DIR] = np.where(np.arange(k) % 2 == 0, 1.0, -1.0)
    flow[:, _SIZE] = rng.normal(54.0, 4.0, size=k).clip(40, None)
    flow[:, _IAT] = rng.lognormal(mean=-5.0, sigma=0.9, size=k)
    flow[0::2, _SYN] = 1.0
    flow[1::2, _RST] = 1.0
    return flow


def _web(rng: np.random.Generator, k: int) -> np.ndarray:
    """Web-based attack: large upstream POST bodies, few packets, high PSH."""
    flow = _new_flow(k)
    flow[:, _DIR] = np.where(np.arange(k) < 2, 1.0, -1.0)
    flow[:, _SIZE] = rng.lognormal(mean=7.0, sigma=0.8, size=k)
    flow[:, _IAT] = rng.lognormal(mean=-2.5, sigma=0.6, size=k)
    flow[:, _PSH] = 1.0
    flow[:, _ACK] = 1.0
    return flow


def _brute(rng: np.random.Generator, k: int) -> np.ndarray:
    """Credential brute force: periodic equal-size request/response pairs."""
    flow = _new_flow(k)
    flow[:, _DIR] = np.where(np.arange(k) % 2 == 0, 1.0, -1.0)
    base = rng.normal(110.0, 5.0)
    flow[:, _SIZE] = base + rng.normal(0.0, 3.0, size=k)
    flow[:, _IAT] = rng.normal(0.2, 0.02, size=k).clip(1e-3, None)  # very regular
    flow[:, _ACK] = 1.0
    flow[1::2, _PSH] = 1.0
    return flow


def _spoofing(rng: np.random.Generator, k: int) -> np.ndarray:
    """ARP/DNS spoofing: bursty downstream, unusual TTLs, mixed protocol."""
    flow = _new_flow(k)
    flow[:, _DIR] = -1.0
    flow[:, _SIZE] = rng.lognormal(mean=4.5, sigma=0.9, size=k)
    flow[:, _IAT] = rng.lognormal(mean=-4.0, sigma=1.0, size=k)
    flow[:, _TTL] = rng.choice([32.0, 128.0, 255.0], size=k)
    flow[:, _IS_TCP] = 0.0  # UDP-heavy
    flow[:, _HLEN] = 8.0
    return flow


def _mirai(rng: np.random.Generator, k: int) -> np.ndarray:
    """Mirai botnet: C2 handshake then sustained outbound flood burst."""
    flow = _new_flow(k)
    half = k // 2
    flow[:half, _DIR] = np.where(np.arange(half) % 2 == 0, 1.0, -1.0)
    flow[half:, _DIR] = 1.0
    flow[:half, _SIZE] = rng.normal(80.0, 10.0, size=half)
    flow[half:, _SIZE] = rng.normal(512.0, 40.0, size=k - half)
    flow[:half, _IAT] = rng.lognormal(mean=-2.0, sigma=0.5, size=half)
    flow[half:, _IAT] = rng.lognormal(mean=-6.5, sigma=0.3, size=k - half)
    flow[0, _SYN] = 1.0
    flow[half:, _PSH] = 1.0
    return flow


_GENERATORS = [_benign, _ddos, _dos, _recon, _web, _brute, _spoofing, _mirai]


def make_synthetic_flows(
    n_per_class: int = 600,
    max_packets: int = 32,
    n_classes: int = 8,
    seed: int = 1337,
    benign_ratio: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a labelled synthetic flow-pattern dataset.

    Parameters
    ----------
    n_per_class : base number of flows per attack class.
    max_packets : K -- sequence length.
    n_classes   : number of classes to emit (1..8); class 0 is always benign.
    seed        : RNG seed for reproducibility.
    benign_ratio: benign flows are oversampled by this factor to mimic the real
                  heavy imbalance (most traffic is benign).

    Returns
    -------
    flows   : float array (N, K, F) -- raw (pre-standardisation) flow patterns.
    labels  : int array (N,) in [0, n_classes).
    lengths : int array (N,) -- true packet count per flow (variable length).
    """
    if not 1 <= n_classes <= len(_GENERATORS):
        raise ValueError(f"n_classes must be in 1..{len(_GENERATORS)}")
    rng = np.random.default_rng(seed)

    flows: List[np.ndarray] = []
    labels: List[int] = []
    lengths: List[int] = []

    for cls in range(n_classes):
        count = int(n_per_class * (benign_ratio if cls == 0 else 1.0))
        gen = _GENERATORS[cls]
        for _ in range(count):
            length = int(rng.integers(max(2, max_packets // 4), max_packets + 1))
            seq = gen(rng, length)
            padded = np.zeros((max_packets, n_features()), dtype=np.float64)
            padded[:length] = seq
            flows.append(padded)
            labels.append(cls)
            lengths.append(length)

    f = np.stack(flows)
    y = np.asarray(labels, dtype=np.int64)
    lens = np.asarray(lengths, dtype=np.int64)

    perm = rng.permutation(len(y))
    return f[perm], y[perm], lens[perm]


def train_val_test_split(
    n: int,
    seed: int = 1337,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    labels: Optional[np.ndarray] = None,
):
    """Return shuffled index arrays for a train/val/test split.

    If ``labels`` is given the split is *stratified*: each class is split by the
    same ratios independently, so rare attack classes (e.g. Recon with ~100
    flows) are guaranteed representation in train, val and test. Without
    ``labels`` the split is a single random permutation, as before.
    """
    rng = np.random.default_rng(seed)
    if labels is None:
        idx = rng.permutation(n)
        n_train = int(ratios[0] * n)
        n_val = int(ratios[1] * n)
        return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]

    labels = np.asarray(labels)
    tr, va, te = [], [], []
    for c in np.unique(labels):
        c_idx = rng.permutation(np.flatnonzero(labels == c))
        n_train = int(ratios[0] * len(c_idx))
        n_val = int(ratios[1] * len(c_idx))
        tr.append(c_idx[:n_train])
        va.append(c_idx[n_train : n_train + n_val])
        te.append(c_idx[n_train + n_val :])
    # Concatenate per-class slices, then reshuffle so batches aren't class-ordered.
    pack = lambda parts: rng.permutation(np.concatenate(parts)) if parts else np.array([], dtype=int)
    return pack(tr), pack(va), pack(te)
