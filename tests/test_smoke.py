"""Smoke tests.

The no-torch tests run anywhere with numpy/sklearn/xgboost. The torch tests are
skipped automatically when PyTorch is not installed, so the suite is green on a
machine without a deep-learning stack.
"""

import importlib.util

import numpy as np
import pytest

from flowmamba.data.features import aggregate_flows, aggregated_feature_names, n_features
from flowmamba.data.preprocess import FlowPreprocessor
from flowmamba.data.synthetic import CLASS_NAMES, make_synthetic_flows

HAS_TORCH = importlib.util.find_spec("torch") is not None
torch_required = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# --------------------------------------------------------------------------- #
# No-torch path
# --------------------------------------------------------------------------- #
def test_synthetic_shapes():
    flows, labels, lengths = make_synthetic_flows(n_per_class=20, max_packets=16, seed=0)
    assert flows.shape[1:] == (16, n_features())
    assert flows.shape[0] == len(labels) == len(lengths)
    assert set(np.unique(labels)).issubset(set(range(len(CLASS_NAMES))))
    assert lengths.min() >= 1 and lengths.max() <= 16


def test_preprocessor_no_leak_and_zero_padding():
    flows, labels, lengths = make_synthetic_flows(n_per_class=30, max_packets=16, seed=1)
    pre = FlowPreprocessor(method="yeo-johnson")
    out = pre.fit_transform(flows)
    # Padding rows must stay exactly zero after transform.
    padding = np.arange(16)[None, :] >= lengths[:, None]
    assert np.allclose(out[padding], 0.0)
    # Transform must be deterministic / reusable.
    out2 = pre.transform(flows)
    assert np.allclose(out, out2)


def test_aggregate_dimensions():
    flows, labels, lengths = make_synthetic_flows(n_per_class=10, max_packets=16, seed=2)
    agg = aggregate_flows(flows, lengths)
    assert agg.shape == (len(labels), len(aggregated_feature_names()))
    assert np.isfinite(agg).all()


def test_xgboost_baseline_runs():
    """Run the XGBoost baseline in a *subprocess*.

    On macOS, PyTorch and XGBoost each bundle their own OpenMP runtime
    (libomp); importing both into one process can deadlock. Isolating the
    baseline in a fresh process avoids the clash so the suite stays green even
    when torch is installed. (For the same reason, prefer running
    `flowmamba baselines` and the torch commands separately at the CLI.)
    """
    import subprocess
    import sys

    # NB: check availability WITHOUT importing xgboost into this process -- a bare
    # import would load xgboost's libomp here and deadlock/segfault the torch
    # tests that run later in the same process.
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")
    proc = subprocess.run(
        [sys.executable, "-m", "flowmamba.cli", "baselines",
         "--n-per-class", "120", "--max-packets", "16"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "XGBoost" in proc.stdout and "Isolation Forest" in proc.stdout


# --------------------------------------------------------------------------- #
# Torch path (skipped without torch)
# --------------------------------------------------------------------------- #
@torch_required
def test_detector_forward_and_modes():
    import torch

    from flowmamba.config import ModelConfig
    from flowmamba.models.detector import Detector

    cfg = ModelConfig(d_model=32, n_layers=2, n_classes=8, proj_dim=8)
    model = Detector(cfg, n_features())
    flow = torch.randn(4, 16, n_features())
    length = torch.tensor([16, 10, 8, 4])

    z = model.embed(flow, length)
    assert z.shape == (4, 32)

    model.anomaly.set_center(model.project(z))
    model.anomaly.calibrate(model.anomaly.scores(model.project(z)), 99.0)

    strong = model.predict(flow, length, mode="strong")
    assert strong.class_logits is not None and strong.class_logits.shape == (4, 8)
    default = model.predict(flow, length, mode="default")
    assert default.class_logits is None
    assert strong.alert.shape == (4,)


@torch_required
def test_masked_collator_masks_valid_positions():
    import torch

    from flowmamba.data.dataset import FlowDataset, MaskedPatternCollator

    flows, labels, lengths = make_synthetic_flows(n_per_class=8, max_packets=16, seed=5)
    pre = FlowPreprocessor(method="none")
    ds = FlowDataset(pre.fit_transform(flows), None, lengths)
    collate = MaskedPatternCollator(mask_prob=0.3, seed=0)
    batch = collate([ds[i] for i in range(4)])
    # Every flow has at least one masked position, and masks fall within length.
    assert batch["mask"].sum(dim=1).min() >= 1
    valid = torch.arange(16)[None, :] < batch["length"][:, None]
    assert bool((batch["mask"] & ~valid).sum() == 0)


@torch_required
def test_tiny_end_to_end():
    """A miniature run of all three stages must complete and produce alerts."""
    from flowmamba.cli import main

    rc = main(
        [
            "demo",
            "--n-per-class", "40",
            "--epochs", "1",
            "--d-model", "32",
            "--n-layers", "2",
            "--max-packets", "12",
            "--device", "cpu",
        ]
    )
    assert rc == 0
