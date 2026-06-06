"""Command-line entry point.

    flowmamba synth       generate a synthetic flow-pattern dataset (no torch)
    flowmamba baselines   XGBoost + Isolation Forest on aggregated stats (no torch)
    flowmamba train-all   run the three training stages on synthetic data (torch)
    flowmamba evaluate    score a trained detector + classical baselines (torch)
    flowmamba demo        end-to-end smoke test on synthetic data (torch)

Torch-dependent commands import PyTorch lazily, so `synth` and `baselines` run on
a machine without a deep-learning stack.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

from flowmamba.config import DataConfig, ModelConfig, TrainConfig
from flowmamba.data.features import aggregate_flows, aggregated_feature_names, n_features
from flowmamba.data.synthetic import CLASS_NAMES, make_synthetic_flows, train_val_test_split
from flowmamba.utils import ensure_dir, set_seed


# --------------------------------------------------------------------------- #
def _make_dataset(args):
    flows, labels, lengths = make_synthetic_flows(
        n_per_class=args.n_per_class,
        max_packets=args.max_packets,
        n_classes=args.n_classes,
        seed=args.seed,
    )
    return flows, labels, lengths


def cmd_synth(args) -> int:
    set_seed(args.seed)
    flows, labels, lengths = _make_dataset(args)
    ensure_dir(os.path.dirname(args.out) or ".")
    np.savez_compressed(args.out, flows=flows, labels=labels, lengths=lengths)
    print(f"Wrote {len(labels)} flows -> {args.out}")
    print(f"  shape: {flows.shape}  classes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return 0


def cmd_baselines(args) -> int:
    """Classical baselines only -- runs without PyTorch."""
    from flowmamba.data.preprocess import FlowPreprocessor
    from flowmamba.eval.baselines import isolation_forest_baseline, xgboost_baseline

    set_seed(args.seed)
    flows, labels, lengths = _make_dataset(args)

    pre = FlowPreprocessor(method=args.transform)
    tr, va, te = train_val_test_split(len(labels), seed=args.seed)
    flows_tr = pre.fit_transform(flows[tr])
    flows_te = pre.transform(flows[te])

    x_tr = aggregate_flows(flows_tr, lengths[tr])
    x_te = aggregate_flows(flows_te, lengths[te])
    print(f"Aggregated stats: {x_tr.shape[1]} features "
          f"({len(aggregated_feature_names())} named)")

    print("\n=== XGBoost (supervised classifier floor) ===")
    xgb = xgboost_baseline(x_tr, labels[tr], x_te, labels[te], class_names=CLASS_NAMES[: args.n_classes])
    rep = xgb["metrics"]["report"]
    print(f"  macro-F1: {rep['macro avg']['f1-score']:.3f}   "
          f"accuracy: {rep['accuracy']:.3f}   PR-AUC(macro): {xgb['pr_auc']['macro']:.3f}")

    print("\n=== Isolation Forest (unsupervised anomaly floor) ===")
    benign_tr = x_tr[labels[tr] == 0]
    is_attack_te = (labels[te] != 0).astype(int)
    iso = isolation_forest_baseline(benign_tr, x_te, is_attack_te)
    print(f"  ROC-AUC: {iso['metrics']['roc_auc']:.3f}   PR-AUC: {iso['metrics']['pr_auc']:.3f}")
    print(f"  detection rate: {iso['rates']['detection_rate']:.3f}   "
          f"FPR: {iso['rates']['false_positive_rate']:.3f}")
    return 0


def _build_and_train(args, flows, labels, lengths, splits, epochs_scale=1.0):
    """Shared three-stage training used by train-all and demo."""
    import torch  # noqa: F401  (ensures a clear error if torch is missing)

    from flowmamba.data.preprocess import FlowPreprocessor
    from flowmamba.models.detector import Detector
    from flowmamba.training.anomaly import fit_anomaly_head
    from flowmamba.training.finetune import finetune_classifier
    from flowmamba.training.pretrain import pretrain_encoder

    tr, va, te = splits
    model_cfg = ModelConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_classes=args.n_classes
    )
    train_cfg = TrainConfig(
        epochs_pretrain=max(1, int(args.epochs * epochs_scale)),
        epochs_finetune=max(1, int(args.epochs * epochs_scale)),
        epochs_anomaly=max(1, int(args.epochs * epochs_scale)),
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    pre = FlowPreprocessor(method=args.transform)
    flows_tr = pre.fit_transform(flows[tr])
    flows_va = pre.transform(flows[va])
    flows_te = pre.transform(flows[te])

    model = Detector(model_cfg, n_features())

    benign_tr = labels[tr] == 0
    print("\n--- Stage A: masked-pattern pre-training (benign only) ---")
    pretrain_encoder(model, flows_tr[benign_tr], lengths[tr][benign_tr], train_cfg)

    print("\n--- Stage B: supervised fine-tune (focal loss) ---")
    finetune_classifier(model, flows_tr, labels[tr], lengths[tr], train_cfg)

    print("\n--- Stage C: anomaly head (encoder frozen) ---")
    # Use a held-out benign slice (the val split's benign) to calibrate honestly.
    benign_va = labels[va] == 0
    fit_anomaly_head(model, flows_va[benign_va], lengths[va][benign_va], train_cfg, model_cfg)

    ensure_dir(args.out_dir)
    pre_path = os.path.join(args.out_dir, "preprocessor.joblib")
    model_path = os.path.join(args.out_dir, "detector.pt")
    pre.save(pre_path)
    model.save(model_path)
    print(f"\nSaved preprocessor -> {pre_path}\nSaved detector     -> {model_path}")
    return model, pre, (flows_te, labels[te], lengths[te])


def _evaluate_model(model, test, n_classes, mode="strong"):
    from flowmamba.eval.metrics import (
        anomaly_detection_metrics,
        classification_metrics,
        latency_percentiles,
    )
    from flowmamba.inference.runner import GatewayRunner

    import torch

    flows_te, labels_te, lengths_te = test
    runner = GatewayRunner(model, mode=mode, class_names=CLASS_NAMES[:n_classes])
    summary = runner.run(flows_te, lengths_te)

    mdev = next(model.parameters()).device
    out = model.predict(
        torch.as_tensor(flows_te, dtype=torch.float32, device=mdev),
        torch.as_tensor(lengths_te, dtype=torch.long, device=mdev),
        mode="strong",
    )
    preds = out.class_logits.argmax(-1).cpu().numpy()
    ascore = out.anomaly_score.cpu().numpy()
    is_attack = (labels_te != 0).astype(int)

    cls = classification_metrics(labels_te, preds, CLASS_NAMES[:n_classes])
    anom = anomaly_detection_metrics(is_attack, ascore)
    lat = latency_percentiles(summary.latencies_ms)

    print("\n=== Detector evaluation ===")
    rep = cls["report"]
    print(f"  classifier  macro-F1: {rep['macro avg']['f1-score']:.3f}   "
          f"accuracy: {rep['accuracy']:.3f}")
    print(f"  anomaly head  ROC-AUC: {anom['roc_auc']:.3f}   PR-AUC: {anom['pr_auc']:.3f}")
    print(f"  latency ms  p50: {lat['p50']:.2f}  p95: {lat['p95']:.2f}  p99: {lat['p99']:.2f}")
    print(f"  alerts: {summary.n_alerts}/{summary.n_flows} flows ({mode} mode)")
    return {"classification": cls, "anomaly": anom, "latency": lat}


def cmd_train_all(args) -> int:
    set_seed(args.seed)
    flows, labels, lengths = _make_dataset(args)
    splits = train_val_test_split(len(labels), seed=args.seed)
    model, _, test = _build_and_train(args, flows, labels, lengths, splits)
    _evaluate_model(model, test, args.n_classes)
    return 0


def cmd_evaluate(args) -> int:
    from flowmamba.data.preprocess import FlowPreprocessor
    from flowmamba.models.detector import Detector

    set_seed(args.seed)
    model = Detector.load(args.model)
    pre = FlowPreprocessor.load(args.preprocessor)
    flows, labels, lengths = _make_dataset(args)
    _, _, te = train_val_test_split(len(labels), seed=args.seed)
    flows_te = pre.transform(flows[te])
    _evaluate_model(model, (flows_te, labels[te], lengths[te]), args.n_classes, mode=args.mode)
    return 0


def cmd_demo(args) -> int:
    """Fast end-to-end smoke test: tiny data, few epochs, both modes."""
    args.n_per_class = min(args.n_per_class, 200)
    set_seed(args.seed)
    flows, labels, lengths = _make_dataset(args)
    splits = train_val_test_split(len(labels), seed=args.seed)
    model, _, test = _build_and_train(args, flows, labels, lengths, splits, epochs_scale=1.0)
    print("\n>>> Strong mode")
    _evaluate_model(model, test, args.n_classes, mode="strong")
    print("\n>>> Default mode (anomaly head only)")
    _evaluate_model(model, test, args.n_classes, mode="default")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flowmamba", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--seed", type=int, default=1337)
        sp.add_argument("--n-per-class", type=int, default=600, dest="n_per_class")
        sp.add_argument("--max-packets", type=int, default=32, dest="max_packets")
        sp.add_argument("--n-classes", type=int, default=8, dest="n_classes")
        sp.add_argument("--transform", choices=["yeo-johnson", "log1p", "none"],
                        default="yeo-johnson")

    def add_train(sp):
        sp.add_argument("--d-model", type=int, default=128, dest="d_model")
        sp.add_argument("--n-layers", type=int, default=4, dest="n_layers")
        sp.add_argument("--epochs", type=int, default=8)
        sp.add_argument("--batch-size", type=int, default=256, dest="batch_size")
        sp.add_argument("--device", default="auto")
        sp.add_argument("--out-dir", default="artifacts", dest="out_dir")

    sp = sub.add_parser("synth", help="generate a synthetic dataset (.npz)")
    add_common(sp)
    sp.add_argument("--out", default="data/synthetic.npz")
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("baselines", help="classical baselines (no torch)")
    add_common(sp)
    sp.set_defaults(func=cmd_baselines)

    sp = sub.add_parser("train-all", help="run the three training stages")
    add_common(sp)
    add_train(sp)
    sp.set_defaults(func=cmd_train_all)

    sp = sub.add_parser("evaluate", help="evaluate a saved detector")
    add_common(sp)
    add_train(sp)
    sp.add_argument("--model", default="artifacts/detector.pt")
    sp.add_argument("--preprocessor", default="artifacts/preprocessor.joblib")
    sp.add_argument("--mode", choices=["default", "strong"], default="strong")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("demo", help="fast end-to-end smoke test")
    add_common(sp)
    add_train(sp)
    sp.set_defaults(func=cmd_demo, epochs=3, n_per_class=150)
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
