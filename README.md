# flowmamba — Flow-Pattern Anomaly Detection at the IoT Gateway

A pre-trained **Mamba** detector for known and zero-day attacks on smart-home /
field-sensor traffic, designed to run at the **home gateway** on Raspberry-Pi-class
hardware. A single state-space encoder feeds two heads: a supervised classifier
for the known attack categories and a one-class **deep SVDD** scorer for zero-day
attacks. Both ride an encoder pre-trained on benign traffic; new attack types are
absorbed by re-training only the lightweight classifier head.

This repository is the implementation of the project proposal
(`docs/proposal_final.tex`). It is structured so the data / preprocessing / classical-
baseline path runs with no deep-learning dependency, while the Mamba core is
complete PyTorch that runs once `torch` is installed.

---

## Why this design (one paragraph)

IoT endpoints are the soft layer — default credentials, no patching, no room for
an on-device agent — so the **gateway** is the only practical vantage point, and
its small hardware is the binding constraint. The detector reads **payload-free
flow metadata** (so it works on encrypted traffic) as an **ordered per-packet
sequence** rather than the usual order-agnostic summary statistics, and uses
**Mamba** instead of a transformer because it reaches comparable accuracy at
linear-time, low-parameter cost. See the proposal for the full argument.

---

## Architecture

```
flow (p_1..p_K) ──▶ Mamba encoder θ*  ──▶ z ──┬──▶ classifier head ─▶ softmax (8 classes)
                    (shared, pre-trained)      │
                                               └──▶ projection (128→32) ─▶ deep SVDD ─▶ anomaly score
                                                                  alert if either head fires ─▶ SHAP ─▶ SOC
```

**Two operating modes** (operator-level switch, not per-flow):

| Mode      | Encoder      | K   | Heads               | SHAP | Use                         |
|-----------|--------------|-----|---------------------|------|-----------------------------|
| `default` | small Mamba  | 8   | anomaly only        | no   | low-power, always on        |
| `strong`  | full Mamba   | 32  | classifier + anomaly| yes  | admin-switched, full fidelity |

**Three training stages, then a loop:**

- **Stage A** — masked-pattern SSL pre-training on benign flows → encoder `θ₀`.
- **Stage B** — supervised fine-tune (focal loss) → `θ*` + classifier.
- **Stage C** — freeze `θ*`, fit projection + deep SVDD on benign embeddings,
  calibrate the radius at the 99th benign percentile (1% FPR).
- **The loop** — a new attack is flagged by the anomaly head on day zero; the
  classifier softmax is extended by one class and **only the head** is re-trained
  with a replay buffer + LwF distillation. The encoder is never touched.

---

## Repository layout

```
flowmamba/
  config.py              # dataclass configs + default/strong mode presets
  data/
    features.py          # per-packet feature schema + flow aggregation
    synthetic.py         # synthetic flow-pattern generator (runnable stand-in)
    preprocess.py        # Yeo-Johnson / log1p + standardisation (leak-free)
    dataset.py           # torch Dataset + masked-pattern SSL collator
    ciciot.py            # CICIoT2023 / TON_IoT / IoT-23 loaders (pcap + CSV)
  models/
    mamba.py             # pure-PyTorch Mamba encoder (CPU-runnable, no CUDA kernel)
    heads.py             # classifier, projection, deep SVDD
    detector.py          # assembled detector + two-mode predict()
  training/
    losses.py            # focal, deep-SVDD, LwF distillation
    pretrain.py          # Stage A
    finetune.py          # Stage B
    anomaly.py           # Stage C
    continual.py         # the loop: embedding-scenario check + head-only update
  eval/
    metrics.py           # per-class P/R/F1, PR-AUC, latency, anomaly AUC
    baselines.py         # XGBoost + Isolation Forest (no torch)
    adversarial.py       # FGSM study (both heads)
    zeroday.py           # leave-one-class-out proxy
  explain/shap_explain.py# per-alert SHAP + global importance
  inference/runner.py    # gateway runner: two modes, uncertainty surfacing
  cli.py                 # `flowmamba` command-line entry point
configs/                 # strong.yaml / default.yaml reference configs
scripts/demo.py          # end-to-end smoke test
tests/test_smoke.py      # no-torch + torch-guarded tests
docs/                    # the written proposal + addenda, notes, study guide
```

---

## Install

The scientific + baseline stack needs only:

```bash
pip install numpy pandas scipy scikit-learn xgboost joblib tqdm
```

The Mamba core additionally needs PyTorch (**Python 3.10–3.12**):

```bash
pip install "torch>=2.1" shap
# or, for everything:  pip install -e ".[all]"
```

> **Note on this machine:** the system interpreter is Python 3.14, for which
> PyTorch has no wheels yet. Create a 3.11/3.12 virtual environment for the
> deep-learning path:
> ```bash
> python3.12 -m venv .venv && source .venv/bin/activate
> pip install -e ".[all]"
> ```
>
> **macOS OpenMP caveat:** PyTorch and XGBoost each ship their own `libomp`.
> Importing both into one Python process on macOS can deadlock or segfault. Run
> the torch commands and `flowmamba baselines` in **separate processes** (the test
> suite already isolates the XGBoost baseline in a subprocess for this reason). A
> conda environment with a single shared OpenMP avoids the clash entirely; the
> pip route works as long as the two libraries are not co-resident in one process.

---

## Quickstart

**No PyTorch required** — synthetic data + classical baselines:

```bash
python -m flowmamba.cli synth --out data/synthetic.npz
python -m flowmamba.cli baselines          # XGBoost + Isolation Forest
```

**Full pipeline** (needs torch) — three stages on synthetic data, both modes:

```bash
python -m flowmamba.cli demo               # fast smoke test
python -m flowmamba.cli train-all --epochs 8   # full synthetic run, saves artifacts/
python -m flowmamba.cli evaluate --mode strong
```

`flowmamba` is also installed as a console script when you `pip install -e .`.

---

## Moving to real data

The synthetic generator is a development stand-in. For the real experiments wire
`flowmamba/data/ciciot.py::flows_from_pcap` to a packet reader (NFStream plugin
or Scapy) to emit first-K-packet sequences from CICIoT2023 / IoT-23 pcaps. The
published aggregated CSVs feed the classical baselines via `load_aggregated_csv`
but cannot drive the Mamba arm (they have already discarded packet order).

Datasets: **CICIoT2023** (primary), **TON_IoT** (cross-dataset generalisation),
**IoT-23** (real malware: Mirai, Torii, Okiru, Gafgyt).

---

## Status / roadmap

Implemented: feature schema, synthetic data, preprocessing, pure-PyTorch Mamba
encoder, both heads, all three training stages, the continual-learning update,
classical baselines, metrics, FGSM, zero-day split, SHAP, two-mode runner, CLI.

Next: real pcap extraction (NFStream), warm-start from a public Mamba checkpoint,
ET-BERT / CNN-BiLSTM comparison arms, mixed-precision int8 quantisation + gateway
latency benchmark, t-SNE/UMAP embedding analysis.
```
