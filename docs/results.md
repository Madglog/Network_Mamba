# flowmamba — Experiment Results

A running log of real-data experiments. Each entry is self-contained and
reproducible from the commands at the end of its section.

---

## Experiment 1 — IoT-23 CTU-IoT-Malware-Capture-34-1 (Mirai)

**Date:** 2026-06-15
**Status:** first genuine real-data run (single scenario). Synthetic-data runs are
omitted here — they are trivially separable by construction and not meaningful.

### Summary

| Metric | Value |
|---|---|
| Overall accuracy | **96.3 %** |
| Macro-F1 (4 present classes) | **0.793** |
| Weighted-F1 | 0.963 |
| **Anomaly head ROC-AUC** | **0.994** (PR-AUC 0.999) |
| Inference latency (p50 / p95 / p99) | 5.12 / 5.51 / 5.96 ms per flow |
| Training wall-clock | **659 s (≈ 11 min)** |
| pcap → dataset extraction | ≈ 1.5 s |

The headline real-world result is the **one-class anomaly head at ROC-AUC 0.994**
— trained on benign flows only, it separates benign from attack almost perfectly
on real malware traffic. Overall accuracy (96.3 %) is inflated by class imbalance
(the capture is ~87 % Mirai); the honest supervised figure is macro-F1 0.793.

### Environment

| | |
|---|---|
| Hardware | Apple M3 Pro, 12 cores, **CPU-only** (no GPU/MPS) |
| Python | 3.12.6 |
| PyTorch | 2.12.0 |
| NumPy / scikit-learn | 2.4.6 / 1.9.0 |
| dpkt (pcap reader) | 1.9.8 |

### Dataset

Source: Stratosphere Labs **IoT-23**, scenario `CTU-IoT-Malware-Capture-34-1`
(Mirai botnet). Per-packet first-K-packet sequences extracted from the pcap with
dpkt; labels joined from the Zeek `conn.log.labeled` by 5-tuple (direction-agnostic).

| | |
|---|---|
| pcap | 120.5 MB, **233,865 packets** |
| conn.log.labeled | 2.8 MB, **23,145 connections** |
| Extracted dataset (`data/iot23_34-1.npz`) | **4,662 flows**, shape (4662, 32, 13) |
| Sequence length K | 32 packets |
| Per-packet features F | 13 (payload-free header/metadata) |

Class distribution (extracted flows):

| Class | Flows | Share |
|---|---|---|
| Mirai (C&C) | 4,055 | 87.0 % |
| Benign | 290 | 6.2 % |
| DDoS | 211 | 4.5 % |
| Recon (PartOfAHorizontalPortScan) | 106 | 2.3 % |
| **Total** | **4,662** | |

Split: 70 / 15 / 15 train / val / test (`seed=1337`) → 3,263 / 699 / 700 flows.
Stage A (SSL pre-train) and Stage C (anomaly calibration) use the benign subset only.

### Model

Mamba encoder + two heads (pure-PyTorch, CPU-runnable).

| Param | Value |
|---|---|
| d_model | 128 |
| n_layers | 4 |
| d_state / d_conv / expand | 16 / 4 / 2 |
| anomaly projection dim | 32 |
| classifier outputs | 8 (benign + 7 categories; 4 present in this capture) |
| detector.pt size | 2.0 MB |

### Training

Three stages, 8 epochs each. `batch_size=256`, `lr=1e-3`, `weight_decay=1e-4`,
`grad_clip=1.0`, focal `gamma=2.0` (uniform class weights), `seed=1337`, `device=cpu`.

| Stage | What | Progression |
|---|---|---|
| A — masked-pattern SSL (benign only) | pre-train encoder | loss 4.28 → 2.22 |
| B — supervised fine-tune (focal) | classifier | train acc 0.85 → ~0.98 |
| C — deep SVDD (encoder frozen) | anomaly head | SVDD 0.273 → 0.029; radius² calibrated to **0.126** at the 99th benign percentile (1 % FPR target) |

Total training wall-clock: **659 s** (Stage B dominates — the full 4,662-flow set
through the pure-Python selective scan on CPU).

### Results (test set, 700 flows)

Per-class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Benign | 0.959 | 0.940 | 0.949 | 50 |
| DDoS | 1.000 | 0.452 | 0.623 | 42 |
| Recon | 0.429 | 1.000 | 0.600 | 18 |
| Mirai | 1.000 | 1.000 | 1.000 | 590 |
| **macro avg** | 0.847 | 0.848 | **0.793** | 700 |
| **weighted avg** | 0.982 | 0.963 | 0.963 | 700 |

Confusion matrix (rows = true, cols = predicted):

| true ↓ / pred → | Benign | DDoS | Recon | Mirai |
|---|---|---|---|---|
| **Benign** | 47 | 0 | 3 | 0 |
| **DDoS** | 2 | 19 | 21 | 0 |
| **Recon** | 0 | 0 | 18 | 0 |
| **Mirai** | 0 | 0 | 0 | 590 |

Anomaly head (one-class, benign-trained; attack = positive):
**ROC-AUC 0.994, PR-AUC 0.999**; 652 / 700 flows alerted in strong mode.

### Interpretation & caveats

- **Anomaly head is the strong, meaningful result (ROC-AUC 0.994)** — the
  zero-day-style detector the design centres on, verified on real malware traffic.
- **Accuracy 96.3 % is flattered by imbalance** (84 % of the test set is Mirai,
  classified perfectly). The honest supervised metric is **macro-F1 0.793**.
- **Main weakness: DDoS↔Recon confusion** — 21 of 42 DDoS flows are predicted as
  Recon (both are short-packet scan/flood patterns, similar over the first 32
  packets), and the classes are tiny (211 / 106 flows total).
- **Single, benign-poor scenario** (290 benign flows). Numbers should be treated
  as a first data point, not a final benchmark.
- Note: the CLI's live `macro-F1` print averages over all 8 class slots and reads
  0.397 here; with only 4 classes present, the 4 empty slots count as F1 = 0. The
  figure above (0.793) is macro-F1 over the **present** classes.

### Next steps to strengthen

1. Add more IoT-23 scenarios (`prep-pcap --pcap-dir data/raw/iot23`) for class
   balance and diversity — directly targets the DDoS/Recon confusion.
2. Stratified train/val/test split.
3. Per-class focal weights (`TrainConfig.focal_alpha`) to offset Mirai dominance.

### Reproduce

```bash
# 1. fetch the scenario (pcap ~120 MB + Zeek conn.log)
mkdir -p data/raw/iot23/CTU-IoT-Malware-Capture-34-1/bro
base=https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-34-1
curl -o data/raw/iot23/CTU-IoT-Malware-Capture-34-1/capture.pcap \
     "$base/2018-12-21-15-50-14-192.168.1.195.pcap"
curl -o data/raw/iot23/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled \
     "$base/bro/conn.log.labeled"

# 2. extract first-K-packet flows + labels
python -m flowmamba.cli prep-pcap \
  --pcap data/raw/iot23/CTU-IoT-Malware-Capture-34-1/capture.pcap \
  --conn-log data/raw/iot23/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled \
  --out data/iot23_34-1.npz

# 3. train all three stages (CPU)
python -m flowmamba.cli train-all --npz data/iot23_34-1.npz \
  --epochs 8 --device cpu --out-dir artifacts/iot23_34-1
```

> On macOS, prefix XGBoost/baseline commands with
> `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` to avoid the duplicate-libomp crash.
