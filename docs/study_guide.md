# Study Guide — Mamba IoT Anomaly Detection

This walks through every concept in the proposal, explains how the pieces fit together, and ends with the questions your mentor is most likely to ask. Read it once for understanding, then re-read the Q&A section the night before. The cheatsheet at the very end has the numbers and facts you should know cold without notes.

---

## 1. The big picture in one paragraph

A flow of packets arrives at the IoT gateway. We take the first ~32 packets of that flow, extract a small feature vector per packet (size, direction, timing, flags, header bytes), and stack them into a sequence. That sequence is fed into a **Mamba encoder** — a state-space model that has been **pre-trained self-supervised on benign-only traffic** so that it already knows what "normal" looks like. The encoder outputs a single embedding vector $\mathbf{z}$ representing the whole flow. Two heads read $\mathbf{z}$ in parallel: a **softmax classifier** that scores known attack categories, and a **one-class anomaly scorer** trained only on benign embeddings that catches anything that looks unlike normal traffic. If either head crosses its threshold, we raise an alert, run SHAP over the input sequence to explain *which packets* drove the decision, and forward everything to the SOC via Kafka. The whole thing runs in under 10ms per flow on Raspberry-Pi-class hardware after int8 quantisation.

That's the project. The rest of this document is unpacking each phrase in that paragraph.

---

## 2. Why this problem, why now

### IoT is the soft layer
Most networks now have hundreds or thousands of IoT devices: cameras, sensors, controllers, smart meters, embedded telemetry. They share a few miserable properties:
- **Weak credentials** — default passwords are common, often hard-coded.
- **No patching** — firmware updates are rare, sometimes impossible.
- **No endpoint compute** — no agent, no EDR, no SIEM sensor can run on them.
- **Physical exposure** — often in unsecured locations.

The Mirai botnet (2016) was built almost entirely from compromised IP cameras and home routers. That's the existence proof that this attack surface is exploitable at scale.

### Why the gateway
Since you can't put anything *on* the device, you have to watch what comes *out* of it. The gateway is the natural place to do that:
- Every device flow passes through it.
- It has compute headroom (Raspberry Pi class or better).
- It's a single deployment point — you don't need to update thousands of devices.

The trade-off: gateway hardware is small. So whatever we deploy has to fit in tens or hundreds of MB of RAM and run within ~10ms per flow. That's the binding constraint that shapes every other design decision.

### What we want the detector to do
Four properties, in priority order:
1. **Catch known attacks** with high precision. Supervised classifier handles this.
2. **Catch unknown (zero-day) attacks**. Unsupervised anomaly scorer handles this.
3. **Work on encrypted traffic**. No payload parsing — flow metadata only.
4. **Explain every alert**. An analyst has to be able to act on the alert.

---

## 3. The input: flow patterns

### What is a "flow"?
Standard 5-tuple definition: all packets between a unique combination of (source IP, destination IP, source port, destination port, protocol) for the duration of a connection. Bidirectional — packets in both directions count as one flow.

### Aggregated representation (the conventional approach)
Most IDS systems take a flow and compute a vector of summary statistics:
- Mean / std / min / max of packet sizes
- Mean / std / min / max of inter-arrival times
- Counts of TCP flags (SYN count, ACK count, etc.)
- Total bytes, total packets
- Duration

This produces a fixed-size vector of around 80 numbers per flow. Tools that do this: **NFStream**, **CICFlowMeter**.

**Problem with this representation**: it's permutation-invariant. You can shuffle the packets and the statistics don't change. But many attack signatures *live* in the ordering — beaconing patterns, scan progressions, handshake oddities, burst behaviour. Aggregation throws this away before the model ever sees it.

### Flow-pattern representation (what we're using)
Instead of aggregating, we **keep the packets as an ordered sequence**. For each flow we take its first $K \approx 32$ packets. Per packet we extract a small feature vector $p_t$ containing:

- **Packet length** in bytes
- **Direction** (uplink: device → server, or downlink: server → device)
- **Inter-arrival time** (IAT) — seconds since the previous packet of this flow
- **TCP flag bits** — SYN, ACK, FIN, RST, PSH, URG
- **Header fields** — TTL, window size, header length
- **Optionally**: the first $M$ bytes of the packet itself (the raw bytes, treated as opaque tokens — same as ET-BERT and NetMamba)

The flow is then $\mathbf{x} = (p_1, p_2, \ldots, p_K)$ — a length-$K$ sequence.

### Why first K and not all packets?
Most flows are short, and most attack signatures show up in the first dozen packets. Long flows would create huge sequences. K=32 is the standard truncation in this literature.

### Why payload-free in semantics?
Even when we include the first M bytes of the packet, we're treating them as an *opaque pattern*, not parsing them. So when traffic is TLS-encrypted (which most modern IoT traffic is), the model doesn't care — it learns statistical patterns in the byte distribution, header fields, and timing, none of which depend on the payload being readable.

---

## 4. The Mamba encoder

This is the technical core of the project. To answer mentor questions confidently, you need to understand four levels: state-space models in general, S4D, Mamba, and what "selective" means.

### State-space models, briefly

A linear state-space model is the workhorse of classical control theory. The basic form:

$$x_{t+1} = A x_t + B u_t$$
$$y_t = C x_t + D u_t$$

Where $u_t$ is the input at time $t$, $x_t$ is a hidden state that summarises the past, and $y_t$ is the output. Matrices $A, B, C, D$ parameterise the dynamics.

For sequence modelling, this is appealing because:
- It has a hidden state that compresses arbitrarily long history.
- It's recurrent — you can run it incrementally on streaming input.
- It can be expressed as a convolution, which makes parallel training fast.

The catch is making $A$, $B$, $C$ behave well over long sequences (gradients don't explode/vanish, the model doesn't forget too fast). This is what the S4 family solved.

### S4 → S4D
**S4** (Gu, Goel, Ré, 2021) was the original "structured state-space model." It used a specific HiPPO-derived initialisation of $A$ that gave the model good long-range memory. The math is clever but the original S4 was hard to train.

**S4D** (Gu, Goel, Ré, NeurIPS 2022) simplified it: $A$ is restricted to be diagonal. This is mathematically cleaner, easier to implement, and only slightly less expressive than full S4. Almost all subsequent work in this family — including Mamba — builds on S4D.

### Mamba — adding selectivity
**Mamba** (Gu & Dao, 2023) made one critical change: it made the state-space parameters **input-dependent**. In S4/S4D, $A$, $B$, $C$ are fixed across time — the same dynamics apply to every input. In Mamba, $B$ and $C$ (and the timestep $\Delta$) are functions of the current input $u_t$. So the model can dynamically *decide what to remember* at each step based on what it just saw.

This is called **selectivity**. The model can "focus" on important tokens and "forget" unimportant ones, the same way attention selectively focuses. It turns out selectivity is the missing ingredient — Mamba matches transformers on long-sequence language modeling at a fraction of the compute.

### Mamba-2 / Structured State-Space Duality (SSD)
**Mamba-2** (Dao & Gu, ICML 2024) showed that transformers and SSMs are two sides of the same underlying mathematical structure (the "duality" in the name). This gives a faster, more expressive Mamba. For the project we plan to start with Mamba-1 and consider Mamba-2 as a stretch-goal upgrade.

### Why this beats transformers at the edge
- **Transformer attention**: $O(K^2)$ in compute and $O(K^2)$ in memory at training time, $O(K)$ memory at inference if you cache the KV.
- **Mamba**: $O(K)$ compute, $O(1)$ memory at inference using its recurrent form.

For $K = 32$ this gap doesn't matter much in raw compute, but two other factors do:
1. **Parameter count.** Mamba models are smaller for comparable accuracy. NetMamba reports ET-BERT-level accuracy at a substantially smaller parameter count.
2. **Inference cost per token.** Mamba is *much* lighter per token than transformer attention even at small $K$, because attention has high constant overhead.

The combined effect: Mamba can fit on a Raspberry Pi where ET-BERT cannot.

---

## 5. The pre-train-then-fine-tune paradigm

This is the recipe that ET-BERT established and NetMamba (and us) inherit.

### ET-BERT — the reference
**ET-BERT** (Lin et al., WWW 2022) applies BERT-style training to network traffic:
1. **Tokenise** raw datagram bytes — treat each byte (or byte-pair) like a word.
2. **Pre-train** a BERT-base transformer on a huge corpus of unlabelled traffic, using two objectives:
   - **Masked-token prediction** — hide some bytes, predict them.
   - **Same-origin-burst prediction** — predict whether two packet bursts come from the same flow.
3. **Fine-tune** the pre-trained encoder for downstream tasks: app identification, encrypted protocol classification, attack detection, etc.

Result: state-of-the-art on multiple encrypted-traffic benchmarks. Cost: roughly 110M parameters — BERT-base scale. Doesn't fit on a gateway.

### NetMamba — the lightweight version
**NetMamba** (Wang et al., 2024) keeps the same recipe but swaps the transformer for a unidirectional Mamba backbone:
- Same tokenisation of packet bytes.
- Same masked-prediction-style pre-training.
- Same fine-tune-for-downstream-task pattern.
- Reports accuracy comparable to ET-BERT.
- Substantially fewer parameters, substantially lower latency.

This is the existence proof that the paradigm works at edge cost. We follow this template directly.

### Why pre-train at all? Why not just train supervised?
Three reasons:
1. **Labelled IoT attack data is limited.** CICIoT2023 has tens of millions of flows, but the *distinct attack types* are few. Pre-training lets us exploit unlabelled benign traffic, which is plentiful.
2. **Better generalisation.** Pre-trained representations transfer to new datasets better than from-scratch supervised representations. This matters for our cross-dataset evaluation on TON_IoT.
3. **The anomaly head needs a good representation of "normal."** Pre-training on benign-only traffic is literally the right way to learn that representation. The anomaly head sits on top of an encoder that has been trained to model the benign distribution.

---

## 6. The two heads, in detail

The encoder produces an embedding $\mathbf{z}$. Two heads sit on top of $\mathbf{z}$ and score it in parallel.

### Classifier head — for known attacks

Architecture: a small MLP (typically two layers) followed by softmax over 7 classes. The classes come from CICIoT2023:
1. **DDoS** — flood attacks (UDP flood, SYN flood, ICMP flood, etc.)
2. **DoS** — denial-of-service from a single source
3. **Recon** — port scans, vulnerability scans
4. **Web** — SQL injection, XSS, brute-force against web forms
5. **Brute Force** — SSH/FTP password attacks
6. **Spoofing** — ARP spoofing, DNS spoofing
7. **Mirai** — Mirai-family botnet traffic

#### Why focal loss instead of cross-entropy

CICIoT2023 is heavily imbalanced. DDoS is overrepresented; Web and Brute Force are tiny. Standard cross-entropy will optimise mostly for DDoS and get bad performance on the rare classes.

**Focal loss** (Lin et al., 2017):

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where $p_t$ is the predicted probability for the true class. The $(1 - p_t)^\gamma$ term **down-weights easy examples** (those the model already gets right with high confidence) and **forces the model to focus on hard examples**. $\gamma = 0$ recovers cross-entropy; $\gamma = 2$ is the standard default. $\alpha_t$ is an optional per-class weight.

### Anomaly head — for zero-days

The point of this head is to score how unlike the benign distribution a flow looks. It's trained only on benign embeddings, so it has no concept of any specific attack — it just measures distance from "normal."

Two implementations are on the table:

**Option 1: Deep SVDD (Support Vector Data Description)**
- Train: minimise $\|f(x) - c\|^2$ over benign flows, where $c$ is a learned (or pre-computed) centre and $f$ is a small network on top of the encoder.
- Effectively: pull all benign embeddings toward a single point in feature space.
- Score: distance from $c$.
- Threshold: anomaly if distance > $\tau$.

**Option 2: Mahalanobis distance**
- Compute the mean $\mu$ and covariance $\Sigma$ of benign embeddings.
- Score: $(z - \mu)^\top \Sigma^{-1} (z - \mu)$.
- This accounts for the *shape* of the benign distribution — directions where benign embeddings vary a lot are weighted less.
- Threshold: anomaly if Mahalanobis distance > $\tau$.

We default to Deep SVDD (more expressive); Mahalanobis is the simpler fallback.

#### Calibrating the threshold

- Embed a held-out benign validation set (never seen during training).
- Compute the score for each benign sample.
- Set the threshold at the **99th percentile** of those scores.
- This targets a **1% false-positive rate** on benign traffic — meaning the anomaly head only fires on 1% of benign flows.

### Score fusion

Rule-based OR:

$$\text{alert} \iff s_\text{cls} \ge \tau_\text{cls} \text{ for some attack class} \;\lor\; s_\text{ano} \ge \tau_\text{ano}$$

The two thresholds are independent. A learned fusion (e.g., logistic regression on $s_\text{cls}$ and $s_\text{ano}$) is possible but adds complexity without an obvious gain — we leave it as future work.

---

## 7. Training in three stages — what happens in each

### Stage A — Self-supervised pre-training

**Input**: unlabelled benign flows from CICIoT2023. No attack labels involved at all.

**Objective**: masked-pattern prediction.

The mechanics:
1. Take a flow's sequence $(p_1, \ldots, p_K)$.
2. Randomly choose ~15% of positions to mask. Replace those $p_t$ with a special learned [MASK] vector.
3. Forward through the Mamba encoder.
4. From the encoder's output at the masked positions, predict the original $p_t$.
5. Loss: MSE between predicted and true feature vectors.

The encoder ends up learning *what should be there given context*. This is a strong inductive bias for modelling normal traffic.

**Alternative — contrastive pre-training** (per ET-SSL):
- For each flow, produce two augmented "views" (e.g., random masking, random reordering of small windows).
- Train the encoder so embeddings of the two views of the same flow are close in embedding space.
- Embeddings of different flows are pushed apart.
- Loss: InfoNCE (a standard contrastive objective).

We plan to ablate both objectives and pick the one that gives better downstream performance.

### Stage B — Supervised fine-tuning

- **Initialise** the encoder from the Stage A weights $\theta_0$.
- **Attach** a fresh classification head (untrained).
- **Train** encoder + head jointly on labelled CICIoT2023 attack data with focal loss.
- **Output**: a fine-tuned encoder $\theta_*$ and a trained classification head.

Why fine-tune the encoder (not just the head)? Because the pre-trained encoder is task-agnostic. Fine-tuning specialises it for the specific attack patterns in CICIoT2023. Linear probing (frozen encoder, head only) would be a useful ablation, but full fine-tuning gives the strongest classifier.

### Stage C — Anomaly head fitting

- **Freeze** the fine-tuned encoder $\theta_*$.
- **Embed** a held-out benign set using $\theta_*$ → get a cloud of benign embedding vectors.
- **Fit** the one-class scorer (Deep SVDD or Mahalanobis) on those benign embeddings.
- **Calibrate** the threshold on a separate benign validation set at the 99th percentile.

Note: the held-out benign set used for fitting the scorer is *disjoint* from the supervised training set, to prevent the threshold from being calibrated on data the encoder has memorised.

### How we test zero-day capability

The anomaly head is trained without any attack data, so we can't evaluate it by standard held-out test split. Instead we do **leave-one-class-out**:

- Hold out, say, the Recon class from Stage B's supervised training.
- The classifier never sees Recon during training.
- The anomaly head never sees any attacks during training either way.
- At test time, evaluate the anomaly head's detection rate on Recon flows.
- Repeat for each class.

This is our proxy for "what happens when a truly novel attack arrives."

---

## 8. The datasets — what to know about each

### CICIoT2023 (primary)
- **Source**: Canadian Institute for Cybersecurity (CIC), University of New Brunswick.
- **Scale**: 105 real IoT devices.
- **Attacks**: 33 distinct attack types grouped into 7 categories (listed above).
- **Format**: PCAP files plus pre-computed flow-feature CSVs.
- **Why it's the default benchmark**: largest and most diverse public IoT-IDS dataset, real devices (not simulated), covers realistic attack types.
- **Known issue**: some label noise in the CSV files (a small fraction of flows are mislabelled). Spot-check during your work.

### TON_IoT (cross-dataset target)
- **Source**: UNSW Canberra.
- **Setup**: testbed of 7 IoT/IIoT devices (fridge, GPS tracker, garage door, weather station, etc.) plus network captures, Linux OS logs, and telemetry.
- **Role in your eval**: never used for training. You train on CICIoT2023, evaluate on TON_IoT, and *measure* how much accuracy drops. Cross-dataset generalisation is hard — the goal is honest measurement, not winning.

### IoT-23 (malware check)
- **Source**: Stratosphere Lab, Czech Technical University in Prague.
- **Content**: 20 real-world malware-infected scenarios + 3 benign scenarios.
- **Malware families**: Mirai, Torii, Okiru, Gafgyt, Hide-and-Seek, Hajime, IRCBot, etc.
- **Why it matters**: lab attacks (like in CICIoT2023) may not look exactly like real-world malware. IoT-23 tests whether your detector generalises to actual malware in the wild.

---

## 9. Evaluation methodology — what gets measured

### Standard classification metrics

**Per-class precision, recall, F1.** Always report these alongside any aggregate. Macro-F1 hides minority-class failure.

**ROC-AUC**: standard but optimistic under heavy imbalance.

**PR-AUC** (area under the precision-recall curve): the honest metric under imbalance. Especially important here because attack classes are very imbalanced relative to benign.

### Cross-dataset evaluation
Train on CICIoT2023 → evaluate on TON_IoT. Expected outcome: meaningful accuracy drop. Real value comes from reporting the drop honestly and analysing where the model fails (which attack types degrade most, etc.).

### Zero-day proxy: leave-one-class-out
Described in Section 7. Held-out classes simulate novel attacks; anomaly head's detection rate on them is the zero-day metric.

### Adversarial: FGSM

**Fast Gradient Sign Method** (Goodfellow, Shlens, Szegedy, 2015):

$$x_\text{adv} = x + \varepsilon \cdot \text{sign}(\nabla_x L(f(x), y))$$

In plain language: take the input, compute the gradient of the loss with respect to the input, take the sign of that gradient (so each feature gets either $+\varepsilon$ or $-\varepsilon$), and add it to the input. This is the simplest adversarial attack — a one-step perturbation in the direction that maximises loss.

In our setting:
- $x$ is the flow-pattern sequence (per-packet features stacked into a tensor).
- We constrain perturbations to *valid* ranges: no negative packet sizes, IAT must stay positive, flag bits stay binary.
- We sweep $\varepsilon$ from small (barely perceptible) to large and plot the detection-rate degradation curve for each head.

**Secondary defence** (per the AdvIoT paper): even when the classifier misclassifies an adversarial input, the *SHAP attribution pattern* on adversarial inputs tends to differ from clean inputs. So an analyst (or an automated system) can flag suspicious attribution patterns as a fallback. We include this check.

### Inference budget
- **Latency**: target $<10$ms p95 per flow at the gateway.
- **Model size**: target small enough to fit on Raspberry Pi 4 (1-8GB RAM). After int8 quantisation, the encoder + heads should be well under 50MB.

---

## 10. Practical deployment concepts

### Int8 quantisation
Default weights and activations are float32 (4 bytes each). After int8 quantisation they become 1 byte each, giving:
- **4x smaller model size**
- **Lower memory bandwidth** at inference
- **Faster inference** on hardware that supports int8 ops (most modern CPUs and accelerators do)

**Post-training quantisation (PTQ)**: convert after training, calibrate with a small representative dataset. No retraining. Quick but may lose some accuracy.

**Quantisation-aware training (QAT)**: fine-tune the model with simulated int8 precision during training, so the weights are robust to quantisation. More work but better accuracy retention.

We default to PTQ. If accuracy loss is too large, fall back to QAT.

### SHAP — explaining alerts

**SHAP** (SHapley Additive exPlanations, Lundberg & Lee 2017) is the standard technique for per-prediction explanation in deep learning.

The intuition comes from cooperative game theory. Imagine the features are players in a game and the model output is the payoff. The Shapley value of a feature is its average marginal contribution to the output, averaged over all possible feature subsets. SHAP computes (approximations of) these Shapley values for any model.

**DeepExplainer** is a SHAP variant specifically for deep networks — it uses gradient information to compute Shapley values more efficiently than the exact game-theoretic formula.

In our case, the input is a sequence of per-packet features. SHAP produces an attribution for *each (packet, feature) pair*. An analyst sees output like:

> "Flow flagged. Top contributors: packet 4 IAT (0.34), packet 5 TCP flags (0.21), packet 7 size (0.18). Pattern suggests beaconing."

Without this, the alert is just a number. With it, the analyst has a starting point.

### Kafka and the SOC
**Kafka** is a distributed message queue / event-streaming system. It's the de-facto standard for shipping alerts from detection systems to downstream consumers.

**SOC** (Security Operations Centre) is the team that triages alerts. In our deployment, the gateway publishes alert events to a Kafka topic, and the SOC dashboard consumes from that topic.

The detector doesn't *block* anything — it raises alerts. Blocking decisions are downstream and out of scope.

---

## 11. The baselines, and what each one proves

You need three baselines because each answers a different question.

### ET-BERT (the upper bound)
**Question it answers**: how much accuracy are we leaving on the table by going to Mamba?

ET-BERT is the published SOTA for encrypted-traffic classification. If our Mamba detector gets within a few F1 points of ET-BERT on CICIoT2023, we're not paying much for the edge-deployability. If it lags badly, the Mamba choice doesn't work.

We can either run ET-BERT directly on a subset or compare against published numbers, depending on compute available.

### CNN-BiLSTM on aggregated statistics (the representation control)
**Question it answers**: is the gain from the Mamba backbone, or from the flow-pattern representation? Or both?

CNN-BiLSTM is a standard supervised IDS baseline using the *old* representation (aggregated flow stats). If our Mamba-on-flow-patterns beats CNN-BiLSTM-on-aggregated-stats by X, we need to know how much of X comes from the input change vs. the model change. Running an additional ablation (Mamba on aggregated stats, or CNN-BiLSTM on flow patterns) would fully decompose this, but the basic two-way comparison is the minimum.

### XGBoost and Isolation Forest (the classical floor)
**Question they answer**: do we need deep learning at all?

If XGBoost on the same features gets comparable performance, then the project's deep-learning machinery is gold-plating. This is a real risk in IDS literature — gradient-boosted trees on hand-crafted features are competitive baselines. We need to *show* the deep models earn their complexity.

---

## 12. Stretch goals — and when each makes sense

**Mamba-2 / SSD swap**. If, after the basic Mamba is deployed and benchmarked, we have latency or memory headroom on the target hardware, we swap the backbone to Mamba-2. Same training pipeline, just a different encoder. Mamba-2 typically gives a small but consistent quality improvement at similar cost.

**E-GraphSAGE branch**. Add a third detection head that operates on a per-window device-communication graph (nodes = devices, edges = recent flows). This catches patterns like "device A suddenly contacts 50 new IPs" that a per-flow detector can miss. Stays within edge budget if the graph is recomputed periodically (e.g., every 60s) rather than streamed continuously.

**Federated pre-training**. If pre-training data has to come from multiple sites that can't share raw pcaps (a real constraint in defence deployments), use FedAvg-style federated learning: each site pre-trains locally, sends weight updates to a central aggregator, the aggregator averages and broadcasts back. No raw data ever leaves the site.

---

## 13. Risks — anticipated, and the mitigation for each

| Risk | Why it might bite | Mitigation |
|------|-------------------|------------|
| Pre-training compute exceeds budget | From-scratch Mamba pre-training on 10M+ flows is expensive | Warm-start from a publicly available Mamba checkpoint and adapt on benign IoT traffic |
| Class imbalance hides minority-class failure | DDoS dominates; Web and Brute Force are rare | Focal loss + always report per-class P/R/F1 |
| Cross-dataset accuracy drop | TON_IoT distribution differs from CICIoT2023 | Report honestly, frame as measurement not target |
| Edge memory overrun | Even quantised Mamba might be too big | Int8 quantise; fall back to smaller config; reduce $K$ if needed |
| Byte-level adversarial sensitivity | Per-byte features can be more perturbation-sensitive than aggregated stats | FGSM study quantifies it; SHAP fingerprint as secondary defence |
| CICIoT2023 label noise | Some flows mislabelled | Spot-check sample; report any patterns |
| Pre-trained encoder doesn't transfer | Pre-training on CICIoT2023 benign might not generalise to TON_IoT benign | Test it; if bad, pre-train on a more diverse benign corpus |

---

## 14. Anticipated mentor questions, with answers

### Architecture questions

**Q: Why Mamba and not Mamba-2?**
A: Mamba-2 is newer and slightly better, but the original Mamba is more mature, has more reference implementations, and is enough to demonstrate the edge-cost argument. We do Mamba-2 as a stretch goal once Mamba is working end-to-end.

**Q: Why a single shared encoder, not two separate networks?**
A: Two reasons. First, the encoder is the expensive part — running it once per flow and reading two heads off the same embedding halves the inference cost. Second, the shared representation is *the* point of the SSL paradigm — both heads benefit from the same pre-training on benign traffic.

**Q: Why first K packets and not all of them?**
A: Most attack signatures are in the early packets. Long flows would create enormous sequences and waste compute. K=32 is the standard truncation in this literature; we'll ablate it.

**Q: Why focal loss instead of just oversampling minority classes?**
A: Focal loss doesn't need to balance the dataset — it weights the loss function instead. Oversampling can produce overfitting on the duplicated minority samples; focal loss is more principled. SMOTE-style synthetic minority samples are also an option but risk creating unrealistic samples in high-dimensional flow space.

### Training questions

**Q: Why pre-train at all, not just train end-to-end with labels?**
A: Three reasons: limited labelled attack data; better generalisation (especially cross-dataset); the anomaly head needs a representation of "benign" that's best learned from benign-only data, which is exactly what pre-training gives us.

**Q: Why mask 15% of positions in pre-training?**
A: 15% is the BERT default and has stuck across most masked-language-model work. We'll ablate it but don't expect big surprises.

**Q: Why use the fine-tuned encoder for the anomaly head, not the pre-trained one?**
A: Mostly empirical. The fine-tuned encoder has been specialised for attack-relevant features, so benign embeddings should be more sharply differentiated from attacks. The alternative — use the purely pre-trained encoder — is a useful ablation; we'll run it.

**Q: Where does benign pre-training data come from?**
A: From CICIoT2023 itself, filtered to the benign-labelled flows. We can also augment with public benign captures from other corpora if we need a larger pre-training set.

### Evaluation questions

**Q: Why CICIoT2023 and not BoT-IoT or N-BaIoT?**
A: CICIoT2023 is newer (2023), larger (105 devices, 33 attacks), and more diverse than older datasets. BoT-IoT (2018) is mostly DDoS. N-BaIoT is botnet-only. CICIoT2023 is the most defensible default.

**Q: What's your acceptance criterion for "good enough"?**
A: At minimum: macro-F1 within 3 points of ET-BERT on CICIoT2023; PR-AUC above 0.85 on the cross-dataset TON_IoT test; p95 inference latency under 10ms on the target gateway hardware.

**Q: How do you know cross-dataset generalisation is realistic, not over-tuned?**
A: TON_IoT is *never* used during training or for any hyperparameter selection. It's a held-out test. The number we report is the number we get on the first evaluation.

**Q: Why FGSM specifically?**
A: It's the standard entry-point adversarial attack in the IDS literature. If the model survives FGSM we extend to PGD or stronger attacks. If it doesn't survive FGSM, the stronger attacks won't change the conclusion.

### Theoretical questions

**Q: What's a state-space model in one sentence?**
A: A model that summarises the past in a learned hidden state via a linear recurrence — $x_{t+1} = A x_t + B u_t$, $y_t = C x_t$ — with $A, B, C$ parameterised cleverly so the recurrence handles long sequences efficiently.

**Q: Why is Mamba linear-time while transformers are quadratic?**
A: Transformer attention computes pairwise interactions between all pairs of sequence positions — that's $O(K^2)$ in $K$. Mamba is a recurrence — each step does $O(1)$ work given the previous state, so the whole sequence is $O(K)$.

**Q: What does "selective" mean in selective state-space models?**
A: The state-space parameters $B, C$ and timestep $\Delta$ are made input-dependent — they're functions of the current input, not fixed constants. This lets the model dynamically decide what information to retain or discard at each step, the same way attention selectively focuses.

**Q: What's the difference between S4D and Mamba?**
A: S4D has *fixed* state-space parameters (same dynamics every step). Mamba makes them *input-dependent* (selectivity). That single change is what made SSMs competitive with transformers on language tasks.

### Deployment questions

**Q: What gateway hardware do you target?**
A: Raspberry Pi 4-class as the default (1–8 GB RAM, ARM Cortex-A72). We can also benchmark on a Jetson Nano if a GPU-class edge device is in scope. Confirm with you what's actually available.

**Q: How do you handle the fact that real traffic is way larger than CICIoT2023?**
A: CICIoT2023 is the training corpus; real deployment traffic just flows through the trained model. The model is small (post-quantisation, well under 100MB), and inference is per-flow, not per-bulk-corpus. Throughput is bounded by feature extraction (NFStream is the bottleneck) not the model.

**Q: What if the encoder is too big after pre-training?**
A: Three knobs: (i) reduce encoder depth or hidden dimension, (ii) quantise more aggressively (int8 → mixed int4/int8), (iii) reduce $K$ to shorten the sequence. We'll pick the combination that hits the latency target with the smallest accuracy hit.

**Q: How does the system actually run in production?**
A: A user-space process at the gateway taps the network interface (via libpcap or eBPF), assembles flows incrementally, extracts the per-packet features for the first $K$ packets of each flow, feeds the sequence to the encoder + heads, and publishes any alert (with SHAP attributions) to a Kafka topic. The SOC dashboard consumes from Kafka. Decisions to block, isolate, or drop are downstream and out of scope for this project.

### Hard questions

**Q: What's your novelty? This sounds like NetMamba applied to IoT.**
A: Honest answer: the contribution is the *combination* — flow-pattern Mamba detector + shared encoder + dual heads + SHAP + cross-dataset + adversarial robustness + edge benchmark, end to end. NetMamba is the architectural template; the integration with anomaly detection, the analyst-facing explanations, the adversarial study, and the deployment evaluation on gateway hardware are what we add.

**Q: Could an adversary poison the benign pre-training data?**
A: Yes if they can inject crafted "benign" flows into our pre-training corpus. Defending against data poisoning is out of scope here — we flag it as a known limitation. Defences would involve data-provenance constraints and statistical filtering of the pre-training corpus.

**Q: What if the gateway itself is compromised?**
A: Out of scope. That's a hardware root-of-trust / TEE problem. Worth flagging as a known limitation of any gateway-resident detector.

---

## 15. Cheatsheet — facts to know cold (no notes)

**Numbers**
- $K = 32$ packets per flow (starting point, will ablate)
- ~80 stats per flow in the aggregated baseline
- Focal loss $\gamma = 2$
- Anomaly threshold at 99th percentile of benign → 1% FPR target
- Inference target: $<10$ms p95 per flow
- Pre-training mask fraction: ~15%
- CICIoT2023: 105 devices, 33 attacks, 7 classes

**One-line definitions**
- **Flow**: bidirectional 5-tuple connection
- **Flow pattern**: ordered sequence of per-packet features
- **State-space model**: $x_{t+1} = A x_t + B u_t$, $y_t = C x_t$
- **Mamba**: SSM with input-dependent $B$, $C$, $\Delta$ — i.e., selective
- **S4D**: SSM with diagonal $A$ matrix
- **Mamba-2 / SSD**: shows transformers and SSMs are one structure; faster Mamba
- **NetMamba**: NetMamba = ET-BERT recipe + Mamba backbone for traffic classification
- **ET-BERT**: BERT pre-trained on raw datagram bytes for encrypted traffic
- **Focal loss**: cross-entropy with $(1 - p_t)^\gamma$ down-weighting easy examples
- **Deep SVDD**: pull benign embeddings toward a centre; score = distance from centre
- **Mahalanobis distance**: covariance-aware distance to benign mean
- **FGSM**: $x_\text{adv} = x + \varepsilon \cdot \text{sign}(\nabla_x L)$
- **SHAP**: Shapley values for feature attribution
- **Int8 quantisation**: weights/activations stored in 1 byte instead of 4

**Three pillars (memorise the phrase)**
1. Flow patterns, not aggregated stats
2. Pre-train on benign-only, share the encoder
3. Mamba backbone — not transformer — for edge feasibility

**The one-paragraph elevator** (use this if the mentor opens with "explain it to me"):
> "We detect bad traffic at the IoT gateway. Each flow becomes a sequence of its first 32 packets' features — sizes, timing, headers — instead of being aggregated into summary statistics. A Mamba encoder, pre-trained self-supervised on benign-only traffic, turns the sequence into an embedding. Two heads sit on top of the encoder: a softmax classifier for known attacks, and a one-class anomaly scorer for zero-days. Either head firing raises an alert, with SHAP attributions explaining why. We chose Mamba over a transformer like ET-BERT because Mamba is linear in sequence length and small enough to fit a Raspberry Pi at sub-10-millisecond inference."

If you can say that paragraph cold and answer most of the Q&A in Section 14 without notes, you're ready.
