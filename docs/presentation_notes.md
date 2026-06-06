# Talking Script — Mentor Meeting

**Setting:** informal, just talking. No slides, no laptops. Joint with Yashika.
**Goal:** explain (1) the area we're working in, (2) the specific problem, (3) what we're going to build.

Read this through a few times. Don't memorise word-for-word — internalise the *flow* so you can talk naturally. Each section below is roughly 1–2 minutes spoken.

---

## Part 1 — The area we're working in (≈ 2 minutes)

Start broad. Set the scene.

> "Our project sits at the intersection of three things: IoT security, network traffic analysis, and machine learning.
>
> The basic setup we care about looks like this: you have a bunch of small IoT devices in the field — sensors, cameras, controllers, whatever. Each one talks to a central collector, usually through a gateway. So the traffic flows in one direction: device → gateway → central compute.
>
> That upstream path is what we're focused on. Not the devices themselves, not the central server, but the *path* between them.
>
> Why does this matter? Because IoT devices are notoriously the weakest part of any network. They ship with default passwords, they run firmware that rarely gets patched, they often have physical access exposure, and they don't have the compute headroom to run any real security software on them. So in practice, if an attacker wants a foothold in a network, IoT is usually where they start. Mirai is the famous example — a botnet built almost entirely out of compromised cameras and routers.
>
> In a defence context this gets more serious. Sensors in the field, surveillance equipment, base infrastructure — these are exactly the kind of high-value, low-trust devices an adversary would target. And we can't go and harden every device individually; many of them physically can't be modified.
>
> So the area we're working in is: **how do you detect that something has gone wrong with one of these devices, without having to touch the device itself?** And the modern answer to that is: you watch the network traffic, and you use machine learning to spot anomalies."

That's your opener. It puts you in the room as people who understand the landscape, not just two students with a topic.

---

## Part 2 — The specific problem (≈ 2 minutes)

Narrow down from the area to the actual problem you're solving.

> "The specific problem is this: you have a stream of network flows coming from IoT devices through a gateway, and you need to flag the ones that are suspicious. In real time, before they cause damage.
>
> That sounds simple but it has four hard parts.
>
> **First**, you don't know in advance what attacks look like. There are known attack patterns — DDoS, port scanning, brute force, things like Mirai — and there are unknown ones, the so-called zero-days. Any detector that only catches known attacks will miss the next new thing. So we need to handle both.
>
> **Second**, most modern traffic is encrypted. So we can't look at the payload of a packet to see what's inside. We have to work with metadata only — packet sizes, timing, how many packets in each direction, things like that. The flow shape, basically.
>
> **Third**, we have to run this at the gateway, which is small hardware. Not a big server. Something like a Raspberry Pi or a Jetson Nano. That means the model has to be lightweight — low memory, fast inference, ideally under ten milliseconds per flow.
>
> **Fourth**, when the model does flag something, an analyst has to act on it. So 'the model said so' is not enough — we have to explain *why* a particular flow was flagged. Which feature triggered it. Otherwise the alert is useless and the analyst learns to ignore the system.
>
> Existing approaches each get one or two of these right. Signature-based firewalls catch known attacks but miss zero-days. Standard enterprise intrusion detection systems are tuned for human traffic, not the periodic machine-to-machine patterns that IoT generates, so they false-alarm constantly. Deep models like transformers work well but don't fit on edge hardware. Most academic papers focus on a single dataset and don't measure how the model holds up when an attacker actually tries to evade it.
>
> So the gap we're trying to fill is a detector that does all four things — known *and* unknown attacks, encrypted traffic, gateway-scale hardware, and explainable alerts — and that we actually stress-test, not just benchmark."

This is the meat. If they only remember one section, this is the one.

---

## Part 3 — The solution we're proposing (≈ 3 minutes)

Now you walk them through what you're actually going to build. Keep it concrete.

> "Our solution is a hybrid detector that sits at the IoT gateway. It has two models running in parallel on every network flow.
>
> The first model is a **supervised classifier**. We train it on a labelled IoT attack dataset — the standard one is called CICIoT2023, which has 105 real devices and 33 attack types across 7 categories. The classifier learns to recognise those known attack patterns. Architecturally it's a CNN followed by a bidirectional LSTM — the CNN picks up local patterns in the feature vector, the LSTM captures sequence structure when you look at multiple flows from the same source. We chose this over a transformer because transformers don't fit on gateway hardware — that's been measured, it's not a guess.
>
> The second model is an **unsupervised autoencoder**. This one is trained only on benign traffic — it never sees an attack during training. What it learns is what *normal* looks like. At test time, it tries to reconstruct an incoming flow; if the reconstruction error is high, that means the flow doesn't look like anything it has seen before, and we flag it. This is the arm that catches zero-days, because by definition you can't train a supervised model on attacks that haven't happened yet. The autoencoder doesn't care what the attack is — it just notices the flow doesn't fit the normal distribution.
>
> Both models score every flow. If *either* of them raises an alert, we flag it. The two arms are complementary: the supervised one is precise on known attacks, the unsupervised one is broad on novel ones.
>
> The features we feed both models are computed from flow metadata — packet counts, byte counts, inter-arrival times, flag distributions, things like that. We use a standard tool called NFStream or CICFlowMeter to extract them from raw packet captures. We deliberately do **not** look at packet payloads, because we want this to work on encrypted traffic. Then we run a mutual-information feature selection step to drop the redundant features — that cuts the dimensionality by more than half with almost no accuracy loss.
>
> When something gets flagged, we run a technique called **SHAP** on the alert, which tells us which specific features drove the decision — like 'this flow was flagged because the packet inter-arrival time and the byte ratio were unusual.' That's the explainability piece. An analyst sees not just the alert but the reasoning.
>
> For evaluation, we don't just test on CICIoT2023. We also do a **cross-dataset check** — train on CICIoT2023, evaluate on a completely different dataset called TON_IoT — to measure how well the model generalises to data it wasn't trained on. And we do a third check on IoT-23, which contains real malware infections, not just lab attacks. That tells us if the model survives outside the comfort of the training distribution.
>
> Finally we do an **adversarial robustness study**. We simulate an attacker who knows the model and tries to perturb a malicious flow just enough to slip past the detector — this is called FGSM. We measure how much perturbation it takes to fool the detector. That's the stress-test that most academic work skips.
>
> And the whole thing has to run at the gateway in under ten milliseconds per flow, so we quantise the trained model and benchmark it on small hardware at the end.
>
> The timeline is 12 weeks: a couple of weeks to set up and validate the datasets, three weeks to train and tune the supervised arm, a week for the autoencoder, a week for cross-dataset evaluation, a week for SHAP, a week for the adversarial study, a week for hardware benchmarking, and the last two weeks for one stretch extension and the final write-up."

---

## How to split this between you and Yashika

You don't have to script this rigidly. A natural split:

- **Madhav** opens with Part 1 (the area).
- **Yashika** picks up Part 2 (the problem) — or you alternate paragraphs.
- **Both of you** handle Part 3 together, with one person describing the supervised arm and the other the unsupervised arm — that's a natural narrative seam.

The key thing is to *not* read in turn from a list. Let it feel like a conversation between you two about a thing you both understand.

---

## Five things to keep handy in your head (no notes)

If you only remember five facts, remember these:

1. **The path matters, not the device.** We monitor traffic at the gateway, not the endpoints, because endpoints can't be trusted.
2. **Two models, not one.** Supervised for known attacks, autoencoder for zero-days. The autoencoder learns "normal" and flags anything that doesn't fit.
3. **No payload, only metadata.** Works on encrypted traffic.
4. **CICIoT2023 is the main dataset.** TON_IoT for cross-dataset check, IoT-23 for malware check.
5. **We measure adversarial robustness.** Most papers don't.

---

## If the mentor asks something you don't know

Don't bluff. Say: *"That's a good question — we haven't pinned that down yet, can we come back to you on it?"* It's better than inventing a number.

Specific things you genuinely might not know yet and shouldn't pretend to:
- Exact hyperparameters (layer sizes, learning rate).
- Exact final model size in MB.
- Exact false-positive rate target.
- What gateway hardware they expect you to deploy on — *ask them this*.

---

## Closing line

After Part 3, hand back to the mentor with something open:

> "That's the picture. Happy to go deeper on any of it — and we'd value your input on what to prioritise in the first couple of weeks, especially around what hardware you'd want this benchmarked on."

That last bit does two things: it signals you respect their input, and it gets you the hardware answer you need for Week 10.

Good luck.
