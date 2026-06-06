"""FGSM adversarial study on the per-packet flow sequence.

Fast Gradient Sign Method perturbs the (continuous, standardised) per-packet
features in the direction that most increases the loss, with the perturbation
clipped to valid feature ranges. We report detection rate vs. perturbation
magnitude epsilon for both heads -- the proposal's adversarial-robustness check.

Requires PyTorch.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import torch
import torch.nn.functional as F

from flowmamba.data.dataset import padding_mask
from flowmamba.models.detector import Detector
from flowmamba.utils import resolve_device


def _valid_mask(flow, length):
    return padding_mask(length, flow.shape[1]) if length is not None else None


def fgsm_perturb(
    model: Detector,
    flow: "torch.Tensor",
    length: Optional["torch.Tensor"],
    target: "torch.Tensor",
    epsilon: float,
    feature_min: Optional["torch.Tensor"] = None,
    feature_max: Optional["torch.Tensor"] = None,
    attack_head: str = "classifier",
) -> "torch.Tensor":
    """Return an FGSM-perturbed copy of ``flow``.

    attack_head:
      * "classifier" -- ascend the focal/CE loss away from the true label.
      * "anomaly"    -- descend the SVDD score (push attacks toward "normal"),
        the evasion an attacker actually wants against the one-class head.
    """
    flow = flow.clone().detach().requires_grad_(True)
    mask = _valid_mask(flow, length)
    z = model.embed(flow, length)

    if attack_head == "classifier":
        logits = model.classifier(z)
        loss = F.cross_entropy(logits, target)
        grad_sign = torch.autograd.grad(loss, flow)[0].sign()
        adv = flow + epsilon * grad_sign            # ascend loss
    elif attack_head == "anomaly":
        score = model.anomaly.scores(model.project(z)).mean()
        grad_sign = torch.autograd.grad(score, flow)[0].sign()
        adv = flow - epsilon * grad_sign            # descend anomaly score (evade)
    else:
        raise ValueError("attack_head must be 'classifier' or 'anomaly'")

    adv = adv.detach()
    if feature_min is not None and feature_max is not None:
        adv = torch.max(torch.min(adv, feature_max), feature_min)
    if mask is not None:
        adv = adv * mask.unsqueeze(-1)              # keep padding zeroed
    return adv


def fgsm_curve(
    model: Detector,
    flows: np.ndarray,
    labels: np.ndarray,
    lengths: Optional[np.ndarray],
    epsilons: List[float],
    attack_head: str = "classifier",
    device_name: str = "auto",
) -> Dict[float, Dict[str, float]]:
    """Detection rate vs. epsilon, evaluated on *attack* flows only.

    For each epsilon, perturbs the attack flows and measures the fraction still
    caught (by the chosen head). epsilon == 0 is the clean baseline.
    """
    from flowmamba.eval.metrics import binary_rates

    device = resolve_device(device_name)
    model.to(device).eval()

    flow_t = torch.as_tensor(flows, dtype=torch.float32, device=device)
    label_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    len_t = (
        torch.as_tensor(lengths, dtype=torch.long, device=device)
        if lengths is not None
        else None
    )
    is_attack = (label_t != 0).cpu().numpy()

    # Per-feature valid range, estimated from the clean data (standardised space).
    fmin = flow_t.reshape(-1, flow_t.shape[-1]).min(0).values
    fmax = flow_t.reshape(-1, flow_t.shape[-1]).max(0).values

    out: Dict[float, Dict[str, float]] = {}
    for eps in epsilons:
        if eps == 0:
            adv = flow_t
        else:
            adv = fgsm_perturb(
                model, flow_t, len_t, label_t, eps, fmin, fmax, attack_head
            )
        with torch.no_grad():
            result = model.predict(adv, len_t, mode="strong")
            flagged = result.alert.cpu().numpy()
        out[eps] = binary_rates(is_attack, flagged)
    return out
