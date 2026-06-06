"""Loss functions for the three training stages."""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "flowmamba.training requires PyTorch. Install with `pip install \"torch>=2.1\"`."
    ) from exc


def focal_loss(
    logits: "torch.Tensor",
    target: "torch.Tensor",
    gamma: float = 2.0,
    alpha: Optional["torch.Tensor"] = None,
    reduction: str = "mean",
) -> "torch.Tensor":
    """Multi-class focal loss (Lin et al., 2017) for heavy class imbalance.

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights well-classified majority examples so minority attack classes
    (rare in IoT traffic) still drive the gradient.

    Parameters
    ----------
    logits : (N, C) raw classifier outputs.
    target : (N,) int class indices.
    gamma  : focusing parameter; gamma == 0 reduces to cross-entropy.
    alpha  : optional (C,) per-class weights.
    """
    log_p = F.log_softmax(logits, dim=-1)
    log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)   # (N,)
    pt = log_pt.exp()
    loss = -((1.0 - pt) ** gamma) * log_pt

    if alpha is not None:
        at = alpha.gather(0, target)
        loss = at * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def deep_svdd_loss(projected: "torch.Tensor", center: "torch.Tensor") -> "torch.Tensor":
    """One-class deep SVDD objective: mean squared distance to the centre.

    Minimising this packs benign projected embeddings into a minimum-volume
    hypersphere around ``center``. Trained on benign data only.
    """
    return torch.mean(torch.sum((projected - center) ** 2, dim=1))


def distillation_loss(
    student_logits: "torch.Tensor",
    teacher_logits: "torch.Tensor",
    temperature: float = 2.0,
) -> "torch.Tensor":
    """Learning-without-Forgetting KD term for the continual-learning update.

    Regularises the (expanded) student head toward the frozen old head's
    predictions on old-class logits, mitigating catastrophic forgetting.
    Only the columns the teacher knows about are matched.
    """
    n_old = teacher_logits.shape[1]
    s = F.log_softmax(student_logits[:, :n_old] / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)
