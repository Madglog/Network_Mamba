"""Central configuration objects.

Configs are plain dataclasses so the codebase has no hard dependency on a YAML
parser. `load_yaml` is provided for convenience and overlays a YAML file onto the
dataclass defaults when PyYAML is installed; the CLI falls back to the dataclass
defaults plus command-line overrides otherwise.

Two named presets capture the two operating modes from the proposal:

  * default mode -- low power, always on: small encoder, short sequence
    (K=8), anomaly head only, no SHAP.
  * strong mode  -- admin-switched, full pipeline: full encoder, K=32, both
    heads, SHAP on every alert.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """How a flow is turned into model input.

    A flow is the first ``max_packets`` packets of a connection, each described
    by the per-packet feature vector defined in :mod:`flowmamba.data.features`.
    """

    max_packets: int = 32          # K -- sequence length (strong mode)
    n_features: int = 13           # per-packet feature dimension (see features.py)
    pad_value: float = 0.0         # value used to pad short flows
    mask_prob: float = 0.15        # fraction of packets masked during SSL pre-training

    # Synthetic-data knobs (used by the runnable demo; real data overrides these)
    n_classes: int = 8             # benign + 7 attack categories
    seed: int = 1337


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Mamba encoder + heads geometry."""

    # Encoder
    d_model: int = 128             # hidden / embedding width
    n_layers: int = 4              # number of Mamba blocks
    d_state: int = 16              # SSM state dimension (N)
    d_conv: int = 4                # depthwise causal conv width
    expand: int = 2               # inner expansion factor (d_inner = expand * d_model)
    dt_rank: Optional[int] = None  # rank of delta projection; None -> ceil(d_model/16)

    # Heads
    n_classes: int = 8             # classifier outputs (benign + 7 attack categories)
    proj_dim: int = 32             # anomaly projection dimension (R^d_model -> R^proj_dim)
    head_hidden: int = 64          # classifier MLP hidden width

    # Anomaly head
    svdd_percentile: float = 99.0  # threshold percentile on benign scores (1% FPR target)

    def resolved_dt_rank(self) -> int:
        if self.dt_rank is not None:
            return self.dt_rank
        return max(1, -(-self.d_model // 16))  # ceil(d_model / 16)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Optimisation settings shared across the three training stages."""

    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs_pretrain: int = 10      # Stage A -- masked-pattern SSL
    epochs_finetune: int = 15      # Stage B -- supervised fine-tune
    epochs_anomaly: int = 10       # Stage C -- projection + deep SVDD
    grad_clip: float = 1.0

    # Focal loss (Stage B)
    focal_gamma: float = 2.0
    focal_alpha: Optional[List[float]] = None  # per-class weights; None -> uniform

    # Continual learning (the loop)
    replay_per_class: int = 200    # labelled flows persisted per existing class

    device: str = "auto"           # "auto" | "cpu" | "cuda" | "mps"
    seed: int = 1337
    out_dir: str = "artifacts"


# --------------------------------------------------------------------------- #
# Mode presets
# --------------------------------------------------------------------------- #
def strong_mode_config() -> Dict[str, Any]:
    """Full pipeline: full Mamba, K=32, both heads, SHAP enabled."""
    return {
        "data": DataConfig(max_packets=32),
        "model": ModelConfig(d_model=128, n_layers=4),
        "mode": "strong",
        "use_classifier": True,
        "use_anomaly": True,
        "use_shap": True,
    }


def default_mode_config() -> Dict[str, Any]:
    """Low-power always-on: small Mamba, K=8, anomaly head only, no SHAP."""
    return {
        "data": DataConfig(max_packets=8),
        "model": ModelConfig(d_model=64, n_layers=2),
        "mode": "default",
        "use_classifier": False,
        "use_anomaly": True,
        "use_shap": False,
    }


# --------------------------------------------------------------------------- #
# YAML overlay (optional)
# --------------------------------------------------------------------------- #
def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a nested dict. Requires PyYAML."""
    try:
        import yaml  # noqa: WPS433 (optional dependency)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyYAML is not installed; either `pip install pyyaml` or use the "
            "dataclass defaults via the CLI flags."
        ) from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge(cfg: Any, overrides: Dict[str, Any]) -> Any:
    """Return a copy of a dataclass config with ``overrides`` applied."""
    valid = {k: v for k, v in overrides.items() if k in asdict(cfg)}
    return replace(cfg, **valid)
