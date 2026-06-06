"""flowmamba -- Flow-pattern anomaly detection at the IoT gateway.

A pre-trained Mamba encoder feeds two heads: a supervised classifier for known
attack categories and a one-class deep SVDD scorer for zero-day attacks. The
detector is built to run on gateway-class hardware (Raspberry-Pi-scale) at the
edge of a smart-home / field-sensor network.

The package is import-safe without PyTorch: only the deep-learning modules
(`flowmamba.models`, `flowmamba.training`, parts of `flowmamba.eval`) require
torch, and they import it lazily. The data, preprocessing and classical-baseline
paths run on numpy / scikit-learn / xgboost alone.
"""

from flowmamba.config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    default_mode_config,
    strong_mode_config,
)

__version__ = "0.1.0"

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "default_mode_config",
    "strong_mode_config",
    "__version__",
]
