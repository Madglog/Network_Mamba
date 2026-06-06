"""The assembled gateway detector: shared Mamba encoder + two heads.

A single encoder produces the flow embedding z; the classifier head scores known
categories and the (projected) deep-SVDD head scores distance from normal. The
same artifacts serve both operating modes -- ``default`` engages only the anomaly
head, ``strong`` engages both plus SHAP downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "flowmamba.models requires PyTorch. Install with `pip install \"torch>=2.1\"`."
    ) from exc

from flowmamba.config import ModelConfig
from flowmamba.data.dataset import padding_mask
from flowmamba.models.heads import ClassifierHead, DeepSVDDHead, ProjectionHead
from flowmamba.models.mamba import MambaEncoder, ReconstructionHead


@dataclass
class DetectorOutput:
    """Per-flow detector result."""

    embedding: "torch.Tensor"                 # z, (b, d_model)
    anomaly_score: "torch.Tensor"             # squared distance to SVDD centre, (b,)
    is_anomaly: "torch.Tensor"                # bool, (b,)
    class_logits: Optional["torch.Tensor"]    # (b, n_classes) or None in default mode
    alert: "torch.Tensor"                     # bool, (b,) -- either head fired


class Detector(nn.Module):
    """Shared encoder + classifier + projected deep-SVDD anomaly head."""

    def __init__(self, cfg: ModelConfig, n_features: int):
        super().__init__()
        self.cfg = cfg
        self.n_features = n_features
        self.encoder = MambaEncoder(cfg, n_features)
        self.classifier = ClassifierHead(cfg)
        self.projection = ProjectionHead(cfg)
        self.anomaly = DeepSVDDHead(cfg)
        # Reconstruction head only exists during Stage A; kept for warm-restart.
        self.recon = ReconstructionHead(cfg, n_features)

    # ------------------------------------------------------------------ #
    def _mask(self, flow, length):
        if length is None:
            return None
        return padding_mask(length, flow.shape[1])

    def embed(self, flow, length=None):
        """Encode to z. flow: (b, K, F) -> (b, d_model)."""
        return self.encoder.encode(flow, self._mask(flow, length))

    def reconstruct(self, flow, length=None):
        """Stage A forward: per-position feature predictions, (b, K, F)."""
        mask = self._mask(flow, length)
        hidden = self.encoder(flow, mask)
        return self.recon(hidden)

    def project(self, z):
        return self.projection(z)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, flow, length=None, mode: str = "strong") -> DetectorOutput:
        """Run inference in ``default`` or ``strong`` mode.

        default -- anomaly head only (low power, always on).
        strong  -- both heads (classifier + projected anomaly head).
        """
        z = self.embed(flow, length)
        proj = self.projection(z)
        ascore = self.anomaly.scores(proj)
        is_anom = ascore > self.anomaly.radius

        logits = None
        alert = is_anom
        if mode == "strong":
            logits = self.classifier(z)
            pred = logits.argmax(dim=-1)
            # Class 0 is benign: a non-benign prediction is also an alert.
            class_alert = pred != 0
            alert = is_anom | class_alert
        elif mode != "default":
            raise ValueError(f"mode must be 'default' or 'strong', got {mode!r}")

        return DetectorOutput(
            embedding=z,
            anomaly_score=ascore,
            is_anomaly=is_anom,
            class_logits=logits,
            alert=alert,
        )

    # ------------------------------------------------------------------ #
    def parameter_groups(self) -> Dict[str, list]:
        """Convenience grouping for stage-specific optimisers / freezing."""
        return {
            "encoder": list(self.encoder.parameters()),
            "classifier": list(self.classifier.parameters()),
            "anomaly": list(self.projection.parameters()) + list(self.anomaly.parameters()),
            "recon": list(self.recon.parameters()),
        }

    def freeze_encoder(self, frozen: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not frozen

    # ------------------------------------------------------------------ #
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "cfg": self.cfg.__dict__,
            "n_features": self.n_features,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "Detector":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ModelConfig(**payload["cfg"])
        model = cls(cfg, payload["n_features"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
