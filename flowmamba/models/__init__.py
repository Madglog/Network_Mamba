"""Model layer: Mamba encoder, heads, and the assembled detector.

All modules here require PyTorch and import it lazily at module load.
"""

from flowmamba.models.mamba import MambaBlock, MambaEncoder, ReconstructionHead
from flowmamba.models.heads import ClassifierHead, ProjectionHead, DeepSVDDHead
from flowmamba.models.detector import Detector

__all__ = [
    "MambaBlock",
    "MambaEncoder",
    "ReconstructionHead",
    "ClassifierHead",
    "ProjectionHead",
    "DeepSVDDHead",
    "Detector",
]
