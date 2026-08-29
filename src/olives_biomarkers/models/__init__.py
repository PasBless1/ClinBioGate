"""Models: encoders, heads, baselines A-C, and the clinical fusion variants."""

from olives_biomarkers.models.baselines import (
    BaseBiomarkerModel,
    ClinicalOnlyModel,
    ConcatFusionModel,
    OCTOnlyModel,
)
from olives_biomarkers.models.encoders import ClinicalEncoder, ImageEncoder
from olives_biomarkers.models.fusion import (
    BoundedFiLMFusionModel,
    ClinicalGate,
    GatedFusionModel,
    ResidualLogitFusionModel,
)
from olives_biomarkers.models.heads import MultiLabelHead
from olives_biomarkers.models.registry import ModelFactory

__all__ = [
    "BaseBiomarkerModel",
    "ClinicalOnlyModel",
    "OCTOnlyModel",
    "ConcatFusionModel",
    "GatedFusionModel",
    "BoundedFiLMFusionModel",
    "ResidualLogitFusionModel",
    "ClinicalGate",
    "ImageEncoder",
    "ClinicalEncoder",
    "MultiLabelHead",
    "ModelFactory",
]
