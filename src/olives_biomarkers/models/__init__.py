"""Models: encoders, heads, baselines A-C and the proposed gated fusion (D)."""

from olives_biomarkers.models.baselines import (
    BaseBiomarkerModel,
    ClinicalOnlyModel,
    ConcatFusionModel,
    OCTOnlyModel,
)
from olives_biomarkers.models.encoders import ClinicalEncoder, ImageEncoder
from olives_biomarkers.models.fusion import ClinicalGate, GatedFusionModel
from olives_biomarkers.models.heads import MultiLabelHead
from olives_biomarkers.models.registry import ModelFactory

__all__ = [
    "BaseBiomarkerModel",
    "ClinicalOnlyModel",
    "OCTOnlyModel",
    "ConcatFusionModel",
    "GatedFusionModel",
    "ClinicalGate",
    "ImageEncoder",
    "ClinicalEncoder",
    "MultiLabelHead",
    "ModelFactory",
]
