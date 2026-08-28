"""Evaluation: metrics, bootstrap CIs, calibration, uncertainty, gating, explainability."""

from olives_biomarkers.evaluation.bootstrap import BootstrapResult, PatientBootstrap
from olives_biomarkers.evaluation.calibration import CalibrationEvaluator, TemperatureScaler
from olives_biomarkers.evaluation.explainability import AttentionSanityChecker, CamResult, GradCAM
from olives_biomarkers.evaluation.gating import GateAnalyzer
from olives_biomarkers.evaluation.metrics import MultiLabelMetrics, ThresholdOptimizer, ThresholdSet
from olives_biomarkers.evaluation.plots import ResultsPlotter
from olives_biomarkers.evaluation.uncertainty import (
    MCDropoutInference,
    SelectivePredictor,
    UncertaintyOutput,
)

__all__ = [
    "MultiLabelMetrics",
    "ThresholdOptimizer",
    "ThresholdSet",
    "PatientBootstrap",
    "BootstrapResult",
    "TemperatureScaler",
    "CalibrationEvaluator",
    "MCDropoutInference",
    "UncertaintyOutput",
    "SelectivePredictor",
    "GradCAM",
    "CamResult",
    "AttentionSanityChecker",
    "GateAnalyzer",
    "ResultsPlotter",
]
