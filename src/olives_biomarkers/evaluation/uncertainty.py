"""Monte Carlo dropout inference and uncertainty-based selective prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from olives_biomarkers.evaluation.metrics import MultiLabelMetrics
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.uncertainty")


@dataclass
class UncertaintyOutput:
    """Per-sample, per-label predictive summary from stochastic passes."""

    mean_probability: np.ndarray
    std_probability: np.ndarray
    predictive_entropy: np.ndarray
    expected_entropy: np.ndarray
    mutual_information: np.ndarray
    n_passes: int
    row_uid: np.ndarray | None = None
    patient_id: np.ndarray | None = None
    targets: np.ndarray | None = None

    def total_uncertainty(self) -> np.ndarray:
        """Mean predictive entropy across labels, one value per sample."""
        return self.predictive_entropy.mean(axis=1)

    def epistemic_uncertainty(self) -> np.ndarray:
        """Mean mutual information across labels, one value per sample."""
        return self.mutual_information.mean(axis=1)

    def to_frame(self, label_names: list[str] | None = None) -> pd.DataFrame:
        """Flatten into a per-sample table for saving with predictions."""
        n_labels = self.mean_probability.shape[1]
        names = label_names or [f"label_{i}" for i in range(n_labels)]
        data: dict[str, Any] = {}
        if self.row_uid is not None:
            data["row_uid"] = self.row_uid
        if self.patient_id is not None:
            data["patient_id"] = self.patient_id
        for index, name in enumerate(names):
            data[f"prob_{name}"] = self.mean_probability[:, index]
            data[f"std_{name}"] = self.std_probability[:, index]
            data[f"mi_{name}"] = self.mutual_information[:, index]
            if self.targets is not None:
                data[f"true_{name}"] = self.targets[:, index]
        data["total_uncertainty"] = self.total_uncertainty()
        data["epistemic_uncertainty"] = self.epistemic_uncertainty()
        return pd.DataFrame(data)


class MCDropoutInference:
    """Runs stochastic forward passes with dropout active and everything else fixed.

    The critical detail is that only ``nn.Dropout`` modules are returned to train
    mode. Batch-norm layers stay in eval mode, so they keep using their running
    statistics; letting them update would make predictions depend on batch
    composition and would not be an uncertainty estimate at all.

    Low variance is not evidence of correctness. A confidently wrong model
    produces low variance too, which is why the selective-prediction curves below
    are reported alongside.
    """

    def __init__(self, model: nn.Module, device: str = "cpu", n_passes: int = 30) -> None:
        self.model = model
        self.device = device
        self.n_passes = n_passes

    @staticmethod
    def enable_dropout(model: nn.Module) -> int:
        """Put every dropout module into train mode; return how many were found."""
        count = 0
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()
                count += 1
        return count

    @staticmethod
    def _entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Bernoulli entropy, elementwise."""
        p = np.clip(probabilities, eps, 1.0 - eps)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    @torch.no_grad()
    def run(self, loader: Any, n_passes: int | None = None) -> UncertaintyOutput:
        """Run ``n_passes`` stochastic passes over a loader.

        Returns:
            :class:`UncertaintyOutput` with mean probability, predictive standard
            deviation, predictive entropy, expected entropy and mutual information.
        """
        passes = n_passes or self.n_passes
        self.model.eval()
        n_dropout = self.enable_dropout(self.model)
        if n_dropout == 0:
            LOGGER.warning(
                "no dropout modules found; MC dropout will return zero variance. "
                "Check that model.dropout > 0."
            )
        LOGGER.info("MC dropout: %d passes with %d dropout modules active", passes, n_dropout)

        all_passes: list[np.ndarray] = []
        targets_out: list[np.ndarray] = []
        uids_out: list[np.ndarray] = []
        patients_out: list[np.ndarray] = []

        for pass_index in range(passes):
            batch_probabilities: list[np.ndarray] = []
            for batch in loader:
                image = (
                    batch["image"].to(self.device, non_blocking=True)
                    if getattr(self.model, "uses_image", True)
                    else None
                )
                clinical = (
                    batch["clinical"].to(self.device, non_blocking=True)
                    if getattr(self.model, "uses_clinical", True)
                    else None
                )
                logits = self.model(image=image, clinical=clinical)
                batch_probabilities.append(torch.sigmoid(logits).float().cpu().numpy())

                if pass_index == 0:
                    targets_out.append(batch["target"].numpy())
                    uids_out.append(batch["row_uid"].numpy())
                    patients_out.append(batch["patient_id"].numpy())
            all_passes.append(np.concatenate(batch_probabilities))

        stacked = np.stack(all_passes)  # (passes, samples, labels)
        mean_probability = stacked.mean(axis=0)
        std_probability = stacked.std(axis=0)
        predictive_entropy = self._entropy(mean_probability)
        expected_entropy = self._entropy(stacked).mean(axis=0)

        return UncertaintyOutput(
            mean_probability=mean_probability,
            std_probability=std_probability,
            predictive_entropy=predictive_entropy,
            expected_entropy=expected_entropy,
            mutual_information=predictive_entropy - expected_entropy,
            n_passes=passes,
            row_uid=np.concatenate(uids_out) if uids_out else None,
            patient_id=np.concatenate(patients_out) if patients_out else None,
            targets=np.concatenate(targets_out) if targets_out else None,
        )


class SelectivePredictor:
    """Coverage/performance analysis for abstaining on uncertain cases.

    Any operating point must be chosen on validation or calibration data. Picking
    the coverage level that happens to look best on test is the selective
    prediction equivalent of tuning on the test set.
    """

    def __init__(self, label_names: list[str] | None = None) -> None:
        self.label_names = label_names
        self.metrics = MultiLabelMetrics(label_names=label_names)

    def coverage_curve(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        uncertainty: np.ndarray,
        thresholds: np.ndarray | float = 0.5,
        coverage_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """Metrics on the retained subset at each coverage level.

        Args:
            uncertainty: One scalar per sample; larger means less certain.
            coverage_levels: Fractions of samples to retain.
        """
        levels = coverage_levels or [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        order = np.argsort(uncertainty)  # most certain first
        n = len(order)

        rows = []
        for coverage in levels:
            keep = max(1, int(round(coverage * n)))
            index = order[:keep]
            scores = self.metrics.compute(targets[index], probabilities[index], thresholds)
            rows.append(
                {
                    "coverage": coverage,
                    "n_retained": keep,
                    "n_abstained": n - keep,
                    "macro_f1": scores["macro_f1"],
                    "macro_auroc": scores["macro_auroc"],
                    "macro_auprc": scores["macro_auprc"],
                    "hamming_loss": scores["hamming_loss"],
                    "exact_match": scores["exact_match"],
                    "uncertainty_cutoff": float(uncertainty[index].max()),
                }
            )
        return pd.DataFrame(rows)

    def uncertainty_error_association(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        uncertainty: np.ndarray,
        thresholds: np.ndarray | float = 0.5,
    ) -> dict[str, float]:
        """Does uncertainty actually track error?

        Returns Spearman correlation between per-sample uncertainty and per-sample
        error rate, plus mean uncertainty for correct versus incorrect predictions.
        """
        from scipy import stats

        predictions = (np.asarray(probabilities) >= thresholds).astype(int)
        per_sample_error = (predictions != targets).mean(axis=1)

        correlation, p_value = stats.spearmanr(uncertainty, per_sample_error)
        any_wrong = per_sample_error > 0
        return {
            "spearman_r": float(correlation),
            "p_value": float(p_value),
            "mean_uncertainty_all_correct": float(uncertainty[~any_wrong].mean())
            if (~any_wrong).any()
            else float("nan"),
            "mean_uncertainty_any_wrong": float(uncertainty[any_wrong].mean())
            if any_wrong.any()
            else float("nan"),
        }

    def select_examples(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        uncertainty: np.ndarray,
        thresholds: np.ndarray | float = 0.5,
        n_per_category: int = 5,
    ) -> dict[str, np.ndarray]:
        """Indices of confidently-correct, confidently-wrong and uncertain cases.

        The confidently-wrong set is the one worth looking at: those are the
        failures that uncertainty did not catch.
        """
        predictions = (np.asarray(probabilities) >= thresholds).astype(int)
        error_rate = (predictions != targets).mean(axis=1)
        order = np.argsort(uncertainty)

        confident = order[: max(1, len(order) // 2)]
        uncertain = order[-max(1, len(order) // 2) :]
        return {
            "confidently_correct": confident[error_rate[confident] == 0][:n_per_category],
            "confidently_wrong": confident[error_rate[confident] > 0][:n_per_category],
            "uncertain_correct": uncertain[error_rate[uncertain] == 0][:n_per_category],
            "uncertain_wrong": uncertain[error_rate[uncertain] > 0][:n_per_category],
        }
