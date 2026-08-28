"""Probability calibration and reliability assessment.

Calibration parameters are fitted on a patient-disjoint calibration partition,
never on test data and never on data that fitted the model weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from olives_biomarkers.utils.io import JsonIO
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.calibration")


@dataclass
class CalibrationState:
    """Fitted per-label temperatures."""

    temperatures: list[float]
    label_names: list[str]
    fitted_on: str = "calibration"
    n_fit_rows: int = 0


class TemperatureScaler:
    """Per-label temperature scaling of logits.

    A single scalar per label divides the logit before the sigmoid, which fixes
    systematic over- or under-confidence without changing the ranking, so AUROC
    is untouched while Brier score and ECE improve.

    Args:
        max_iterations: Optimiser steps per label.
        learning_rate: Step size for the log-temperature.
    """

    def __init__(self, max_iterations: int = 200, learning_rate: float = 0.05) -> None:
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.state: CalibrationState | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether temperatures have been fitted."""
        return self.state is not None

    def fit(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        label_names: list[str] | None = None,
    ) -> TemperatureScaler:
        """Fit one temperature per label by minimising NLL on the calibration set.

        Args:
            logits: Raw logits from the calibration partition.
            targets: Binary targets, same shape.
        """
        import torch

        logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        targets_t = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
        n_labels = logits_t.shape[1]
        names = label_names or [f"label_{i}" for i in range(n_labels)]

        temperatures: list[float] = []
        for index in range(n_labels):
            y = targets_t[:, index]
            if y.sum() == 0 or y.sum() == len(y):
                LOGGER.warning(
                    "label '%s' is single-class in the calibration fold; temperature fixed at 1.0",
                    names[index],
                )
                temperatures.append(1.0)
                continue

            log_temperature = torch.zeros(1, requires_grad=True)
            optimizer = torch.optim.Adam([log_temperature], lr=self.learning_rate)
            z = logits_t[:, index]
            for _ in range(self.max_iterations):
                optimizer.zero_grad()
                scaled = z / torch.exp(log_temperature)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled, y)
                loss.backward()
                optimizer.step()
            temperatures.append(float(torch.exp(log_temperature).detach()))

        self.state = CalibrationState(
            temperatures=temperatures,
            label_names=list(names),
            n_fit_rows=int(len(targets)),
        )
        LOGGER.info(
            "fitted temperatures on %d calibration rows: %s",
            len(targets),
            [round(t, 3) for t in temperatures],
        )
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply the fitted temperatures and return calibrated probabilities."""
        if self.state is None:
            raise RuntimeError("call fit() on the calibration partition first")
        temperatures = np.asarray(self.state.temperatures, dtype=float)
        scaled = np.asarray(logits, dtype=float) / temperatures
        return 1.0 / (1.0 + np.exp(-np.clip(scaled, -50, 50)))

    def save(self, path: str) -> None:
        """Persist fitted temperatures."""
        if self.state is None:
            raise RuntimeError("nothing to save; scaler is not fitted")
        JsonIO.write(vars(self.state), path)

    @classmethod
    def load(cls, path: str) -> TemperatureScaler:
        """Restore a fitted scaler."""
        scaler = cls()
        scaler.state = CalibrationState(**JsonIO.read(path))
        return scaler


class CalibrationEvaluator:
    """Expected calibration error, Brier score and reliability curves."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def expected_calibration_error(
        self, targets: np.ndarray, probabilities: np.ndarray
    ) -> float:
        """Binned ECE for a single label."""
        targets = np.asarray(targets, dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        error, total = 0.0, len(probabilities)
        if total == 0:
            return float("nan")
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (probabilities > low) & (probabilities <= high)
            if not mask.any():
                continue
            confidence = probabilities[mask].mean()
            accuracy = targets[mask].mean()
            error += (mask.sum() / total) * abs(confidence - accuracy)
        return float(error)

    def reliability_curve(
        self, targets: np.ndarray, probabilities: np.ndarray
    ) -> pd.DataFrame:
        """Bin-wise mean confidence versus observed frequency."""
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        rows = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (probabilities > low) & (probabilities <= high)
            rows.append(
                {
                    "bin_lower": low,
                    "bin_upper": high,
                    "n": int(mask.sum()),
                    "mean_confidence": float(probabilities[mask].mean()) if mask.any() else np.nan,
                    "observed_frequency": float(targets[mask].mean()) if mask.any() else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def evaluate(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        label_names: list[str] | None = None,
        min_positives: int = 20,
    ) -> pd.DataFrame:
        """Per-label ECE and Brier score.

        Labels with fewer than ``min_positives`` positives are marked
        ``reliable=False``: their reliability diagrams are too noisy to read.
        """
        targets = np.asarray(targets)
        probabilities = np.asarray(probabilities)
        names = label_names or [f"label_{i}" for i in range(targets.shape[1])]

        rows = []
        for index, name in enumerate(names):
            y = targets[:, index]
            p = probabilities[:, index]
            positives = int(y.sum())
            rows.append(
                {
                    "label": name,
                    "n_positive": positives,
                    "ece": self.expected_calibration_error(y, p),
                    "brier": float(np.mean((p - y) ** 2)),
                    "mean_predicted": float(p.mean()),
                    "observed_rate": float(y.mean()),
                    "reliable": positives >= min_positives,
                }
            )
        return pd.DataFrame(rows)

    def compare(
        self,
        targets: np.ndarray,
        before: np.ndarray,
        after: np.ndarray,
        label_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Side-by-side pre/post-calibration ECE and Brier score."""
        pre = self.evaluate(targets, before, label_names).add_suffix("_pre")
        post = self.evaluate(targets, after, label_names).add_suffix("_post")
        merged = pd.concat([pre, post], axis=1)
        merged["label"] = merged["label_pre"]
        merged["ece_delta"] = merged["ece_post"] - merged["ece_pre"]
        merged["brier_delta"] = merged["brier_post"] - merged["brier_pre"]
        return merged[
            ["label", "n_positive_pre", "ece_pre", "ece_post", "ece_delta",
             "brier_pre", "brier_post", "brier_delta", "reliable_pre"]
        ].rename(columns={"n_positive_pre": "n_positive", "reliable_pre": "reliable"})
