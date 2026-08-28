"""Patient-level bootstrap confidence intervals.

Resampling individual B-scans would treat 49 slices of one volume as 49
independent observations and produce intervals several times too narrow. Every
resample here draws **patients** with replacement and takes all of their scans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from olives_biomarkers.evaluation.metrics import MultiLabelMetrics
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.bootstrap")


@dataclass
class BootstrapResult:
    """Point estimate and percentile interval for one metric."""

    metric: str
    point_estimate: float
    lower: float
    upper: float
    n_iterations: int
    n_patients: int
    confidence: float = 0.95

    def format(self, decimals: int = 4) -> str:
        """Render as ``point [lower, upper]``."""
        return (
            f"{self.point_estimate:.{decimals}f} "
            f"[{self.lower:.{decimals}f}, {self.upper:.{decimals}f}]"
        )

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "metric": self.metric,
            "point_estimate": self.point_estimate,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "confidence": self.confidence,
            "n_iterations": self.n_iterations,
            "n_patients": self.n_patients,
        }


class PatientBootstrap:
    """Bootstrap over patients, never over scans.

    Args:
        n_iterations: Number of resamples.
        confidence: Interval width, e.g. 0.95.
        seed: RNG seed.
    """

    def __init__(self, n_iterations: int = 1000, confidence: float = 0.95, seed: int = 42) -> None:
        self.n_iterations = n_iterations
        self.confidence = confidence
        self.seed = seed

    def _resample_indices(
        self, patient_ids: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw patients with replacement and return all their row positions."""
        unique = np.unique(patient_ids)
        drawn = rng.choice(unique, size=len(unique), replace=True)
        by_patient = {p: np.flatnonzero(patient_ids == p) for p in unique}
        return np.concatenate([by_patient[p] for p in drawn])

    def run(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        patient_ids: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        metric_name: str = "metric",
    ) -> BootstrapResult:
        """Bootstrap one scalar metric.

        Args:
            metric_fn: Callable taking ``(targets, probabilities)`` and returning a float.
        """
        targets = np.asarray(targets)
        probabilities = np.asarray(probabilities)
        patient_ids = np.asarray(patient_ids)
        rng = np.random.default_rng(self.seed)

        point = float(metric_fn(targets, probabilities))
        samples: list[float] = []
        for _ in range(self.n_iterations):
            index = self._resample_indices(patient_ids, rng)
            try:
                samples.append(float(metric_fn(targets[index], probabilities[index])))
            except ValueError:
                # A resample can omit every positive of a rare label.
                samples.append(float("nan"))

        valid = np.asarray(samples, dtype=float)
        valid = valid[~np.isnan(valid)]
        alpha = (1.0 - self.confidence) / 2.0
        lower = float(np.percentile(valid, 100 * alpha)) if len(valid) else float("nan")
        upper = float(np.percentile(valid, 100 * (1 - alpha))) if len(valid) else float("nan")

        if len(valid) < self.n_iterations:
            LOGGER.warning(
                "%d/%d bootstrap resamples were undefined for '%s' (rare label with no positives "
                "in the resample); the interval uses the remainder",
                self.n_iterations - len(valid),
                self.n_iterations,
                metric_name,
            )

        return BootstrapResult(
            metric=metric_name,
            point_estimate=point,
            lower=lower,
            upper=upper,
            n_iterations=self.n_iterations,
            n_patients=int(len(np.unique(patient_ids))),
            confidence=self.confidence,
        )

    def run_many(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        patient_ids: np.ndarray,
        thresholds: np.ndarray | float = 0.5,
        label_names: list[str] | None = None,
        metrics: tuple[str, ...] = ("macro_f1", "macro_auroc", "macro_auprc"),
    ) -> pd.DataFrame:
        """Bootstrap several aggregate metrics at once."""
        calculator = MultiLabelMetrics(label_names=label_names)
        rows = []
        for name in metrics:

            def metric_fn(y: np.ndarray, p: np.ndarray, _name: str = name) -> float:
                return calculator.compute(y, p, thresholds)[_name]

            result = self.run(targets, probabilities, patient_ids, metric_fn, metric_name=name)
            rows.append(result.to_dict())
            LOGGER.info("  %-14s %s", name, result.format())
        return pd.DataFrame(rows)
