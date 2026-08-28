"""Multilabel metrics and threshold selection.

Rare labels are the rule in OLIVES, not the exception: several biomarkers have
under 20 positives. Every metric here returns ``nan`` for a label with no
positive (or no negative) examples rather than inventing a score, and macro
averages skip those labels while reporting how many were skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.metrics")


class MultiLabelMetrics:
    """Computes discrimination, calibration and per-label metrics.

    Args:
        label_names: Names in target order; defaults to ``label_0 ...``.
    """

    def __init__(self, label_names: list[str] | None = None) -> None:
        self.label_names = label_names

    def _names(self, n_labels: int) -> list[str]:
        if self.label_names and len(self.label_names) == n_labels:
            return list(self.label_names)
        return [f"label_{i}" for i in range(n_labels)]

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """AUROC, or nan when the label is single-class in this partition."""
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(skm.roc_auc_score(y_true, y_score))

    @staticmethod
    def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Average precision, or nan when the label has no positives."""
        if y_true.sum() == 0:
            return float("nan")
        return float(skm.average_precision_score(y_true, y_score))

    @staticmethod
    def _nanmean(values: list[float]) -> float:
        """Mean ignoring nan; nan when every entry is nan."""
        array = np.asarray(values, dtype=float)
        return float(np.nanmean(array)) if not np.all(np.isnan(array)) else float("nan")

    # ------------------------------------------------------------------
    def per_label(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        thresholds: np.ndarray | float = 0.5,
    ) -> pd.DataFrame:
        """Per-label metric table.

        Args:
            targets: Binary array ``(n_samples, n_labels)``.
            probabilities: Predicted probabilities, same shape.
            thresholds: Scalar or per-label decision thresholds.
        """
        targets = np.asarray(targets)
        probabilities = np.asarray(probabilities)
        n_labels = targets.shape[1]
        names = self._names(n_labels)
        threshold_array = (
            np.full(n_labels, float(thresholds))
            if np.isscalar(thresholds)
            else np.asarray(thresholds, dtype=float)
        )

        rows: list[dict[str, Any]] = []
        for index in range(n_labels):
            y_true = targets[:, index]
            y_score = probabilities[:, index]
            y_pred = (y_score >= threshold_array[index]).astype(int)

            positives = int(y_true.sum())
            negatives = int(len(y_true) - positives)
            degenerate = positives == 0 or negatives == 0

            true_pos = int(((y_pred == 1) & (y_true == 1)).sum())
            false_pos = int(((y_pred == 1) & (y_true == 0)).sum())
            false_neg = int(((y_pred == 0) & (y_true == 1)).sum())
            true_neg = int(((y_pred == 0) & (y_true == 0)).sum())

            rows.append(
                {
                    "label": names[index],
                    "n_positive": positives,
                    "n_negative": negatives,
                    "threshold": float(threshold_array[index]),
                    "auroc": self._safe_auroc(y_true, y_score),
                    "auprc": self._safe_auprc(y_true, y_score),
                    "f1": float("nan") if degenerate else float(skm.f1_score(y_true, y_pred, zero_division=0)),
                    "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
                    "sensitivity": float(true_pos / positives) if positives else float("nan"),
                    "specificity": float(true_neg / negatives) if negatives else float("nan"),
                    "brier": float(np.mean((y_score - y_true) ** 2)),
                    "tp": true_pos,
                    "fp": false_pos,
                    "fn": false_neg,
                    "tn": true_neg,
                    "degenerate": degenerate,
                }
            )
        return pd.DataFrame(rows)

    def compute(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        thresholds: np.ndarray | float = 0.5,
    ) -> dict[str, float]:
        """Aggregate metrics, with the primary metric being macro F1.

        Returns:
            Scalar metrics including ``macro_f1``, ``micro_f1``, ``macro_auroc``,
            ``macro_auprc``, ``exact_match``, ``hamming_loss``, ``brier`` and
            ``n_degenerate_labels``.
        """
        table = self.per_label(targets, probabilities, thresholds)
        targets = np.asarray(targets)
        n_labels = targets.shape[1]
        threshold_array = (
            np.full(n_labels, float(thresholds))
            if np.isscalar(thresholds)
            else np.asarray(thresholds, dtype=float)
        )
        predictions = (np.asarray(probabilities) >= threshold_array).astype(int)

        return {
            "macro_f1": self._nanmean(table["f1"].tolist()),
            "micro_f1": float(skm.f1_score(targets, predictions, average="micro", zero_division=0)),
            "macro_auroc": self._nanmean(table["auroc"].tolist()),
            "macro_auprc": self._nanmean(table["auprc"].tolist()),
            "macro_precision": self._nanmean(table["precision"].tolist()),
            "macro_recall": self._nanmean(table["recall"].tolist()),
            "macro_specificity": self._nanmean(table["specificity"].tolist()),
            "exact_match": float((predictions == targets).all(axis=1).mean()),
            "hamming_loss": float(skm.hamming_loss(targets, predictions)),
            "brier": float(np.mean((np.asarray(probabilities) - targets) ** 2)),
            "n_degenerate_labels": int(table["degenerate"].sum()),
            "n_samples": int(len(targets)),
        }


@dataclass
class ThresholdSet:
    """Per-label decision thresholds, frozen after validation fitting."""

    thresholds: np.ndarray
    label_names: list[str]
    fitted_on: str = "validation"
    objective: str = "f1"

    def to_frame(self) -> pd.DataFrame:
        """Thresholds as a table for saving with the run."""
        return pd.DataFrame({"label": self.label_names, "threshold": self.thresholds})

    def as_array(self) -> np.ndarray:
        """The raw threshold vector."""
        return np.asarray(self.thresholds, dtype=float)


class ThresholdOptimizer:
    """Selects one decision threshold per label on validation data.

    0.5 is rarely optimal under heavy class imbalance. Thresholds are fitted on
    the validation partition and then frozen; test evaluation must reuse them.
    """

    def __init__(self, objective: str = "f1", grid: np.ndarray | None = None) -> None:
        if objective not in {"f1", "youden"}:
            raise ValueError(f"unknown objective {objective!r}")
        self.objective = objective
        self.grid = grid if grid is not None else np.linspace(0.01, 0.99, 99)

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.objective == "f1":
            return float(skm.f1_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        return float(sensitivity + specificity - 1.0)

    def fit(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        label_names: list[str] | None = None,
    ) -> ThresholdSet:
        """Find the best threshold per label on the validation partition."""
        targets = np.asarray(targets)
        probabilities = np.asarray(probabilities)
        n_labels = targets.shape[1]
        names = label_names or [f"label_{i}" for i in range(n_labels)]

        thresholds = np.full(n_labels, 0.5)
        for index in range(n_labels):
            y_true = targets[:, index]
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                LOGGER.warning(
                    "label '%s' is single-class in the validation fold; keeping threshold 0.5",
                    names[index],
                )
                continue
            y_score = probabilities[:, index]
            scores = [self._score(y_true, (y_score >= t).astype(int)) for t in self.grid]
            thresholds[index] = float(self.grid[int(np.argmax(scores))])

        return ThresholdSet(
            thresholds=thresholds,
            label_names=list(names),
            fitted_on="validation",
            objective=self.objective,
        )
