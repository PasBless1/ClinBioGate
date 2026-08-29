"""Paired comparison between two models on the same patients.

Comparing two independently computed confidence intervals and asking whether
they overlap is the wrong test, and the least powerful one available. Most of
the width of each interval comes from patient difficulty -- some patients are
simply harder than others -- and that component is *shared* by both models, so
it cancels when the difference itself is resampled. Two heavily overlapping
intervals routinely conceal a difference that is consistent in every resample.

Every resample here draws patients with replacement once and scores **both**
arms on that same draw. That is what makes it paired, and it is why the interval
on the difference is much tighter than either marginal interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from olives_biomarkers.evaluation.metrics import MultiLabelMetrics
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.comparison")


class PairedComparisonError(RuntimeError):
    """Raised when two arms cannot be compared as a matched pair."""


@dataclass
class PairedDifferenceResult:
    """Bootstrap interval for the difference between two models on one metric."""

    metric: str
    arm_a: str
    arm_b: str
    estimate_a: float
    estimate_b: float
    difference: float
    lower: float
    upper: float
    p_two_sided: float
    n_iterations: int
    n_patients: int
    confidence: float = 0.95

    @property
    def supported(self) -> bool:
        """Whether the interval on the difference excludes zero."""
        if not (np.isfinite(self.lower) and np.isfinite(self.upper)):
            return False
        return self.lower > 0.0 or self.upper < 0.0

    @property
    def conclusion(self) -> str:
        """Plain reading of the interval."""
        if not self.supported:
            return "no supported difference"
        return f"{self.arm_a} > {self.arm_b}" if self.difference > 0 else f"{self.arm_b} > {self.arm_a}"

    def format(self, decimals: int = 4) -> str:
        """Render as ``+diff [lower, upper]  p=...``."""
        return (
            f"{self.difference:+.{decimals}f} "
            f"[{self.lower:+.{decimals}f}, {self.upper:+.{decimals}f}]  p={self.p_two_sided:.4f}"
        )

    def to_dict(self) -> dict[str, float | str | int | bool]:
        return {
            "metric": self.metric,
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "estimate_a": self.estimate_a,
            "estimate_b": self.estimate_b,
            "difference": self.difference,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "p_two_sided": self.p_two_sided,
            "supported": self.supported,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "n_iterations": self.n_iterations,
            "n_patients": self.n_patients,
        }


class PairedPatientBootstrap:
    """Bootstrap the difference between two arms over patients, not scans.

    Args:
        n_iterations: Number of resamples.
        confidence: Interval width, e.g. 0.95.
        seed: RNG seed.

    Example:
        >>> test = PairedPatientBootstrap()
        >>> result = test.run(targets, fusion_probs, oct_probs, patients,
        ...                   metric_fn, arm_a="fusion", arm_b="oct_only")
        >>> result.supported, result.conclusion
    """

    def __init__(self, n_iterations: int = 2000, confidence: float = 0.95, seed: int = 42) -> None:
        self.n_iterations = n_iterations
        self.confidence = confidence
        self.seed = seed

    # ------------------------------------------------------------------
    @staticmethod
    def align(row_uid_a: np.ndarray, row_uid_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Index arrays putting both arms into the same row order.

        Alignment is by identifier, never by position: two runs of the same split
        can emit rows in a different order and still be perfectly comparable.

        Raises:
            PairedComparisonError: The arms were not evaluated on the same rows.
        """
        a = np.asarray(row_uid_a)
        b = np.asarray(row_uid_b)
        if a.shape != b.shape or set(a.tolist()) != set(b.tolist()):
            symmetric_difference = set(a.tolist()) ^ set(b.tolist())
            raise PairedComparisonError(
                "a paired comparison needs both arms evaluated on identical rows; got "
                f"{len(a)} and {len(b)} rows, {len(symmetric_difference)} not shared"
            )
        return np.argsort(a), np.argsort(b)

    def _resample_indices(self, patient_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw patients with replacement and return all of their row positions."""
        unique = np.unique(patient_ids)
        drawn = rng.choice(unique, size=len(unique), replace=True)
        by_patient = {patient: np.flatnonzero(patient_ids == patient) for patient in unique}
        return np.concatenate([by_patient[patient] for patient in drawn])

    def _interval(self, differences: np.ndarray, metric_name: str) -> tuple[float, float, float]:
        """Percentile interval and two-sided bootstrap p-value."""
        valid = differences[~np.isnan(differences)]
        if len(valid) < len(differences):
            LOGGER.warning(
                "%d/%d paired resamples were undefined for '%s' (a rare label lost all its "
                "positives); the interval uses the remainder",
                len(differences) - len(valid),
                len(differences),
                metric_name,
            )
        if not len(valid):
            return float("nan"), float("nan"), float("nan")

        alpha = (1.0 - self.confidence) / 2.0
        lower = float(np.percentile(valid, 100 * alpha))
        upper = float(np.percentile(valid, 100 * (1 - alpha)))
        # How often the difference lands on the other side of zero, doubled. The
        # smallest value resolvable with N resamples is 1/N, so it is floored
        # there rather than reported as an impossible exact zero.
        tail = min(float(np.mean(valid <= 0.0)), float(np.mean(valid >= 0.0)))
        p_value = min(1.0, 2.0 * max(tail, 1.0 / len(valid)))
        return lower, upper, p_value

    # ------------------------------------------------------------------
    def run(
        self,
        targets: np.ndarray,
        probabilities_a: np.ndarray,
        probabilities_b: np.ndarray,
        patient_ids: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        metric_name: str = "metric",
        arm_a: str = "a",
        arm_b: str = "b",
        metric_fn_b: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> PairedDifferenceResult:
        """Bootstrap ``metric(a) - metric(b)`` over patients.

        Args:
            targets: Shared ground truth, ``(n_rows, n_labels)``.
            probabilities_a: First arm's predictions, in the same row order.
            probabilities_b: Second arm's predictions, in the same row order.
            patient_ids: Patient of each row; the resampling unit.
            metric_fn: Callable taking ``(targets, probabilities)``.
            metric_fn_b: Separate callable for the second arm. Needed for
                threshold-dependent metrics, where each arm carries thresholds
                fitted on its own validation predictions.
        """
        targets = np.asarray(targets)
        a = np.asarray(probabilities_a)
        b = np.asarray(probabilities_b)
        patient_ids = np.asarray(patient_ids)
        if a.shape != b.shape:
            raise PairedComparisonError(f"arms have different shapes: {a.shape} vs {b.shape}")
        if len(patient_ids) != len(a):
            raise PairedComparisonError(
                f"patient_ids has {len(patient_ids)} entries for {len(a)} rows"
            )

        score_a = metric_fn
        score_b = metric_fn_b or metric_fn
        rng = np.random.default_rng(self.seed)
        point_a = float(score_a(targets, a))
        point_b = float(score_b(targets, b))

        differences = np.full(self.n_iterations, np.nan, dtype=float)
        for iteration in range(self.n_iterations):
            index = self._resample_indices(patient_ids, rng)
            try:
                differences[iteration] = float(score_a(targets[index], a[index])) - float(
                    score_b(targets[index], b[index])
                )
            except ValueError:
                continue  # resample lost every positive of a rare label

        lower, upper, p_value = self._interval(differences, metric_name)
        return PairedDifferenceResult(
            metric=metric_name,
            arm_a=arm_a,
            arm_b=arm_b,
            estimate_a=point_a,
            estimate_b=point_b,
            difference=point_a - point_b,
            lower=lower,
            upper=upper,
            p_two_sided=p_value,
            n_iterations=self.n_iterations,
            n_patients=int(len(np.unique(patient_ids))),
            confidence=self.confidence,
        )

    def run_many(
        self,
        targets: np.ndarray,
        probabilities_a: np.ndarray,
        probabilities_b: np.ndarray,
        patient_ids: np.ndarray,
        thresholds_a: np.ndarray | float = 0.5,
        thresholds_b: np.ndarray | float = 0.5,
        label_names: list[str] | None = None,
        metrics: tuple[str, ...] = ("macro_f1", "macro_auroc", "macro_auprc"),
        arm_a: str = "a",
        arm_b: str = "b",
    ) -> pd.DataFrame:
        """Paired differences for several aggregate metrics at once.

        Each arm keeps its own thresholds, because they were fitted on that arm's
        own validation predictions; forcing one arm's thresholds onto the other
        measures threshold transfer rather than model quality.
        """
        calculator = MultiLabelMetrics(label_names=label_names)
        rows = []
        for name in metrics:

            def score_a(y: np.ndarray, p: np.ndarray, _name: str = name) -> float:
                return calculator.compute(y, p, thresholds_a)[_name]

            def score_b(y: np.ndarray, p: np.ndarray, _name: str = name) -> float:
                return calculator.compute(y, p, thresholds_b)[_name]

            result = self.run(
                targets, probabilities_a, probabilities_b, patient_ids,
                metric_fn=score_a, metric_fn_b=score_b,
                metric_name=name, arm_a=arm_a, arm_b=arm_b,
            )
            rows.append(result.to_dict())
            LOGGER.info("  %-14s %s  -> %s", name, result.format(), result.conclusion)
        return pd.DataFrame(rows)

    def per_label(
        self,
        targets: np.ndarray,
        probabilities_a: np.ndarray,
        probabilities_b: np.ndarray,
        patient_ids: np.ndarray,
        label_names: list[str],
        metric: str = "auprc",
        arm_a: str = "a",
        arm_b: str = "b",
        restrict_to: list[str] | None = None,
    ) -> pd.DataFrame:
        """Paired differences per biomarker.

        This is the table to report when the hypothesis names specific labels. A
        macro average over thirteen biomarkers dilutes an effect confined to
        three of them by a factor of four, which is how a real result becomes
        invisible.

        Args:
            restrict_to: Report only these labels -- pass the pre-registered
                target set so the confirmatory test is separated from the
                exploratory table.
        """
        if metric not in {"auprc", "auroc"}:
            raise ValueError(f"metric must be 'auprc' or 'auroc', got {metric!r}")
        scorer = (
            MultiLabelMetrics._safe_auprc if metric == "auprc" else MultiLabelMetrics._safe_auroc
        )
        targets = np.asarray(targets)
        wanted = set(restrict_to) if restrict_to else None

        rows = []
        for index, label in enumerate(label_names):
            if wanted is not None and label not in wanted:
                continue
            n_positive = int(targets[:, index].sum())
            if n_positive == 0:
                LOGGER.info("label '%s' has no positives in this partition; skipped", label)
                continue

            def metric_fn(y: np.ndarray, p: np.ndarray, _index: int = index) -> float:
                return float(scorer(y[:, _index], p[:, _index]))

            result = self.run(
                targets, probabilities_a, probabilities_b, patient_ids,
                metric_fn=metric_fn, metric_name=f"{label}_{metric}",
                arm_a=arm_a, arm_b=arm_b,
            )
            record = result.to_dict()
            record["label"] = label
            record["n_positive"] = n_positive
            rows.append(record)

        frame = pd.DataFrame(rows)
        if not len(frame):
            return frame
        columns = ["label", "n_positive"] + [c for c in frame.columns if c not in {"label", "n_positive"}]
        return frame[columns].sort_values("difference", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    def confirmatory_report(
        self,
        per_label_table: pd.DataFrame,
        preregistered: list[str],
    ) -> pd.DataFrame:
        """Split a per-label table into pre-registered and exploratory rows.

        The pre-registered rows are the confirmatory test of the hypothesis. The
        rest are exploratory and must be reported as such: with thirteen labels
        and a 95% interval, roughly one apparent win is expected by chance.
        """
        if not len(per_label_table):
            return per_label_table
        out = per_label_table.copy()
        out["preregistered"] = out["label"].isin(set(preregistered))
        out["role"] = np.where(out["preregistered"], "confirmatory", "exploratory")
        n_confirmatory = int(out["preregistered"].sum())
        LOGGER.info(
            "confirmatory rows: %d of %d (%d exploratory, expect ~%.1f false wins by chance)",
            n_confirmatory,
            len(out),
            len(out) - n_confirmatory,
            (len(out) - n_confirmatory) * (1.0 - self.confidence),
        )
        return out.sort_values(["preregistered", "difference"], ascending=[False, False])
