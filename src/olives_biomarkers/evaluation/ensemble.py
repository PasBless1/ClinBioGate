"""Seed ensembling.

With 13 test patients, a single run's macro AUPRC moves substantially with the
seed alone. Averaging several independently trained models is the cheapest way
to separate a real effect from initialisation noise: it both raises the point
estimate and, more importantly, makes the comparison between architectures less
dependent on which seed happened to land well.

Two correctness requirements are enforced rather than assumed:

* **Runs must be aligned by ``row_uid``.** Prediction files are not guaranteed to
  share a row order, so averaging them positionally would silently mix scans.
* **Runs must share a test partition.** Ensembling models evaluated on different
  patients produces a number that corresponds to no experiment at all.

Averaging in *logit* space is the default. Probability averaging pulls results
toward the mean prediction and is better calibrated for badly-scaled members, so
both are offered and the choice is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from olives_biomarkers.evaluation.bootstrap import PatientBootstrap
from olives_biomarkers.evaluation.metrics import MultiLabelMetrics, ThresholdOptimizer, ThresholdSet
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.ensemble")


class EnsembleMismatchError(RuntimeError):
    """Raised when runs cannot be ensembled because they are not comparable."""


@dataclass
class EnsemblePrediction:
    """Averaged predictions over an aligned set of runs."""

    logits: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray
    row_uid: np.ndarray
    patient_id: np.ndarray
    label_names: list[str]
    member_run_ids: list[str]
    space: str = "logit"

    @property
    def n_members(self) -> int:
        """How many runs were averaged."""
        return len(self.member_run_ids)


class SeedEnsemble:
    """Averages several runs of the same architecture across seeds.

    Args:
        space: ``"logit"`` averages pre-sigmoid outputs, ``"probability"``
            averages post-sigmoid ones.

    Example:
        >>> members = [r for r in runs if r.model_name == "oct_only"]
        >>> ensemble = SeedEnsemble().combine(members)
        >>> metrics = SeedEnsemble().evaluate(ensemble, thresholds)
    """

    def __init__(self, space: str = "logit") -> None:
        if space not in {"logit", "probability"}:
            raise ValueError(f"unknown ensemble space {space!r}; use 'logit' or 'probability'")
        self.space = space

    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

    def _check_comparable(self, members: Sequence, partition: str) -> None:
        """Refuse to ensemble runs that are not measuring the same thing."""
        if len(members) < 2:
            raise EnsembleMismatchError(
                f"ensembling needs at least two runs, got {len(members)}"
            )
        labels = {tuple(member.label_names) for member in members}
        if len(labels) > 1:
            raise EnsembleMismatchError("runs use different label sets")
        architectures = {member.model_name for member in members}
        if len(architectures) > 1:
            LOGGER.warning(
                "ensembling across different architectures (%s); this is a model ensemble, "
                "not a seed ensemble",
                sorted(architectures),
            )
        patients = {
            frozenset(np.unique(member.predictions[partition]["patient_id"]).tolist())
            for member in members
        }
        if len(patients) > 1:
            raise EnsembleMismatchError(
                f"runs were evaluated on different {partition} patients; their predictions "
                "cannot be averaged"
            )

    def combine(self, members: Sequence, partition: str = "test") -> EnsemblePrediction:
        """Average the members' predictions on one partition.

        Args:
            members: :class:`RunResult` objects, normally the same config at
                several seeds.
            partition: Which partition to combine.
        """
        self._check_comparable(members, partition)

        reference = members[0].predictions[partition]
        order = np.argsort(reference["row_uid"])
        row_uid = reference["row_uid"][order]
        targets = reference["targets"][order]
        patient_id = reference["patient_id"][order]

        stacked_logits = []
        for member in members:
            payload = member.predictions[partition]
            member_order = np.argsort(payload["row_uid"])
            if not np.array_equal(payload["row_uid"][member_order], row_uid):
                raise EnsembleMismatchError(
                    f"run {member.run_id} covers different scans than {members[0].run_id}"
                )
            if not np.array_equal(payload["targets"][member_order], targets):
                raise EnsembleMismatchError(
                    f"run {member.run_id} has different ground truth than {members[0].run_id}"
                )
            stacked_logits.append(payload["logits"][member_order])

        logits = np.stack(stacked_logits)
        if self.space == "logit":
            mean_logits = logits.mean(axis=0)
            probabilities = self._sigmoid(mean_logits)
        else:
            probabilities = self._sigmoid(logits).mean(axis=0)
            # Report an equivalent logit so downstream calibration still works.
            mean_logits = np.log(np.clip(probabilities, 1e-7, 1 - 1e-7) / np.clip(1 - probabilities, 1e-7, 1))

        LOGGER.info(
            "ensembled %d runs in %s space over %d scans from %d patients",
            len(members),
            self.space,
            len(row_uid),
            len(np.unique(patient_id)),
        )
        return EnsemblePrediction(
            logits=mean_logits,
            probabilities=probabilities,
            targets=targets,
            row_uid=row_uid,
            patient_id=patient_id,
            label_names=list(members[0].label_names),
            member_run_ids=[member.run_id for member in members],
            space=self.space,
        )

    # ------------------------------------------------------------------
    def fit_thresholds(self, members: Sequence) -> ThresholdSet:
        """Fit ensemble thresholds on the ensembled *validation* predictions.

        Member thresholds were each fitted against their own model's output, so
        reusing one of them on the averaged prediction would be inconsistent. The
        thresholds have to come from the ensemble's own validation output.
        """
        validation = self.combine(members, partition="val")
        return ThresholdOptimizer(objective="f1").fit(
            validation.targets, validation.probabilities, validation.label_names
        )

    def evaluate(
        self,
        ensemble: EnsemblePrediction,
        thresholds: np.ndarray | ThresholdSet | float = 0.5,
    ) -> dict[str, float]:
        """Aggregate metrics for the ensembled prediction."""
        values = thresholds.as_array() if isinstance(thresholds, ThresholdSet) else thresholds
        return MultiLabelMetrics(label_names=ensemble.label_names).compute(
            ensemble.targets, ensemble.probabilities, values
        )

    def per_label(
        self,
        ensemble: EnsemblePrediction,
        thresholds: np.ndarray | ThresholdSet | float = 0.5,
    ) -> pd.DataFrame:
        """Per-label metrics for the ensembled prediction."""
        values = thresholds.as_array() if isinstance(thresholds, ThresholdSet) else thresholds
        return MultiLabelMetrics(label_names=ensemble.label_names).per_label(
            ensemble.targets, ensemble.probabilities, values
        )

    def bootstrap(
        self,
        ensemble: EnsemblePrediction,
        thresholds: np.ndarray | ThresholdSet | float = 0.5,
        n_iterations: int = 1000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Patient-level bootstrap intervals for the ensembled prediction."""
        values = thresholds.as_array() if isinstance(thresholds, ThresholdSet) else thresholds
        table = PatientBootstrap(n_iterations=n_iterations, seed=seed).run_many(
            ensemble.targets,
            ensemble.probabilities,
            ensemble.patient_id,
            values,
            ensemble.label_names,
        )
        table.insert(0, "run_id", f"ensemble_{ensemble.n_members}x")
        return table

    # ------------------------------------------------------------------
    def compare_with_members(
        self,
        members: Sequence,
        thresholds: np.ndarray | ThresholdSet | float | None = None,
        partition: str = "test",
    ) -> pd.DataFrame:
        """Members and their ensemble side by side.

        The row that matters is the gap between the best single seed and the
        ensemble. If the ensemble merely matches the mean member, averaging
        bought nothing; if it beats the *best* member, the individual runs were
        making uncorrelated errors and the ensemble is doing real work.
        """
        ensemble = self.combine(members, partition=partition)
        if thresholds is None:
            thresholds = self.fit_thresholds(members)
        values = thresholds.as_array() if isinstance(thresholds, ThresholdSet) else thresholds
        metrics = MultiLabelMetrics(label_names=ensemble.label_names)

        rows = []
        for member in members:
            payload = member.predictions[partition]
            scores = metrics.compute(payload["targets"], payload["probabilities"], values)
            rows.append({"model": member.model_name, "run": member.run_id, "seed": member.seed, **scores})

        scores = self.evaluate(ensemble, values)
        rows.append(
            {
                "model": f"{members[0].model_name} (ensemble)",
                "run": f"{ensemble.n_members}x {self.space}-mean",
                "seed": None,
                **scores,
            }
        )
        table = pd.DataFrame(rows)

        member_rows = table.iloc[:-1]
        for metric in ("macro_f1", "macro_auprc", "macro_auroc"):
            if metric in table.columns:
                LOGGER.info(
                    "%s: members mean %.4f, best %.4f, ensemble %.4f",
                    metric,
                    member_rows[metric].mean(),
                    member_rows[metric].max(),
                    table.iloc[-1][metric],
                )
        return table
