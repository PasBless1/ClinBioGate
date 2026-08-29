"""Paired patient-level comparison between two model arms."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from olives_biomarkers.evaluation.bootstrap import PatientBootstrap
from olives_biomarkers.evaluation.comparison import (
    PairedComparisonError,
    PairedPatientBootstrap,
)

LABELS = ["irf", "drt_me", "pavf"]


def _auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Single-label AUROC, raising the way the metric layer does on one class."""
    return float(roc_auc_score(targets[:, 0], probabilities[:, 0]))


def _cohort(
    n_patients: int = 24,
    rows_per_patient: int = 25,
    advantage: float = 0.35,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Two arms scored on the same rows, arm A consistently slightly better.

    Patient difficulty is deliberately large relative to the advantage. That is
    the situation the paired test exists for: each marginal interval is wide
    because patients differ, while the difference is stable in every resample.
    """
    rng = np.random.default_rng(seed)
    targets, probs_a, probs_b, patients, row_uid = [], [], [], [], []
    uid = 0
    for patient in range(n_patients):
        difficulty = rng.normal(0.0, 1.4)  # shared by both arms, so it cancels
        y = rng.integers(0, 2, size=rows_per_patient).astype(float)
        noise = rng.normal(0.0, 1.0, size=rows_per_patient)
        signal = y * 1.0 + difficulty + noise
        probs_b.append(signal)
        probs_a.append(signal + advantage * y)  # arm A separates the classes more
        targets.append(y)
        patients.append(np.full(rows_per_patient, patient))
        row_uid.append(np.arange(uid, uid + rows_per_patient))
        uid += rows_per_patient

    def stack(values: list[np.ndarray]) -> np.ndarray:
        flat = np.concatenate(values)
        return np.stack([flat, flat, flat], axis=1)

    return {
        "targets": stack(targets),
        "probabilities_a": stack(probs_a),
        "probabilities_b": stack(probs_b),
        "patient_ids": np.concatenate(patients),
        "row_uid": np.concatenate(row_uid),
    }


class TestAlignment:
    """Two arms must be matched by identifier, never by position."""

    def test_shuffled_row_order_gives_the_same_result(self) -> None:
        data = _cohort()
        order = np.random.default_rng(5).permutation(len(data["row_uid"]))
        test = PairedPatientBootstrap(n_iterations=150)

        straight = test.run(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], _auroc,
        )
        index_a, index_b = test.align(data["row_uid"], data["row_uid"][order])
        shuffled = test.run(
            data["targets"][index_a],
            data["probabilities_a"][index_a],
            data["probabilities_b"][order][index_b],
            data["patient_ids"][index_a],
            _auroc,
        )
        assert shuffled.difference == pytest.approx(straight.difference)

    def test_different_rows_are_refused(self) -> None:
        with pytest.raises(PairedComparisonError, match="identical rows"):
            PairedPatientBootstrap.align(np.arange(10), np.arange(5, 15))

    def test_different_lengths_are_refused(self) -> None:
        with pytest.raises(PairedComparisonError, match="identical rows"):
            PairedPatientBootstrap.align(np.arange(10), np.arange(9))

    def test_mismatched_shapes_are_refused(self) -> None:
        data = _cohort()
        with pytest.raises(PairedComparisonError, match="different shapes"):
            PairedPatientBootstrap(n_iterations=10).run(
                data["targets"], data["probabilities_a"],
                data["probabilities_b"][:, :2], data["patient_ids"], _auroc,
            )

    def test_patient_ids_must_cover_every_row(self) -> None:
        data = _cohort()
        with pytest.raises(PairedComparisonError, match="patient_ids"):
            PairedPatientBootstrap(n_iterations=10).run(
                data["targets"], data["probabilities_a"], data["probabilities_b"],
                data["patient_ids"][:-5], _auroc,
            )


class TestPairedDifference:
    """The test itself."""

    def test_identical_arms_give_exactly_zero_and_no_support(self) -> None:
        data = _cohort()
        result = PairedPatientBootstrap(n_iterations=200).run(
            data["targets"], data["probabilities_a"], data["probabilities_a"],
            data["patient_ids"], _auroc,
        )
        assert result.difference == pytest.approx(0.0)
        assert not result.supported
        assert result.conclusion == "no supported difference"

    def test_a_consistent_advantage_is_detected(self) -> None:
        data = _cohort(advantage=0.35)
        result = PairedPatientBootstrap(n_iterations=400, seed=1).run(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], _auroc, arm_a="fusion", arm_b="oct_only",
        )
        assert result.difference > 0
        assert result.supported
        assert result.conclusion == "fusion > oct_only"
        assert result.p_two_sided < 0.05

    def test_the_paired_interval_is_tighter_than_either_marginal_one(self) -> None:
        """The whole reason this class exists.

        Patient difficulty inflates both marginal intervals independently. It is
        shared, so it cancels in the difference. If this ever stops holding, the
        pairing has been broken somewhere.
        """
        data = _cohort(advantage=0.35)
        paired = PairedPatientBootstrap(n_iterations=400, seed=1).run(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], _auroc,
        )
        marginal = PatientBootstrap(n_iterations=400, seed=1)
        width_a = _width(marginal.run(
            data["targets"], data["probabilities_a"], data["patient_ids"], _auroc,
        ))
        width_b = _width(marginal.run(
            data["targets"], data["probabilities_b"], data["patient_ids"], _auroc,
        ))
        paired_width = paired.upper - paired.lower
        assert paired_width < min(width_a, width_b)

    def test_overlapping_marginal_intervals_can_still_hide_a_real_difference(self) -> None:
        """The failure mode that motivated replacing the overlap check."""
        data = _cohort(advantage=0.35)
        marginal = PatientBootstrap(n_iterations=400, seed=1)
        a = marginal.run(data["targets"], data["probabilities_a"], data["patient_ids"], _auroc)
        b = marginal.run(data["targets"], data["probabilities_b"], data["patient_ids"], _auroc)
        overlap = not (a.lower > b.upper or b.lower > a.upper)

        paired = PairedPatientBootstrap(n_iterations=400, seed=1).run(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], _auroc,
        )
        assert overlap, "fixture no longer reproduces the overlapping-interval case"
        assert paired.supported, "the paired test should still find the difference"

    def test_direction_reverses_when_the_arms_swap(self) -> None:
        data = _cohort(advantage=0.35)
        test = PairedPatientBootstrap(n_iterations=200, seed=1)
        forward = test.run(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], _auroc, arm_a="a", arm_b="b",
        )
        backward = test.run(
            data["targets"], data["probabilities_b"], data["probabilities_a"],
            data["patient_ids"], _auroc, arm_a="b", arm_b="a",
        )
        assert forward.difference == pytest.approx(-backward.difference)
        assert backward.conclusion == "a > b"

    def test_p_value_is_never_reported_as_exactly_zero(self) -> None:
        """With N resamples the smallest resolvable p is 1/N, not 0."""
        data = _cohort(advantage=3.0)
        result = PairedPatientBootstrap(n_iterations=100, seed=1).run(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], _auroc,
        )
        assert result.p_two_sided > 0.0

    def test_resamples_draw_patients_not_rows(self) -> None:
        data = _cohort(n_patients=6, rows_per_patient=10)
        test = PairedPatientBootstrap(n_iterations=5)
        rng = np.random.default_rng(0)
        index = test._resample_indices(data["patient_ids"], rng)
        drawn = data["patient_ids"][index]
        counts = np.unique(drawn, return_counts=True)[1]
        # Every drawn patient contributes all ten of their rows, or none.
        assert set(counts.tolist()) <= {10, 20, 30, 40, 50, 60}

    def test_each_arm_can_use_its_own_thresholds(self) -> None:
        """Threshold-dependent metrics must not force one arm's thresholds."""
        data = _cohort()
        calls = {"a": 0, "b": 0}

        def metric_a(y: np.ndarray, p: np.ndarray) -> float:
            calls["a"] += 1
            return 1.0

        def metric_b(y: np.ndarray, p: np.ndarray) -> float:
            calls["b"] += 1
            return 0.5

        result = PairedPatientBootstrap(n_iterations=20).run(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], metric_fn=metric_a, metric_fn_b=metric_b,
        )
        assert result.difference == pytest.approx(0.5)
        assert calls["a"] > 0 and calls["b"] > 0

    def test_to_dict_carries_the_conclusion(self) -> None:
        data = _cohort()
        payload = PairedPatientBootstrap(n_iterations=50).run(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], _auroc,
        ).to_dict()
        for key in ("difference", "ci_lower", "ci_upper", "p_two_sided", "supported", "conclusion"):
            assert key in payload


class TestPerLabelAndPreregistration:
    """Per-biomarker reporting, split into confirmatory and exploratory."""

    def test_per_label_has_a_row_per_label_with_positives(self) -> None:
        data = _cohort()
        table = PairedPatientBootstrap(n_iterations=60).per_label(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], LABELS,
        )
        assert set(table["label"]) == set(LABELS)
        assert (table["n_positive"] > 0).all()

    def test_labels_without_positives_are_skipped_not_scored_zero(self) -> None:
        data = _cohort()
        data["targets"][:, 2] = 0.0
        table = PairedPatientBootstrap(n_iterations=60).per_label(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], LABELS,
        )
        assert "pavf" not in set(table["label"])

    def test_restrict_to_limits_the_table(self) -> None:
        data = _cohort()
        table = PairedPatientBootstrap(n_iterations=60).per_label(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], LABELS, restrict_to=["irf"],
        )
        assert list(table["label"]) == ["irf"]

    def test_unknown_metric_is_rejected(self) -> None:
        data = _cohort()
        with pytest.raises(ValueError, match="auprc"):
            PairedPatientBootstrap(n_iterations=10).per_label(
                data["targets"], data["probabilities_a"], data["probabilities_b"],
                data["patient_ids"], LABELS, metric="accuracy",
            )

    def test_confirmatory_report_marks_preregistered_rows(self) -> None:
        data = _cohort()
        test = PairedPatientBootstrap(n_iterations=60)
        table = test.per_label(
            data["targets"], data["probabilities_a"], data["probabilities_b"],
            data["patient_ids"], LABELS,
        )
        marked = test.confirmatory_report(table, ["irf", "pavf"])
        roles = dict(zip(marked["label"], marked["role"]))
        assert roles["irf"] == "confirmatory"
        assert roles["drt_me"] == "exploratory"
        # Confirmatory rows sort first so they are read before the fishing.
        assert marked.iloc[0]["role"] == "confirmatory"

    def test_confirmatory_report_on_an_empty_table_is_safe(self) -> None:
        import pandas as pd

        empty = pd.DataFrame()
        assert PairedPatientBootstrap().confirmatory_report(empty, ["irf"]).empty


def _width(result) -> float:
    return result.upper - result.lower
