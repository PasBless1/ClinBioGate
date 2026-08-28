"""Metric, threshold, bootstrap and calibration tests.

The recurring theme is rare labels: a metric must return ``nan`` for a label with
no positives rather than a number that looks like a score.
"""

from __future__ import annotations

import numpy as np
import pytest

from olives_biomarkers.evaluation.bootstrap import PatientBootstrap
from olives_biomarkers.evaluation.calibration import CalibrationEvaluator, TemperatureScaler
from olives_biomarkers.evaluation.metrics import MultiLabelMetrics, ThresholdOptimizer


@pytest.fixture
def perfect_predictions():
    targets = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
    return targets, targets * 0.99 + 0.005


@pytest.fixture
def random_predictions():
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 2, (200, 4)).astype(float)
    return targets, rng.random((200, 4))


class TestMultiLabelMetrics:
    """Aggregate and per-label behaviour."""

    def test_perfect_predictions_score_one(self, perfect_predictions) -> None:
        targets, probabilities = perfect_predictions
        scores = MultiLabelMetrics().compute(targets, probabilities)
        assert scores["macro_f1"] == pytest.approx(1.0)
        assert scores["macro_auroc"] == pytest.approx(1.0)
        assert scores["exact_match"] == pytest.approx(1.0)
        assert scores["hamming_loss"] == pytest.approx(0.0)

    def test_random_predictions_score_near_chance(self, random_predictions) -> None:
        targets, probabilities = random_predictions
        scores = MultiLabelMetrics().compute(targets, probabilities)
        assert 0.35 < scores["macro_auroc"] < 0.65

    def test_label_with_no_positives_yields_nan_not_zero(self) -> None:
        targets = np.zeros((20, 2))
        targets[:, 0] = np.random.default_rng(0).integers(0, 2, 20)
        probabilities = np.random.default_rng(1).random((20, 2))

        table = MultiLabelMetrics().per_label(targets, probabilities)
        assert np.isnan(table.loc[1, "auroc"]), "AUROC invented for a single-class label"
        assert np.isnan(table.loc[1, "auprc"])
        assert bool(table.loc[1, "degenerate"])

    def test_macro_average_skips_degenerate_labels(self) -> None:
        targets = np.zeros((20, 2))
        targets[:, 0] = 1
        targets[:10, 0] = 0
        probabilities = np.zeros((20, 2))
        probabilities[:, 0] = np.linspace(0, 1, 20)

        scores = MultiLabelMetrics().compute(targets, probabilities)
        assert not np.isnan(scores["macro_auroc"])
        assert scores["n_degenerate_labels"] == 1

    def test_per_label_confusion_counts_are_consistent(self, random_predictions) -> None:
        targets, probabilities = random_predictions
        table = MultiLabelMetrics().per_label(targets, probabilities)
        totals = table[["tp", "fp", "fn", "tn"]].sum(axis=1)
        assert (totals == len(targets)).all()

    def test_per_label_thresholds_are_applied(self, random_predictions) -> None:
        targets, probabilities = random_predictions
        low = MultiLabelMetrics().compute(targets, probabilities, thresholds=0.1)
        high = MultiLabelMetrics().compute(targets, probabilities, thresholds=0.9)
        assert low["macro_recall"] > high["macro_recall"]

    def test_label_names_are_carried_through(self, random_predictions) -> None:
        targets, probabilities = random_predictions
        names = ["a", "b", "c", "d"]
        table = MultiLabelMetrics(label_names=names).per_label(targets, probabilities)
        assert table["label"].tolist() == names


class TestThresholdOptimizer:
    """Thresholds are fitted on validation data and then frozen."""

    def test_recovers_a_shifted_optimum(self) -> None:
        rng = np.random.default_rng(0)
        targets = rng.integers(0, 2, (400, 1)).astype(float)
        # Probabilities compressed into the low range: 0.5 is a poor cut.
        probabilities = np.where(targets == 1, rng.uniform(0.2, 0.4, (400, 1)),
                                 rng.uniform(0.0, 0.2, (400, 1)))
        thresholds = ThresholdOptimizer("f1").fit(targets, probabilities)
        assert thresholds.as_array()[0] < 0.5

    def test_beats_the_default_threshold(self) -> None:
        rng = np.random.default_rng(1)
        targets = (rng.random((500, 3)) < 0.1).astype(float)
        probabilities = np.clip(targets * 0.3 + rng.normal(0.15, 0.08, (500, 3)), 0, 1)

        metrics = MultiLabelMetrics()
        default = metrics.compute(targets, probabilities, 0.5)["macro_f1"]
        tuned_thresholds = ThresholdOptimizer("f1").fit(targets, probabilities).as_array()
        tuned = metrics.compute(targets, probabilities, tuned_thresholds)["macro_f1"]
        assert tuned >= default

    def test_single_class_label_keeps_the_default(self) -> None:
        targets = np.zeros((50, 2))
        targets[:, 0] = 1
        probabilities = np.random.default_rng(0).random((50, 2))
        thresholds = ThresholdOptimizer().fit(targets, probabilities)
        assert thresholds.as_array()[0] == pytest.approx(0.5)
        assert thresholds.as_array()[1] == pytest.approx(0.5)

    def test_youden_objective_is_available(self, random_predictions) -> None:
        targets, probabilities = random_predictions
        thresholds = ThresholdOptimizer("youden").fit(targets, probabilities)
        assert thresholds.objective == "youden"
        assert len(thresholds.as_array()) == targets.shape[1]


class TestPatientBootstrap:
    """Resampling must happen at patient level."""

    def test_resamples_whole_patients(self) -> None:
        patient_ids = np.repeat([1, 2, 3, 4], 10)
        rng = np.random.default_rng(0)
        index = PatientBootstrap()._resample_indices(patient_ids, rng)
        drawn = patient_ids[index]
        counts = {p: int((drawn == p).sum()) for p in np.unique(drawn)}
        # Every drawn patient contributes all 10 of its rows, or a multiple of 10.
        assert all(count % 10 == 0 for count in counts.values())

    def test_interval_brackets_the_point_estimate(self) -> None:
        rng = np.random.default_rng(0)
        targets = rng.integers(0, 2, (200, 3)).astype(float)
        probabilities = np.clip(targets * 0.6 + rng.normal(0.2, 0.2, (200, 3)), 0, 1)
        patient_ids = np.repeat(np.arange(20), 10)

        metrics = MultiLabelMetrics()
        result = PatientBootstrap(n_iterations=50, seed=0).run(
            targets,
            probabilities,
            patient_ids,
            lambda y, p: metrics.compute(y, p)["macro_auroc"],
            "macro_auroc",
        )
        assert result.lower <= result.point_estimate <= result.upper
        assert result.n_patients == 20

    def test_patient_level_interval_is_wider_than_a_scan_level_one(self) -> None:
        # 20 patients x 10 correlated scans each.
        rng = np.random.default_rng(3)
        patient_effect = rng.normal(0, 1.2, 20)
        targets, probabilities, patients = [], [], []
        for patient, effect in enumerate(patient_effect):
            for _ in range(10):
                label = float(rng.random() < 0.5)
                targets.append([label])
                probabilities.append([1 / (1 + np.exp(-(effect + 2 * label + rng.normal(0, 0.3))))])
                patients.append(patient)
        targets = np.array(targets)
        probabilities = np.array(probabilities)
        patients = np.array(patients)

        metrics = MultiLabelMetrics()

        def auroc(y, p):
            return metrics.compute(y, p)["macro_auroc"]

        grouped = PatientBootstrap(n_iterations=200, seed=0).run(
            targets, probabilities, patients, auroc, "macro_auroc"
        )
        scan_level = PatientBootstrap(n_iterations=200, seed=0).run(
            targets, probabilities, np.arange(len(targets)), auroc, "macro_auroc"
        )
        assert (grouped.upper - grouped.lower) > (scan_level.upper - scan_level.lower)

    def test_run_many_returns_a_row_per_metric(self) -> None:
        rng = np.random.default_rng(0)
        targets = rng.integers(0, 2, (100, 2)).astype(float)
        probabilities = rng.random((100, 2))
        patients = np.repeat(np.arange(10), 10)
        table = PatientBootstrap(n_iterations=20, seed=0).run_many(
            targets, probabilities, patients, metrics=("macro_f1", "macro_auroc")
        )
        assert len(table) == 2
        assert set(table["metric"]) == {"macro_f1", "macro_auroc"}


class TestCalibration:
    """Temperature scaling and reliability."""

    def test_improves_a_deliberately_overconfident_model(self) -> None:
        rng = np.random.default_rng(0)
        targets = rng.integers(0, 2, (600, 2)).astype(float)
        # Inflated logits: correct ranking, badly overconfident.
        logits = (targets * 2 - 1) * rng.uniform(3, 9, (600, 2))

        before = 1 / (1 + np.exp(-logits))
        scaler = TemperatureScaler().fit(logits, targets)
        after = scaler.transform(logits)

        evaluator = CalibrationEvaluator()
        ece_before = evaluator.evaluate(targets, before)["ece"].mean()
        ece_after = evaluator.evaluate(targets, after)["ece"].mean()
        assert ece_after <= ece_before + 1e-6

    def test_does_not_change_ranking(self) -> None:
        rng = np.random.default_rng(1)
        targets = rng.integers(0, 2, (200, 1)).astype(float)
        logits = rng.normal(0, 3, (200, 1))
        scaler = TemperatureScaler().fit(logits, targets)
        calibrated = scaler.transform(logits)
        assert np.array_equal(np.argsort(logits[:, 0]), np.argsort(calibrated[:, 0]))

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            TemperatureScaler().transform(np.zeros((4, 2)))

    def test_single_class_label_keeps_temperature_one(self) -> None:
        targets = np.zeros((50, 2))
        targets[:, 0] = 1
        logits = np.random.default_rng(0).normal(0, 1, (50, 2))
        scaler = TemperatureScaler().fit(logits, targets)
        assert scaler.state is not None
        assert scaler.state.temperatures[0] == pytest.approx(1.0)

    def test_perfect_calibration_yields_near_zero_ece(self) -> None:
        rng = np.random.default_rng(0)
        probabilities = rng.uniform(0, 1, 5000)
        targets = (rng.uniform(0, 1, 5000) < probabilities).astype(float)
        ece = CalibrationEvaluator(n_bins=10).expected_calibration_error(targets, probabilities)
        assert ece < 0.05

    def test_reliability_curve_covers_every_bin(self) -> None:
        rng = np.random.default_rng(0)
        probabilities = rng.uniform(0, 1, 1000)
        targets = (rng.uniform(0, 1, 1000) < probabilities).astype(float)
        curve = CalibrationEvaluator(n_bins=10).reliability_curve(targets, probabilities)
        assert len(curve) == 10
        assert curve["n"].sum() > 0

    def test_rare_labels_are_marked_unreliable(self) -> None:
        targets = np.zeros((500, 2))
        targets[:5, 0] = 1
        targets[:200, 1] = 1
        probabilities = np.random.default_rng(0).random((500, 2))
        table = CalibrationEvaluator().evaluate(targets, probabilities, min_positives=20)
        assert not bool(table.loc[0, "reliable"])
        assert bool(table.loc[1, "reliable"])
