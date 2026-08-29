"""Experiment orchestration, aggregation and gate-analysis tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from olives_biomarkers.evaluation.gating import GateAnalyzer  # noqa: E402
from olives_biomarkers.evaluation.metrics import ThresholdSet  # noqa: E402
from olives_biomarkers.experiment import ResultsAggregator, RunResult  # noqa: E402
from olives_biomarkers.models import ClinicalOnlyModel, GatedFusionModel  # noqa: E402
from olives_biomarkers.training.callbacks import MetricHistory  # noqa: E402

LABELS = ["irf", "drt_me", "pavf"]


def _make_result(
    model_name: str,
    seed: int = 42,
    macro_auprc: float = 0.5,
    macro_f1: float = 0.4,
    tmp_path=None,
) -> RunResult:
    """A RunResult with just enough filled in for the aggregator."""
    history = MetricHistory()
    history.append(1, train_loss=0.9, val_loss=0.8, val_macro_auprc=macro_auprc)
    return RunResult(
        run_id=f"{model_name}_seed{seed}",
        experiment=model_name,
        model_name=model_name,
        seed=seed,
        split_name="holdout",
        target_set="sixteen",
        run_dir=tmp_path or "/tmp",
        history=history,
        thresholds=ThresholdSet(thresholds=np.full(len(LABELS), 0.5), label_names=LABELS),
        label_names=LABELS,
        test_metrics={"macro_auprc": macro_auprc, "macro_f1": macro_f1, "macro_auroc": 0.7},
        per_label=pd.DataFrame(
            {
                "label": LABELS,
                "n_positive": [100, 50, 5],
                "auprc": [macro_auprc + 0.1, macro_auprc, macro_auprc - 0.1],
                "degenerate": [False, False, True],
            }
        ),
        model_description={"n_trainable_parameters": 1000},
    )


class TestResultsAggregator:
    """Comparison tables built from many runs."""

    def test_comparison_has_a_row_per_run(self) -> None:
        results = [_make_result("oct_only"), _make_result("gated_fusion", macro_auprc=0.6)]
        table = ResultsAggregator(results).comparison()
        assert len(table) == 2
        assert set(table["model"]) == {"oct_only", "gated_fusion"}

    def test_comparison_sorts_by_the_requested_metric(self) -> None:
        results = [
            _make_result("oct_only", macro_auprc=0.4),
            _make_result("gated_fusion", macro_auprc=0.6),
        ]
        table = ResultsAggregator(results).comparison(sort_by="macro_auprc")
        assert table.iloc[0]["model"] == "gated_fusion"

    def test_across_seeds_reports_mean_and_spread(self) -> None:
        results = [
            _make_result("oct_only", seed=1, macro_auprc=0.4),
            _make_result("oct_only", seed=2, macro_auprc=0.6),
        ]
        table = ResultsAggregator(results).across_seeds()
        row = table[table["model"] == "oct_only"].iloc[0]
        assert row["macro_auprc_mean"] == pytest.approx(0.5)
        assert row["macro_auprc_count"] == 2
        assert row["macro_auprc_std"] > 0

    def test_per_label_pivot_keeps_positive_counts(self) -> None:
        results = [_make_result("oct_only"), _make_result("gated_fusion", macro_auprc=0.6)]
        pivot = ResultsAggregator(results).per_label_pivot("auprc")
        assert "n_positive" in pivot.columns
        assert set(pivot.index) == set(LABELS)
        assert pivot["n_positive"].is_monotonic_decreasing

    def test_overlapping_intervals_report_no_supported_difference(self, tmp_path) -> None:
        results = []
        for name, low, high in [("oct_only", 0.30, 0.60), ("gated_fusion", 0.40, 0.70)]:
            run_dir = tmp_path / name
            run_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "run_id": name,
                        "metric": "macro_auprc",
                        "point_estimate": (low + high) / 2,
                        "ci_lower": low,
                        "ci_upper": high,
                    }
                ]
            ).to_csv(run_dir / "bootstrap_ci.csv", index=False)
            results.append(_make_result(name, tmp_path=run_dir))

        overlap = ResultsAggregator(results).intervals_overlap("macro_auprc")
        assert len(overlap) == 1
        assert bool(overlap.iloc[0]["intervals_overlap"])
        assert overlap.iloc[0]["conclusion"] == "no supported difference"

    def test_separated_intervals_are_reported_as_separated(self, tmp_path) -> None:
        results = []
        for name, low, high in [("oct_only", 0.10, 0.20), ("gated_fusion", 0.60, 0.70)]:
            run_dir = tmp_path / name
            run_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "run_id": name,
                        "metric": "macro_auprc",
                        "point_estimate": (low + high) / 2,
                        "ci_lower": low,
                        "ci_upper": high,
                    }
                ]
            ).to_csv(run_dir / "bootstrap_ci.csv", index=False)
            results.append(_make_result(name, tmp_path=run_dir))

        overlap = ResultsAggregator(results).intervals_overlap("macro_auprc")
        assert not bool(overlap.iloc[0]["intervals_overlap"])
        assert overlap.iloc[0]["conclusion"] == "separated"


class _GateLoader:
    """Loader yielding clinical batches, for gate analysis."""

    def __init__(self, n_batches: int = 3, batch: int = 8, clinical_dim: int = 4) -> None:
        generator = torch.Generator().manual_seed(0)
        self.batches = [
            {
                "clinical": torch.randn(batch, clinical_dim, generator=generator),
                "target": torch.randint(0, 2, (batch, len(LABELS)), generator=generator).float(),
                "patient_id": torch.full((batch,), index),
            }
            for index in range(n_batches)
        ]

    def __iter__(self):
        return iter(self.batches)


class TestGateAnalyzer:
    """Whether the gate did anything."""

    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return GatedFusionModel(clinical_dim=4, n_labels=len(LABELS), pretrained=False)

    def test_rejects_a_model_without_a_gate(self) -> None:
        with pytest.raises(TypeError, match="no clinical gate"):
            GateAnalyzer(ClinicalOnlyModel(clinical_dim=4, n_labels=3))

    def test_collect_returns_aligned_arrays(self, model) -> None:
        collected = GateAnalyzer(model).collect(_GateLoader())
        assert collected["gate"].shape[0] == 24
        assert collected["gate"].shape == collected["scale"].shape
        assert collected["clinical"].shape[0] == 24
        assert collected["target"].shape == (24, len(LABELS))

    def test_untrained_gate_is_detected_as_identity(self, model) -> None:
        collected = GateAnalyzer(model).collect(_GateLoader())
        summary = GateAnalyzer(model).summary(collected)
        assert summary["mean_absolute_deviation_from_identity"] < 0.02
        assert summary["scale_mean"] == pytest.approx(1.0, abs=0.02)

    def test_interpretation_flags_an_inert_gate(self, model) -> None:
        analyzer = GateAnalyzer(model)
        collected = analyzer.collect(_GateLoader())
        notes = analyzer.interpret(analyzer.summary(collected))
        assert any("identity" in note for note in notes)

    def test_gate_values_stay_bounded(self, model) -> None:
        collected = GateAnalyzer(model).collect(_GateLoader())
        assert collected["gate"].min() >= 0.0
        assert collected["gate"].max() <= 1.0

    def test_clinical_response_has_a_row_per_feature(self, model) -> None:
        analyzer = GateAnalyzer(model)
        collected = analyzer.collect(_GateLoader())
        table = analyzer.clinical_response(collected, ["bcva", "cst", "bcva_missing", "cst_missing"])
        assert len(table) == 4
        assert set(table.columns) >= {"feature", "spearman_r", "note"}

    def test_gate_by_label_covers_every_label(self, model) -> None:
        analyzer = GateAnalyzer(model)
        collected = analyzer.collect(_GateLoader())
        table = analyzer.gate_by_label(collected, LABELS)
        assert set(table["label"]) == set(LABELS)

    def test_channel_activity_is_ordered_by_deviation(self, model) -> None:
        analyzer = GateAnalyzer(model)
        collected = analyzer.collect(_GateLoader())
        table = analyzer.channel_activity(collected, top_n=10)
        assert len(table) == 10
        assert table["abs_deviation_from_identity"].is_monotonic_decreasing


class TestRunResultRoundTrip:
    """Saved runs must reload well enough for the later notebooks."""

    def test_discover_ignores_directories_without_metadata(self, tmp_path) -> None:
        (tmp_path / "incomplete").mkdir()
        (tmp_path / "complete").mkdir()
        (tmp_path / "complete" / "run_metadata.json").write_text("{}", encoding="utf-8")
        found = RunResult.discover(tmp_path)
        assert [p.name for p in found] == ["complete"]


class TestPretrainingPoolGuard:
    """Self-supervised pretraining must not see held-out patients.

    "Unlabelled" is not the same as "safe": pretraining on a test patient's scans
    teaches the encoder that patient's anatomy even without their labels.
    """

    def test_pool_contains_only_training_patients(self, manifest) -> None:
        from olives_biomarkers.data.splits import PatientGroupedSplitter
        from olives_biomarkers.pipeline import OlivesPipeline

        frame = manifest.modelling_frame(policy="keep_first", labelled_only=True)
        assignment = PatientGroupedSplitter(seed=42).split(frame)

        pipeline = OlivesPipeline.__new__(OlivesPipeline)  # no data root needed
        pipeline.config = _minimal_config()
        pipeline.get_manifest = lambda *a, **k: manifest  # type: ignore[assignment]

        pool = OlivesPipeline.pretraining_frame(pipeline, assignment, manifest=manifest)
        assert set(pool["patient_id"]) <= set(assignment.train)
        for held_out in (assignment.val, assignment.test, assignment.calibration):
            assert not set(pool["patient_id"]) & set(held_out)

    def test_pool_includes_unlabelled_scans(self, manifest) -> None:
        from olives_biomarkers.data.splits import PatientGroupedSplitter
        from olives_biomarkers.pipeline import OlivesPipeline

        frame = manifest.modelling_frame(policy="keep_first", labelled_only=True)
        assignment = PatientGroupedSplitter(seed=42).split(frame)

        pipeline = OlivesPipeline.__new__(OlivesPipeline)
        pipeline.config = _minimal_config()
        pipeline.get_manifest = lambda *a, **k: manifest  # type: ignore[assignment]

        pool = OlivesPipeline.pretraining_frame(pipeline, assignment, manifest=manifest)
        labelled_only = frame[frame["patient_id"].isin(assignment.train)]
        assert len(pool) > len(labelled_only), "pool should add the unlabelled scans"

    def test_unknown_partition_is_rejected(self, manifest) -> None:
        from olives_biomarkers.data.splits import PatientGroupedSplitter
        from olives_biomarkers.pipeline import OlivesPipeline

        frame = manifest.modelling_frame(policy="keep_first", labelled_only=True)
        assignment = PatientGroupedSplitter(seed=42).split(frame)

        pipeline = OlivesPipeline.__new__(OlivesPipeline)
        pipeline.config = _minimal_config()
        pipeline.get_manifest = lambda *a, **k: manifest  # type: ignore[assignment]

        with pytest.raises(KeyError, match="not in this split"):
            OlivesPipeline.pretraining_frame(
                pipeline, assignment, manifest=manifest, include_partitions=("nope",)
            )


def _minimal_config():
    from olives_biomarkers.config import ExperimentConfig

    return ExperimentConfig()
