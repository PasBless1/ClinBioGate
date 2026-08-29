"""Experiment orchestration.

``scripts/train.py`` and the modelling notebooks must not each re-implement the
train / threshold / evaluate sequence, or they will drift apart and stop being
comparable. Everything lives here instead:

* :class:`ExperimentRunner` - one model, one split, one seed, end to end.
* :class:`RunEvaluator` - the post-hoc pass: calibration, MC dropout,
  selective prediction, patient-level bootstrap.
* :class:`ExperimentSuite` - the same recipe across models, seeds and folds.
* :class:`ResultsAggregator` - turns many runs into the comparison tables.

The ordering inside :meth:`ExperimentRunner.run` is the part that matters:
preprocessing statistics and class weights come from the training partition,
thresholds are fitted on validation and then frozen, calibration is fitted on a
patient-disjoint calibration partition, and only then is test touched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from olives_biomarkers.config import ExperimentConfig
from olives_biomarkers.data.dataset import OlivesDataModule, ParquetImageReader
from olives_biomarkers.data.longitudinal import ClinicalPerturbation, LongitudinalClinicalFeatures
from olives_biomarkers.data.manifests import Manifest
from olives_biomarkers.data.preprocessing import ImageTransformFactory
from olives_biomarkers.data.splits import SplitAssignment
from olives_biomarkers.evaluation.bootstrap import PatientBootstrap
from olives_biomarkers.evaluation.calibration import CalibrationEvaluator, TemperatureScaler
from olives_biomarkers.evaluation.metrics import MultiLabelMetrics, ThresholdOptimizer, ThresholdSet
from olives_biomarkers.evaluation.uncertainty import (
    MCDropoutInference,
    SelectivePredictor,
    UncertaintyOutput,
)
from olives_biomarkers.models import ModelFactory
from olives_biomarkers.pipeline import OlivesPipeline
from olives_biomarkers.training import LossFactory, Trainer
from olives_biomarkers.training.callbacks import MetricHistory
from olives_biomarkers.utils.io import JsonIO
from olives_biomarkers.utils.logging import LoggerFactory
from olives_biomarkers.utils.reproducibility import RunMetadata, SeedManager

LOGGER = LoggerFactory.get("olives.experiment")


@dataclass
class RunResult:
    """Everything one training run produced, in memory and on disk."""

    run_id: str
    experiment: str
    model_name: str
    seed: int
    split_name: str
    target_set: str
    run_dir: Path
    history: MetricHistory
    thresholds: ThresholdSet
    label_names: list[str]
    predictions: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    per_label: pd.DataFrame | None = None
    model_description: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def test(self) -> dict[str, np.ndarray]:
        """Test-partition predictions."""
        return self.predictions["test"]

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, run_dir: str | Path) -> RunResult:
        """Reconstruct a finished run from its artefacts.

        Lets the uncertainty and explainability notebooks work from saved runs
        instead of retraining, which matters when training happened on Colab.
        """
        run_dir = Path(run_dir)
        metadata = JsonIO.read(run_dir / "run_metadata.json")
        config = metadata.get("config", {})
        thresholds_frame = pd.read_csv(run_dir / "thresholds.csv")
        label_names = thresholds_frame["label"].tolist()

        predictions: dict[str, dict[str, np.ndarray]] = {}
        for partition in ("val", "calibration", "test"):
            path = run_dir / f"{partition}_predictions.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            predictions[partition] = {
                "logits": frame[[f"logit_{c}" for c in label_names]].to_numpy(),
                "probabilities": frame[[f"prob_{c}" for c in label_names]].to_numpy(),
                "targets": frame[[f"true_{c}" for c in label_names]].to_numpy(),
                "row_uid": frame["row_uid"].to_numpy(),
                "patient_id": frame["patient_id"].to_numpy(),
            }

        history = MetricHistory()
        history_path = run_dir / f"{run_dir.name}_history.json"
        if history_path.exists():
            history.records = JsonIO.read(history_path)

        per_label_path = run_dir / "test_per_label.csv"
        metrics_path = run_dir / "test_metrics.json"
        description_path = run_dir / "model_description.json"
        return cls(
            run_id=metadata.get("run_id", run_dir.name),
            experiment=metadata.get("experiment", ""),
            model_name=config.get("model", {}).get("name", ""),
            seed=metadata.get("seed", 0),
            split_name=config.get("split", {}).get("strategy", "holdout"),
            target_set=config.get("data", {}).get("target_set", ""),
            run_dir=run_dir,
            history=history,
            thresholds=ThresholdSet(
                thresholds=thresholds_frame["threshold"].to_numpy(), label_names=label_names
            ),
            label_names=label_names,
            predictions=predictions,
            test_metrics=JsonIO.read(metrics_path) if metrics_path.exists() else {},
            per_label=pd.read_csv(per_label_path) if per_label_path.exists() else None,
            model_description=JsonIO.read(description_path) if description_path.exists() else {},
        )

    @staticmethod
    def discover(runs_dir: str | Path) -> list[Path]:
        """Directories under ``runs_dir`` that hold a completed run."""
        return sorted(
            path
            for path in Path(runs_dir).glob("*")
            if path.is_dir() and (path / "run_metadata.json").exists()
        )

    @classmethod
    def load_all(cls, runs_dir: str | Path) -> list[RunResult]:
        """Load every completed run under ``runs_dir``."""
        return [cls.load(path) for path in cls.discover(runs_dir)]

    def summary_row(self) -> dict[str, Any]:
        """One row for the cross-model comparison table."""
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "model": self.model_name,
            "seed": self.seed,
            "split": self.split_name,
            "target_set": self.target_set,
            "n_parameters": self.model_description.get("n_trainable_parameters"),
            "epochs_run": len(self.history.records),
            "minutes": round(self.duration_s / 60, 1),
        }
        row.update({k: v for k, v in self.test_metrics.items() if isinstance(v, (int, float))})
        return row


class ExperimentRunner:
    """Runs one model on one split, from data assembly to frozen test metrics.

    Args:
        pipeline: A configured :class:`OlivesPipeline`.
        config: Experiment config; ``config.model.name`` picks the architecture.
        output_root: Where run directories are created.
    """

    def __init__(
        self,
        pipeline: OlivesPipeline,
        config: ExperimentConfig | None = None,
        output_root: str | Path | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or pipeline.config
        self.output_root = Path(output_root) if output_root else pipeline.output_dir / "runs"
        self._trainer: Trainer | None = None
        self._data: OlivesDataModule | None = None
        self._model: Any = None

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    @property
    def needs_images(self) -> bool:
        """Whether the configured model consumes images."""
        return self.config.model.name != "clinical_only"

    def prepare_clinical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the clinical control ladder, then derive within-eye contrasts.

        Order matters. A perturbation replaces the raw BCVA/CST values, so it has
        to run *before* the deltas are computed -- otherwise a control arm would
        keep honest deltas alongside perturbed absolutes and would no longer test
        what it claims to.

        Both steps are no-ops under the default configuration, so existing runs
        are unaffected.
        """
        data = self.config.data
        features = tuple(f for f in ("bcva", "cst") if f in frame.columns)

        if data.is_control_arm:
            frame = ClinicalPerturbation(
                mode=data.clinical_perturbation,
                features=features,
                seed=data.clinical_perturbation_seed,
                bins=data.clinical_bins,
            ).transform(frame)

        if data.longitudinal_clinical:
            frame = LongitudinalClinicalFeatures(features=features).transform(frame)
            requested = set(self.config.model.clinical_features)
            available = set(frame.columns)
            unknown = sorted(requested - available)
            if unknown:
                raise KeyError(
                    f"model.clinical_features names {unknown}, which the frame does not provide. "
                    f"Derived columns available: {LongitudinalClinicalFeatures(features=features).derived_names()}"
                )
        return frame

    def build_data_module(
        self, frame: pd.DataFrame, assignment: SplitAssignment, label_columns: list[str]
    ) -> OlivesDataModule:
        """Assemble loaders, fitting clinical statistics on the training fold only."""
        frame = self.prepare_clinical_frame(frame)
        reader = None
        if self.needs_images and "cache_path" not in frame.columns:
            reader = ParquetImageReader(
                self.pipeline.data_root, self.config.data.config_name, "train"
            )
            LOGGER.warning(
                "no image cache found; reading images straight from parquet, which is much "
                "slower. Run ImageCacheExporter once to avoid this."
            )
        data = self.config.data
        training = self.config.training
        transform_factory = ImageTransformFactory(
            image_size=data.image_size,
            image_mode=data.image_mode,
            crop_retina=data.crop_retina,
            crop_threshold=data.crop_threshold,
            crop_padding=data.crop_padding,
            preserve_aspect_ratio=data.preserve_aspect_ratio,
            normalization=data.normalization,
            horizontal_flip=data.horizontal_flip,
        )
        return OlivesDataModule(
            frame=frame,
            assignment=assignment,
            label_columns=label_columns,
            transform_factory=transform_factory,
            clinical_features=self.config.model.clinical_features,
            use_missingness_indicators=self.config.model.use_missingness_indicators,
            reader=reader,
            return_image=self.needs_images,
            batch_size=training.batch_size,
            num_workers=data.num_workers if self.needs_images else 0,
            group_key=data.group_key,
            image_mode=data.image_mode,
            normalization_samples=data.normalization_samples,
            sampler=training.sampler,
            samples_per_epoch=training.samples_per_epoch,
            rare_positive_sampling_power=training.rare_positive_sampling_power,
        ).setup()

    def build_criterion(self, pos_weight: np.ndarray | None = None) -> Any:
        """Build the configured loss, passing its own hyperparameters through.

        ``pos_weight`` applies to weighted BCE only. Asymmetric loss handles
        imbalance through its own gamma/clip terms instead, so passing class
        weights as well would double-count the correction.
        """
        training = self.config.training
        name = training.loss
        if name == "asl":
            return LossFactory().build(
                "asl",
                gamma_negative=training.asl_gamma_negative,
                gamma_positive=training.asl_gamma_positive,
                clip=training.asl_clip,
            )
        if name == "bce":
            return LossFactory().build("bce", pos_weight=pos_weight)
        return LossFactory().build(name)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(
        self,
        manifest: Manifest | None = None,
        assignment: SplitAssignment | None = None,
        frame: pd.DataFrame | None = None,
        seed: int | None = None,
        run_id: str | None = None,
        save: bool = True,
    ) -> RunResult:
        """Train, tune thresholds on validation, then evaluate once on test."""
        started = time.time()
        seed = self.config.project.seed if seed is None else seed
        SeedManager(seed).apply()

        manifest = manifest or self.pipeline.get_manifest()
        if frame is None:
            frame = self.pipeline.modelling_frame(manifest, attach_cache=self.needs_images)
        assignment = assignment or self.pipeline.make_holdout_split(manifest, write=False)

        run_id = run_id or (
            f"{self.config.experiment.name}_{assignment.name}_seed{seed}"
            f"_{datetime.now().strftime('%H%M%S')}"
        )
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        data = self.build_data_module(frame, assignment, manifest.label_columns)
        model = ModelFactory().build(
            self.config.model,
            n_labels=len(manifest.label_columns),
            clinical_dim=data.preprocessor.output_dim,
        )
        LOGGER.info(
            "run %s | model=%s | %d trainable params | device=%s",
            run_id,
            self.config.model.name,
            model.n_parameters(),
            self.pipeline.env.device,
        )

        training = self.config.training
        # Class weights from the training fold. Aggregating at patient or visit
        # level first stops a patient with many B-scans from dominating them.
        pos_weight = (
            data.pos_weight(training.pos_weight_cap, unit=training.pos_weight_unit)
            if training.pos_weight == "from_train_fold"
            else None
        )
        trainer = Trainer(
            model=model,
            criterion=self.build_criterion(pos_weight),
            config=self.config.training,
            device=self.pipeline.env.device,
            checkpoint_dir=run_dir / "checkpoints",
            run_id=run_id,
            label_names=manifest.label_columns,
        )
        trainer.fit(data.dataloader("train"), data.dataloader("val", shuffle=False))
        trainer.load_best()

        predictions: dict[str, dict[str, np.ndarray]] = {}
        for partition in ("val", "calibration", "test"):
            if partition in data.datasets:
                predictions[partition] = trainer.predict(
                    data.dataloader(partition, shuffle=False)
                )

        thresholds = ThresholdOptimizer(objective="f1").fit(
            predictions["val"]["targets"],
            predictions["val"]["probabilities"],
            manifest.label_columns,
        )
        metrics = MultiLabelMetrics(label_names=manifest.label_columns)
        test_metrics = metrics.compute(
            predictions["test"]["targets"],
            predictions["test"]["probabilities"],
            thresholds.as_array(),
        )
        per_label = metrics.per_label(
            predictions["test"]["targets"],
            predictions["test"]["probabilities"],
            thresholds.as_array(),
        )

        result = RunResult(
            run_id=run_id,
            experiment=self.config.experiment.name,
            model_name=self.config.model.name,
            seed=seed,
            split_name=assignment.name,
            target_set=self.config.data.target_set,
            run_dir=run_dir,
            history=trainer.history,
            thresholds=thresholds,
            label_names=manifest.label_columns,
            predictions=predictions,
            test_metrics=test_metrics,
            per_label=per_label,
            model_description=model.describe(),
            duration_s=time.time() - started,
        )
        self._trainer = trainer
        self._data = data
        self._model = model

        if save:
            self.save(result, data)
        LOGGER.info(
            "run %s finished in %.1f min | test macro_f1=%.4f macro_auprc=%.4f",
            run_id,
            result.duration_s / 60,
            test_metrics["macro_f1"],
            test_metrics["macro_auprc"],
        )
        return result

    # ------------------------------------------------------------------
    def save(self, result: RunResult, data: OlivesDataModule) -> None:
        """Write every artefact needed to reproduce and audit the run."""
        run_dir = result.run_dir
        result.history.save(run_dir, result.run_id)
        result.thresholds.to_frame().to_csv(run_dir / "thresholds.csv", index=False)
        if result.per_label is not None:
            result.per_label.to_csv(run_dir / "test_per_label.csv", index=False)
        JsonIO.write(result.test_metrics, run_dir / "test_metrics.json")

        for partition, payload in result.predictions.items():
            frame = pd.DataFrame(
                {"row_uid": payload["row_uid"], "patient_id": payload["patient_id"]}
            )
            for index, label in enumerate(result.label_names):
                frame[f"logit_{label}"] = payload["logits"][:, index]
                frame[f"prob_{label}"] = payload["probabilities"][:, index]
                frame[f"true_{label}"] = payload["targets"][:, index]
            frame.to_parquet(run_dir / f"{partition}_predictions.parquet", index=False)

        data.preprocessor.save(run_dir / "clinical_preprocessor.json")
        JsonIO.write(result.model_description, run_dir / "model_description.json")
        self.config.save(run_dir / "resolved_config.yaml")
        RunMetadata(
            run_id=result.run_id,
            experiment=result.experiment,
            seed=result.seed,
            git_commit=RunMetadata.current_git_commit(self.pipeline.env.repo_root),
            device=self.pipeline.env.device,
            config=self.config.to_dict(),
            package_versions=RunMetadata.collect_package_versions(),
            manifest_hash=RunMetadata.hash_file(self.pipeline.manifest_path()),
        ).to_json(run_dir / "run_metadata.json")

    # ------------------------------------------------------------------
    def _require_run(self, attribute: str) -> Any:
        value = getattr(self, attribute)
        if value is None:
            raise RuntimeError("call run() before accessing the trained artefacts")
        return value

    @property
    def trainer(self) -> Trainer:
        """The trainer from the most recent :meth:`run`."""
        return self._require_run("_trainer")

    @property
    def data_module(self) -> OlivesDataModule:
        """The data module from the most recent :meth:`run`."""
        return self._require_run("_data")

    @property
    def model(self) -> Any:
        """The model from the most recent :meth:`run`."""
        return self._require_run("_model")


class RunEvaluator:
    """Post-hoc analysis of a completed run.

    Kept separate from training so it can be re-run without retraining, and so
    the calibration-fitting partition is an explicit argument rather than an
    implicit choice buried in the training loop.
    """

    def __init__(
        self,
        result: RunResult,
        model: Any,
        data: OlivesDataModule,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.result = result
        self.model = model
        self.data = data
        self.device = device
        self.seed = seed
        self.scaler: TemperatureScaler | None = None
        self.uncertainty: UncertaintyOutput | None = None
        self._calibrated_thresholds: ThresholdSet | None = None

    def calibrate(self) -> pd.DataFrame:
        """Fit temperature scaling on the calibration partition, score on test."""
        if "calibration" not in self.result.predictions:
            raise KeyError("run has no calibration partition; check the split configuration")
        calibration = self.result.predictions["calibration"]
        self.scaler = TemperatureScaler().fit(
            calibration["logits"], calibration["targets"], self.result.label_names
        )
        self.scaler.save(str(self.result.run_dir / "temperature.json"))

        test = self.result.test
        comparison = CalibrationEvaluator().compare(
            test["targets"],
            test["probabilities"],
            self.scaler.transform(test["logits"]),
            self.result.label_names,
        )
        comparison.to_csv(self.result.run_dir / "calibration_comparison.csv", index=False)
        return comparison

    def calibrated_test_probabilities(self) -> np.ndarray:
        """Test probabilities after temperature scaling (raw if not calibrated)."""
        if self.scaler is None:
            return self.result.test["probabilities"]
        return self.scaler.transform(self.result.test["logits"])

    def refit_thresholds_on_calibrated(self) -> ThresholdSet:
        """Refit per-label thresholds against calibrated validation probabilities.

        Thresholds are originally fitted on *uncalibrated* validation output. Once
        temperature scaling moves the probabilities, those cut points sit at
        different places on the new scale, so any threshold-dependent metric
        (F1, precision, recall, Hamming) becomes inconsistent.

        Refitting on the calibrated validation partition restores that
        consistency. It is still validation-only, so test remains untouched.
        Threshold-free metrics (AUROC, AUPRC) are unaffected either way.
        """
        if self.scaler is None:
            raise RuntimeError("call calibrate() before refitting thresholds")
        validation = self.result.predictions["val"]
        calibrated = self.scaler.transform(validation["logits"])
        thresholds = ThresholdOptimizer(objective="f1").fit(
            validation["targets"], calibrated, self.result.label_names
        )
        thresholds.to_frame().to_csv(
            self.result.run_dir / "thresholds_calibrated.csv", index=False
        )
        self._calibrated_thresholds = thresholds
        LOGGER.info("thresholds refitted on calibrated validation probabilities")
        return thresholds

    @property
    def active_thresholds(self) -> np.ndarray:
        """Thresholds matching whichever probability scale is in use."""
        calibrated = getattr(self, "_calibrated_thresholds", None)
        if self.scaler is not None and calibrated is not None:
            return calibrated.as_array()
        return self.result.thresholds.as_array()

    def estimate_uncertainty(self, n_passes: int = 30) -> UncertaintyOutput:
        """Run MC dropout over the test partition."""
        self.uncertainty = MCDropoutInference(
            self.model, device=self.device, n_passes=n_passes
        ).run(self.data.dataloader("test", shuffle=False))
        self.uncertainty.to_frame(self.result.label_names).to_parquet(
            self.result.run_dir / "test_uncertainty.parquet", index=False
        )
        return self.uncertainty

    def selective_prediction(
        self, coverage_levels: Sequence[float] | None = None
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Coverage/performance curve and the uncertainty-error association."""
        if self.uncertainty is None:
            raise RuntimeError("call estimate_uncertainty() first")
        test = self.result.test
        probabilities = self.calibrated_test_probabilities()
        thresholds = self.active_thresholds
        selector = SelectivePredictor(label_names=self.result.label_names)

        curve = selector.coverage_curve(
            test["targets"],
            probabilities,
            self.uncertainty.total_uncertainty(),
            thresholds,
            list(coverage_levels) if coverage_levels else None,
        )
        curve.to_csv(self.result.run_dir / "coverage_curve.csv", index=False)
        association = selector.uncertainty_error_association(
            test["targets"], probabilities, self.uncertainty.total_uncertainty(), thresholds
        )
        JsonIO.write(association, self.result.run_dir / "uncertainty_error_association.json")
        return curve, association

    def bootstrap(
        self,
        n_iterations: int = 1000,
        metrics: tuple[str, ...] = ("macro_f1", "macro_auroc", "macro_auprc"),
        use_calibrated: bool = True,
    ) -> pd.DataFrame:
        """Patient-level bootstrap confidence intervals on the test partition."""
        test = self.result.test
        probabilities = (
            self.calibrated_test_probabilities() if use_calibrated else test["probabilities"]
        )
        table = PatientBootstrap(n_iterations=n_iterations, seed=self.seed).run_many(
            test["targets"],
            probabilities,
            test["patient_id"],
            self.active_thresholds if use_calibrated else self.result.thresholds.as_array(),
            self.result.label_names,
            metrics,
        )
        table.insert(0, "run_id", self.result.run_id)
        table.to_csv(self.result.run_dir / "bootstrap_ci.csv", index=False)
        return table

    def run_all(
        self, n_passes: int = 30, n_bootstrap: int = 1000
    ) -> dict[str, Any]:
        """Calibration, uncertainty, selective prediction and bootstrap in order."""
        calibration = self.calibrate()
        self.refit_thresholds_on_calibrated()
        self.estimate_uncertainty(n_passes)
        curve, association = self.selective_prediction()
        bootstrap = self.bootstrap(n_bootstrap)
        return {
            "calibration": calibration,
            "coverage": curve,
            "association": association,
            "bootstrap": bootstrap,
        }


class ExperimentSuite:
    """Runs the same recipe across several models, seeds or folds.

    Comparisons are only meaningful when every arm sees identical partitions and
    an identical training budget, so the suite holds the split and manifest fixed
    and varies only the model and seed.
    """

    def __init__(self, pipeline: OlivesPipeline, output_root: str | Path | None = None) -> None:
        self.pipeline = pipeline
        self.output_root = Path(output_root) if output_root else pipeline.output_dir / "runs"
        self.results: list[RunResult] = []

    def run_models(
        self,
        configs: dict[str, ExperimentConfig],
        assignment: SplitAssignment,
        manifest: Manifest,
        seeds: Iterable[int] = (42,),
        evaluate: bool = False,
        n_passes: int = 30,
        n_bootstrap: int = 1000,
    ) -> list[RunResult]:
        """Train every config at every seed on one fixed split."""
        results: list[RunResult] = []
        frames: dict[bool, pd.DataFrame] = {}

        for seed in seeds:
            for name, config in configs.items():
                runner = ExperimentRunner(self.pipeline, config, self.output_root)
                if runner.needs_images not in frames:
                    frames[runner.needs_images] = self.pipeline.modelling_frame(
                        manifest, attach_cache=runner.needs_images
                    )
                LOGGER.info("=== %s | seed %d | split %s ===", name, seed, assignment.name)
                result = runner.run(
                    manifest=manifest,
                    assignment=assignment,
                    frame=frames[runner.needs_images],
                    seed=seed,
                    run_id=f"{name}_{assignment.name}_seed{seed}",
                )
                if evaluate:
                    RunEvaluator(
                        result,
                        runner.model,
                        runner.data_module,
                        device=self.pipeline.env.device,
                        seed=seed,
                    ).run_all(n_passes=n_passes, n_bootstrap=n_bootstrap)
                results.append(result)
                self.results.append(result)
        return results

    def run_folds(
        self,
        config: ExperimentConfig,
        folds: Sequence[SplitAssignment],
        manifest: Manifest,
        seed: int = 42,
    ) -> list[RunResult]:
        """Train one config across every fold of a patient-grouped CV."""
        runner = ExperimentRunner(self.pipeline, config, self.output_root)
        frame = self.pipeline.modelling_frame(manifest, attach_cache=runner.needs_images)
        results = []
        for fold in folds:
            LOGGER.info("=== %s | %s ===", config.experiment.name, fold.name)
            result = runner.run(
                manifest=manifest,
                assignment=fold,
                frame=frame,
                seed=seed,
                run_id=f"{config.experiment.name}_{fold.name}_seed{seed}",
            )
            results.append(result)
            self.results.append(result)
        return results


class ResultsAggregator:
    """Turns a collection of runs into the tables the report needs."""

    def __init__(self, results: Sequence[RunResult]) -> None:
        self.results = list(results)

    def comparison(self, sort_by: str = "macro_auprc") -> pd.DataFrame:
        """Headline table: one row per run."""
        table = pd.DataFrame([r.summary_row() for r in self.results])
        if sort_by in table.columns:
            table = table.sort_values(sort_by, ascending=False)
        return table.reset_index(drop=True)

    def across_seeds(self, metrics: Sequence[str] = ("macro_f1", "macro_auroc", "macro_auprc")) -> pd.DataFrame:
        """Mean and spread per model across seeds or folds.

        A single-seed difference is not evidence. This is the table that decides
        whether one model actually beats another.
        """
        table = self.comparison()
        available = [m for m in metrics if m in table.columns]
        grouped = table.groupby("model")[available].agg(["mean", "std", "count"])
        grouped.columns = ["_".join(c) for c in grouped.columns]
        return grouped.round(4).reset_index()

    def per_label(self) -> pd.DataFrame:
        """Per-label metrics for every run, stacked."""
        frames = []
        for result in self.results:
            if result.per_label is None:
                continue
            frame = result.per_label.copy()
            frame.insert(0, "model", result.model_name)
            frame.insert(0, "run_id", result.run_id)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def per_label_pivot(self, metric: str = "auprc") -> pd.DataFrame:
        """Labels as rows, models as columns, for one metric."""
        table = self.per_label()
        if table.empty:
            return table
        pivot = table.pivot_table(index="label", columns="model", values=metric, aggfunc="mean")
        counts = table.groupby("label")["n_positive"].first()
        pivot.insert(0, "n_positive", counts)
        return pivot.sort_values("n_positive", ascending=False).round(4)

    def collect_bootstrap(self) -> pd.DataFrame:
        """Read back every run's bootstrap table."""
        frames = []
        for result in self.results:
            path = Path(result.run_dir) / "bootstrap_ci.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            # Some writers already stamp the model name; do not duplicate the column.
            if "model" in frame.columns:
                frame["model"] = result.model_name
            else:
                frame.insert(min(1, frame.shape[1]), "model", result.model_name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def paired_difference(
        self,
        model_a: str,
        model_b: str,
        seed: int | None = None,
        metrics: Sequence[str] = ("macro_f1", "macro_auroc", "macro_auprc"),
        n_iterations: int = 2000,
        partition: str = "test",
    ) -> pd.DataFrame:
        """Bootstrap the difference between two models on the same patients.

        Prefer this to :meth:`intervals_overlap`. Resampling the *difference*
        cancels patient difficulty, which is shared by both arms and accounts for
        most of the width of each marginal interval. Two overlapping intervals
        routinely hide a difference that holds in every resample.

        Args:
            model_a: Model name of the first arm, e.g. ``"residual_logit_fusion"``.
            model_b: Model name of the reference arm, e.g. ``"oct_only"``.
            seed: Restrict to one seed. Required when several exist, since runs
                from different seeds are not a matched pair.
            partition: Which partition's predictions to compare.
        """
        from olives_biomarkers.evaluation.comparison import PairedPatientBootstrap

        run_a = self._single_run(model_a, seed)
        run_b = self._single_run(model_b, seed)
        for run in (run_a, run_b):
            if partition not in run.predictions:
                raise KeyError(
                    f"run '{run.run_id}' has no stored {partition} predictions; "
                    "re-run it or load from its run directory"
                )
        if run_a.label_names != run_b.label_names:
            raise ValueError("the two runs use different label sets and cannot be compared")

        left, right = run_a.predictions[partition], run_b.predictions[partition]
        test = PairedPatientBootstrap(n_iterations=n_iterations)
        order_a, order_b = test.align(left["row_uid"], right["row_uid"])
        return test.run_many(
            targets=left["targets"][order_a],
            probabilities_a=left["probabilities"][order_a],
            probabilities_b=right["probabilities"][order_b],
            patient_ids=left["patient_id"][order_a],
            thresholds_a=run_a.thresholds.as_array(),
            thresholds_b=run_b.thresholds.as_array(),
            label_names=run_a.label_names,
            metrics=tuple(metrics),
            arm_a=model_a,
            arm_b=model_b,
        )

    def paired_per_label(
        self,
        model_a: str,
        model_b: str,
        seed: int | None = None,
        preregistered: Sequence[str] | None = None,
        n_iterations: int = 2000,
        partition: str = "test",
    ) -> pd.DataFrame:
        """Per-biomarker paired differences, split into confirmatory and exploratory.

        When the hypothesis names specific labels, this is the table to report.
        A macro average over thirteen biomarkers dilutes an effect confined to
        three of them by a factor of four.
        """
        from olives_biomarkers.evaluation.comparison import PairedPatientBootstrap

        run_a = self._single_run(model_a, seed)
        run_b = self._single_run(model_b, seed)
        left, right = run_a.predictions[partition], run_b.predictions[partition]
        test = PairedPatientBootstrap(n_iterations=n_iterations)
        order_a, order_b = test.align(left["row_uid"], right["row_uid"])
        table = test.per_label(
            targets=left["targets"][order_a],
            probabilities_a=left["probabilities"][order_a],
            probabilities_b=right["probabilities"][order_b],
            patient_ids=left["patient_id"][order_a],
            label_names=run_a.label_names,
            arm_a=model_a,
            arm_b=model_b,
        )
        if preregistered is None:
            return table
        return test.confirmatory_report(table, list(preregistered))

    def _single_run(self, model_name: str, seed: int | None) -> RunResult:
        """Find exactly one run for a model, or explain why it is ambiguous."""
        matches = [r for r in self.results if r.model_name == model_name]
        if seed is not None:
            matches = [r for r in matches if r.seed == seed]
        if not matches:
            available = sorted({r.model_name for r in self.results})
            raise KeyError(f"no run for model {model_name!r}; loaded models: {available}")
        if len(matches) > 1:
            seeds = sorted(r.seed for r in matches)
            raise ValueError(
                f"{len(matches)} runs match {model_name!r} (seeds {seeds}); pass seed= to pick "
                "one. Runs from different seeds are not a matched pair."
            )
        return matches[0]

    def intervals_overlap(self, metric: str = "macro_auprc") -> pd.DataFrame:
        """Pairwise check of whether two models' confidence intervals overlap.

        Kept for the descriptive table only. As a *test* it is both wrong and the
        least powerful option available: it discards the pairing, so patient
        difficulty inflates both intervals independently. Use
        :meth:`paired_difference` to decide whether one model beats another.
        """
        bootstrap = self.collect_bootstrap()
        if bootstrap.empty:
            return bootstrap
        scoped = bootstrap[bootstrap["metric"] == metric].set_index("model")
        rows = []
        for left in scoped.index:
            for right in scoped.index:
                if left >= right:
                    continue
                a, b = scoped.loc[left], scoped.loc[right]
                overlap = not (a["ci_upper"] < b["ci_lower"] or b["ci_upper"] < a["ci_lower"])
                rows.append(
                    {
                        "metric": metric,
                        "model_a": left,
                        "model_b": right,
                        "a": f"{a['point_estimate']:.4f} [{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]",
                        "b": f"{b['point_estimate']:.4f} [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]",
                        "difference": round(a["point_estimate"] - b["point_estimate"], 4),
                        "intervals_overlap": overlap,
                        "conclusion": "no supported difference" if overlap else "separated",
                    }
                )
        return pd.DataFrame(rows)
