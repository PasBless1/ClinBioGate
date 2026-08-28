"""Post-hoc evaluation of a trained run.

Thin wrapper over :class:`RunEvaluator`: temperature scaling fitted on the
patient-disjoint calibration partition, MC dropout uncertainty, selective
prediction, and patient-level bootstrap confidence intervals.

Usage:
    python scripts/evaluate.py --run-dir outputs/runs/local_cpu/<run_id>
    python scripts/evaluate.py --run-dir outputs/runs/<run_id> --mc-passes 30 --bootstrap 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from olives_biomarkers import ExperimentRunner, OlivesPipeline, RunEvaluator, RunResult  # noqa: E402
from olives_biomarkers.config import ConfigLoader  # noqa: E402
from olives_biomarkers.models import ModelFactory  # noqa: E402
from olives_biomarkers.training.callbacks import CheckpointManager  # noqa: E402
from olives_biomarkers.utils.logging import LoggerFactory  # noqa: E402

LOGGER = LoggerFactory.get("olives.evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mc-passes", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Skip MC dropout; calibration and bootstrap still run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    if not (run_dir / "run_metadata.json").exists():
        LOGGER.error("no completed run at %s", run_dir)
        return 2

    result = RunResult.load(run_dir)
    pipeline = OlivesPipeline.from_config(REPO_ROOT / "configs" / "data.yaml", repo_root=REPO_ROOT)
    config = ConfigLoader(REPO_ROOT).load(run_dir / "resolved_config.yaml")
    LOGGER.info("evaluating %s (%s)", result.run_id, config.model.name)

    n_passes = args.mc_passes or config.uncertainty.mc_dropout_passes
    n_bootstrap = args.bootstrap or config.evaluation.bootstrap_iterations

    manifest = pipeline.get_manifest()
    runner = ExperimentRunner(pipeline, config, output_root=run_dir.parent)

    model = None
    data = None
    if not args.no_uncertainty:
        # MC dropout needs the live model, so rebuild it and restore the weights.
        frame = pipeline.modelling_frame(manifest, attach_cache=runner.needs_images)
        assignment = pipeline.make_holdout_split(manifest, write=False)
        data = runner.build_data_module(frame, assignment, manifest.label_columns)
        model = ModelFactory().build(
            config.model,
            n_labels=len(manifest.label_columns),
            clinical_dim=data.preprocessor.output_dim,
        )
        CheckpointManager(run_dir / "checkpoints", run_id=result.run_id).load(
            model, map_location=pipeline.env.device
        )
        model.to(pipeline.env.device)

    evaluator = RunEvaluator(
        result, model, data, device=pipeline.env.device, seed=config.project.seed
    )
    calibration = evaluator.calibrate()
    # Thresholds were fitted on uncalibrated validation output; refit them on the
    # calibrated scale so F1/precision/recall stay coherent with the probabilities.
    evaluator.refit_thresholds_on_calibrated()

    coverage = None
    association = None
    if not args.no_uncertainty:
        evaluator.estimate_uncertainty(n_passes)
        coverage, association = evaluator.selective_prediction(config.evaluation.coverage_levels)

    bootstrap = evaluator.bootstrap(n_bootstrap)

    print()
    print("=" * 70)
    print(f"EVALUATION  ({result.run_id})")
    print("=" * 70)
    print("\nTest metrics with 95% patient-level bootstrap CI:")
    for _, row in bootstrap.iterrows():
        print(
            f"  {row['metric']:>14}: {row['point_estimate']:.4f} "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]  (n={int(row['n_patients'])} patients)"
        )

    reliable = calibration[calibration["reliable"]] if "reliable" in calibration else calibration
    print(f"\nCalibration (mean over {len(reliable)} labels with enough positives):")
    print(f"  {'ECE':>14}: {reliable['ece_pre'].mean():.4f} -> {reliable['ece_post'].mean():.4f}")
    print(f"  {'Brier':>14}: {reliable['brier_pre'].mean():.4f} -> {reliable['brier_post'].mean():.4f}")

    if coverage is not None:
        print("\nSelective prediction:")
        print(
            coverage[["coverage", "n_retained", "macro_f1", "macro_auroc"]].to_string(index=False)
        )
    if association is not None:
        print("\nDoes uncertainty track error?")
        for key, value in association.items():
            print(f"  {key:>32}: {value:.4f}")
        if association["spearman_r"] < 0.1 or association["p_value"] > 0.05:
            print("\n  Uncertainty does NOT reliably track error here; selective prediction")
            print("  cannot be recommended on this evidence.")

    print(f"\nArtefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
