"""Train one model on the patient-grouped holdout split.

Thin wrapper over :class:`ExperimentRunner`, which owns the train / threshold /
evaluate sequence. Keeping the orchestration in one place is what stops this
script and the notebooks from drifting into incomparable procedures.

Usage:
    python scripts/train.py --config configs/baseline_clinical.yaml
    python scripts/train.py --config configs/baseline_oct.yaml --budget local_cpu
    python scripts/train.py --config configs/fusion_gated.yaml --epochs 2 --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from olives_biomarkers import ExperimentRunner, OlivesPipeline, RunEvaluator  # noqa: E402
from olives_biomarkers.config import ConfigLoader  # noqa: E402
from olives_biomarkers.utils.logging import LoggerFactory  # noqa: E402

LOGGER = LoggerFactory.get("olives.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one biomarker model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--budget", default=None, help="Budget profile stem, e.g. local_cpu.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Also run calibration, MC dropout, selective prediction and bootstrap.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run for wiring checks: 2 epochs, small batches, no pretrained weights.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = OlivesPipeline.from_config(args.config, repo_root=REPO_ROOT)
    config = pipeline.config
    loader = ConfigLoader(REPO_ROOT)

    if args.budget:
        budget = loader.load(REPO_ROOT / "configs" / f"{args.budget}.yaml")
        config.data.image_size = budget.data.image_size
        config.data.num_workers = budget.data.num_workers
        config.training.epochs = budget.training.epochs
        config.training.batch_size = budget.training.batch_size
        config.training.learning_rate = budget.training.learning_rate
        config.training.early_stopping_patience = budget.training.early_stopping_patience
        config.training.amp = budget.training.amp
        LOGGER.info("budget profile %s applied", args.budget)

    if args.seed is not None:
        config.project.seed = args.seed
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.smoke:
        config.training.epochs = min(config.training.epochs, 2)
        config.training.batch_size = 8
        config.model.pretrained = False
        LOGGER.warning("smoke mode: results are meaningless, this only checks the wiring")

    if not pipeline.data_root.exists():
        LOGGER.error(
            "OLIVES data not found at %s. Download 'gOLIVES/OLIVES_Dataset' there, or edit "
            "data.root in %s.",
            pipeline.data_root,
            args.config,
        )
        return 2

    manifest = pipeline.get_manifest()
    if config.model.name != "clinical_only" and not pipeline.image_cache_dir.exists():
        LOGGER.info("exporting the image cache first (one-off)")
        pipeline.export_image_cache(manifest)

    runner = ExperimentRunner(pipeline, config)
    result = runner.run(manifest=manifest, seed=config.project.seed, run_id=args.run_id)

    if args.evaluate:
        RunEvaluator(
            result,
            runner.model,
            runner.data_module,
            device=pipeline.env.device,
            seed=config.project.seed,
        ).run_all(
            n_passes=config.uncertainty.mc_dropout_passes,
            n_bootstrap=config.evaluation.bootstrap_iterations,
        )

    print()
    print("=" * 66)
    print(f"TEST RESULTS  ({result.run_id})")
    print("=" * 66)
    for key, value in result.test_metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {key:>22}: {formatted}")
    print()
    if result.per_label is not None:
        print(
            result.per_label[["label", "n_positive", "auroc", "auprc", "f1"]].to_string(index=False)
        )
    print()
    print(f"Artefacts: {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
