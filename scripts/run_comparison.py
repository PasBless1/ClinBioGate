"""Run the mandatory model comparison (A-D) on one fixed split.

Every arm sees identical partitions and an identical training budget, which is
the only way the comparison means anything.

Usage:
    python scripts/run_comparison.py --budget local_cpu
    python scripts/run_comparison.py --budget colab_gpu --seeds 42 43 44 --evaluate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from olives_biomarkers import ExperimentSuite, OlivesPipeline, ResultsAggregator  # noqa: E402
from olives_biomarkers.config import ConfigLoader  # noqa: E402
from olives_biomarkers.utils.logging import LoggerFactory  # noqa: E402

LOGGER = LoggerFactory.get("olives.comparison")

#: Model id -> (config file stem, registry name).
ARMS = {
    "A": ("baseline_clinical", "clinical_only"),
    "B": ("baseline_oct", "oct_only"),
    "C": ("fusion_concat", "concat_fusion"),
    "D": ("fusion_gated", "gated_fusion"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A-D model comparison.")
    parser.add_argument("--budget", default="local_cpu", help="Budget profile config stem.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), help="Subset of A B C D.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the profile's epochs.")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Also run calibration, MC dropout and bootstrap for each run.",
    )
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=400)
    return parser.parse_args(argv)


def build_configs(loader: ConfigLoader, budget, arms: list[str], epochs: int | None) -> dict:
    """Architecture from each model's config, schedule from the budget profile."""
    configs = {}
    for arm in arms:
        stem, registry_name = ARMS[arm]
        config = loader.load(REPO_ROOT / "configs" / f"{stem}.yaml")
        config.data.image_size = budget.data.image_size
        config.data.num_workers = budget.data.num_workers
        config.training.epochs = epochs or budget.training.epochs
        config.training.batch_size = budget.training.batch_size
        config.training.learning_rate = budget.training.learning_rate
        config.training.early_stopping_patience = budget.training.early_stopping_patience
        config.training.amp = budget.training.amp
        configs[registry_name] = config
    return configs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = OlivesPipeline.from_config(REPO_ROOT / "configs" / "data.yaml", repo_root=REPO_ROOT)
    loader = ConfigLoader(REPO_ROOT)
    budget = loader.load(REPO_ROOT / "configs" / f"{args.budget}.yaml")

    LOGGER.info(
        "budget=%s device=%s image=%s epochs=%s seeds=%s",
        args.budget,
        pipeline.env.device,
        budget.data.image_size,
        args.epochs or budget.training.epochs,
        args.seeds,
    )

    manifest = pipeline.get_manifest()
    if not pipeline.image_cache_dir.exists():
        LOGGER.info("exporting image cache first")
        pipeline.export_image_cache(manifest)

    assignment = pipeline.make_holdout_split(manifest, write=True)
    configs = build_configs(loader, budget, args.arms, args.epochs)

    runs_dir = pipeline.output_dir / "runs" / args.budget
    suite = ExperimentSuite(pipeline, output_root=runs_dir)

    started = time.time()
    results = suite.run_models(
        configs=configs,
        assignment=assignment,
        manifest=manifest,
        seeds=args.seeds,
        evaluate=args.evaluate,
        n_passes=args.mc_passes,
        n_bootstrap=args.bootstrap,
    )
    LOGGER.info("%d runs in %.1f min", len(results), (time.time() - started) / 60)

    aggregator = ResultsAggregator(results)
    comparison = aggregator.comparison()
    comparison.to_csv(runs_dir / "comparison.csv", index=False)
    aggregator.per_label_pivot("auprc").to_csv(runs_dir / "per_label_auprc.csv")

    print()
    print("=" * 78)
    print(f"COMPARISON  (budget={args.budget}, seeds={args.seeds})")
    print("=" * 78)
    columns = [
        c
        for c in ["model", "seed", "epochs_run", "minutes", "macro_f1", "macro_auroc",
                  "macro_auprc", "n_degenerate_labels"]
        if c in comparison.columns
    ]
    print(comparison[columns].to_string(index=False))

    if len(args.seeds) > 1:
        print()
        print(aggregator.across_seeds().to_string(index=False))

    print()
    print("Per-label AUPRC:")
    print(aggregator.per_label_pivot("auprc").to_string())
    print()
    print(f"Artefacts: {runs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
