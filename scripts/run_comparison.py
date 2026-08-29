"""Run a model comparison on one fixed split.

Every arm sees identical partitions, so the comparison means something.

Arms A-D are the original mandatory comparison (clinical / OCT / concat / gated).
Arms E-H are the improvement plan: a stronger OCT reference, 2.5D input, and the
two bounded fusion designs that replace the near-saturated multiplicative gate.

E-H carry their own resolution and fine-tuning schedule, which is the point of
them, so the generic budget profile is not forced onto those arms unless you pass
--no-budget-override.

Usage:
    python scripts/run_comparison.py --budget colab_gpu --seeds 42 43 44 --evaluate
    python scripts/run_comparison.py --arms B E --seeds 42 43 44 --ensemble
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
from olives_biomarkers.evaluation import SeedEnsemble  # noqa: E402
from olives_biomarkers.config import ConfigLoader  # noqa: E402
from olives_biomarkers.utils.logging import LoggerFactory  # noqa: E402

LOGGER = LoggerFactory.get("olives.comparison")

#: Arm id -> (config file stem, run label used in output paths).
#
#: A-D are the original mandatory comparison. E-H are the improvement plan: a
#: stronger OCT reference and the two bounded fusion designs that replace the
#: near-saturated multiplicative gate.
ARMS = {
    "A": ("baseline_clinical", "clinical_only"),
    "B": ("baseline_oct", "oct_only"),
    "C": ("fusion_concat", "concat_fusion"),
    "D": ("fusion_gated", "gated_fusion"),
    "E": ("oct_improved", "oct_improved"),
    "F": ("oct_adjacent", "oct_adjacent"),
    "G": ("fusion_residual_logit", "residual_logit_fusion"),
    "H": ("fusion_film", "bounded_film_fusion"),
    "I": ("fusion_longitudinal", "fusion_longitudinal"),
    "J": ("fusion_delta_only", "fusion_delta_only"),
    "K": ("control_patient_mean", "control_patient_mean"),
    "L": ("control_within_shuffle", "control_within_shuffle"),
    "M": ("control_across_shuffle", "control_across_shuffle"),
    "N": ("control_quantise", "control_quantise"),
}

#: The perturbation ladder. These are controls, not models: each destroys one
#: candidate explanation for a fusion gain. They are only interpretable against
#: arm I, so --controls pulls I and E in with them.
CONTROL_ARMS = ("K", "L", "M", "N")

#: Arms that must beat this reference for the fusion hypothesis to hold.
REFERENCE_ARM_MODEL = "oct_improved"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A-D model comparison.")
    parser.add_argument("--budget", default="local_cpu", help="Budget profile config stem.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["A", "B", "C", "D"],
        help=(
            "Subset of A-N. A-D original comparison; E-H improvement plan; "
            "I-J within-eye clinical context; K-N the control ladder."
        ),
    )
    parser.add_argument(
        "--controls",
        action="store_true",
        help="Add the clinical control ladder (K-N) plus arms E and I, which they are read against.",
    )
    parser.add_argument(
        "--preregistered",
        nargs="*",
        default=None,
        help=(
            "Labels named in advance as the confirmatory set, from "
            "scripts/preregister_targets.py. Splits the per-label table into "
            "confirmatory and exploratory rows."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the profile's epochs.")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Also run calibration, MC dropout and bootstrap for each run.",
    )
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Also average each arm across seeds and report the ensemble. Needs >1 seed.",
    )
    parser.add_argument(
        "--no-budget-override",
        action="store_true",
        help="Keep each arm's own schedule instead of forcing the budget profile onto it.",
    )
    return parser.parse_args(argv)


#: Arms whose config already specifies a deliberate resolution and schedule.
#: Forcing the generic budget profile onto these would undo the very changes
#: being tested, so they keep their own settings.
SELF_TUNED_ARMS = frozenset({"E", "F", "G", "H", "I", "J", "K", "L", "M", "N"})


def build_configs(
    loader: ConfigLoader,
    budget,
    arms: list[str],
    epochs: int | None,
    respect_own_schedule: bool = True,
) -> dict:
    """Architecture from each model's config, schedule from the budget profile."""
    configs = {}
    for arm in arms:
        stem, label = ARMS[arm]
        config = loader.load(REPO_ROOT / "configs" / f"{stem}.yaml")
        if not (respect_own_schedule and arm in SELF_TUNED_ARMS):
            config.data.image_size = budget.data.image_size
            config.data.num_workers = budget.data.num_workers
            config.training.epochs = budget.training.epochs
            config.training.batch_size = budget.training.batch_size
            config.training.learning_rate = budget.training.learning_rate
            config.training.early_stopping_patience = budget.training.early_stopping_patience
            config.training.amp = budget.training.amp
        if epochs is not None:
            config.training.epochs = epochs
        configs[label] = config
    return configs


def report_paired(aggregator, results, seed: int, preregistered, runs_dir) -> None:
    """Paired difference of every arm against the OCT reference, plus the ladder.

    This is the test that decides the fusion question. Comparing two marginal
    intervals for overlap discards the pairing and is far less powerful: patient
    difficulty is shared by both arms and cancels in the difference.
    """
    models = sorted({r.model_name for r in results})
    if REFERENCE_ARM_MODEL not in models:
        LOGGER.info(
            "no '%s' arm in this run, so no paired comparison; add arm E to enable it",
            REFERENCE_ARM_MODEL,
        )
        return
    contenders = [m for m in models if m != REFERENCE_ARM_MODEL]
    if not contenders:
        return

    print()
    print("=" * 78)
    print(f"PAIRED DIFFERENCES vs {REFERENCE_ARM_MODEL}  (seed {seed}, same patients)")
    print("=" * 78)
    tables = []
    for model in contenders:
        try:
            table = aggregator.paired_difference(model, REFERENCE_ARM_MODEL, seed=seed)
        except (KeyError, ValueError) as error:
            LOGGER.warning("skipping %s: %s", model, error)
            continue
        tables.append(table)
    if not tables:
        return
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(runs_dir / "paired_differences.csv", index=False)
    columns = ["arm_a", "metric", "estimate_a", "estimate_b", "difference",
               "ci_lower", "ci_upper", "p_two_sided", "conclusion"]
    print(combined[columns].round(4).to_string(index=False))

    control_models = [ARMS[a][1] for a in CONTROL_ARMS if ARMS[a][1] in models]
    if control_models and "fusion_longitudinal" in models:
        print()
        print("Control ladder reading:")
        headline = combined[
            (combined["metric"] == "macro_auprc")
            & (combined["arm_a"] == "fusion_longitudinal")
        ]
        gain = float(headline["difference"].iloc[0]) if len(headline) else float("nan")
        print(f"  fusion_longitudinal - {REFERENCE_ARM_MODEL} = {gain:+.4f} macro AUPRC")
        for control in control_models:
            row = combined[
                (combined["metric"] == "macro_auprc") & (combined["arm_a"] == control)
            ]
            if not len(row):
                continue
            value = float(row["difference"].iloc[0])
            retained = value / gain if abs(gain) > 1e-9 else float("nan")
            print(f"  {control:<26} {value:+.4f}   retains {retained:6.1%} of the gain")
        print(
            "  A control that retains most of the gain has NOT been passed: it means "
            "the thing it destroys was not what the fusion arm was using."
        )

    if preregistered is not None and "fusion_longitudinal" in models:
        print()
        print("=" * 78)
        print("PER-LABEL, CONFIRMATORY vs EXPLORATORY")
        print("=" * 78)
        per_label = aggregator.paired_per_label(
            "fusion_longitudinal", REFERENCE_ARM_MODEL, seed=seed,
            preregistered=preregistered,
        )
        per_label.to_csv(runs_dir / "paired_per_label.csv", index=False)
        show = ["label", "role", "n_positive", "estimate_a", "estimate_b",
                "difference", "ci_lower", "ci_upper", "p_two_sided"]
        print(per_label[[c for c in show if c in per_label.columns]].round(4).to_string(index=False))


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

    arms = list(dict.fromkeys(args.arms))
    if args.controls:
        arms = list(dict.fromkeys([*arms, "E", "I", *CONTROL_ARMS]))
        LOGGER.info("control ladder requested; arms expanded to %s", arms)
    args.arms = arms

    manifest = pipeline.get_manifest()
    if not pipeline.image_cache_dir.exists():
        LOGGER.info("exporting image cache first")
        pipeline.export_image_cache(manifest)

    assignment = pipeline.make_holdout_split(manifest, write=True)
    configs = build_configs(
        loader, budget, args.arms, args.epochs,
        respect_own_schedule=not args.no_budget_override,
    )

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

    report_paired(aggregator, results, args.seeds[0], args.preregistered, runs_dir)

    if args.ensemble:
        print()
        print("=" * 78)
        print("SEED ENSEMBLES")
        print("=" * 78)
        ensembler = SeedEnsemble(space="logit")
        tables = []
        for model_name in sorted({r.model_name for r in results}):
            members = [r for r in results if r.model_name == model_name]
            if len(members) < 2:
                print()
                print(f"{model_name}: only {len(members)} seed, skipping")
                continue
            table = ensembler.compare_with_members(members)
            table.insert(0, "arm", model_name)
            tables.append(table)
            print()
            print(
                table[
                    ["run", "seed", "macro_f1", "macro_auroc", "macro_auprc"]
                ].to_string(index=False)
            )
        if tables:
            combined = pd.concat(tables, ignore_index=True)
            combined.to_csv(runs_dir / "ensemble_comparison.csv", index=False)

    print()
    print(f"Artefacts: {runs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
