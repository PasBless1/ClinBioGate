"""Name the biomarkers where clinical context is predicted to help, in advance.

Run this **before** training any fusion arm and commit its output. It measures
the within-eye association between clinical change and biomarker change using
the fold's training patients only, and writes the qualifying labels to a JSON
file. Those labels are the confirmatory test; everything else in the per-label
table is exploratory and has to be reported as such.

The reason for the ceremony: with thirteen labels and a 95% interval, roughly
one apparent win is expected by chance. Choosing which labels to highlight after
seeing the test results converts that into a finding, and it is the first thing
a reviewer looks for.

Usage:
    python scripts/preregister_targets.py --config configs/fusion_longitudinal.yaml
    python scripts/preregister_targets.py --folds            # all five CV folds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from olives_biomarkers.data import WithinEyeAssociation  # noqa: E402
from olives_biomarkers.data.splits import GroupedCrossValidator, PatientGroupedSplitter  # noqa: E402
from olives_biomarkers.pipeline import OlivesPipeline  # noqa: E402
from olives_biomarkers.utils.logging import LoggerFactory  # noqa: E402

LOGGER = LoggerFactory.get("olives.preregister")

# Per-label OCT-only AUPRC from the A100 holdout run. A label only qualifies if
# the image model leaves room; a strong association on a label already predicted
# at 0.97 is not an opportunity.
OCT_BASELINE = {
    "irf": 0.9731, "drt_me": 0.9676, "irhrf": 0.9545, "srf": 0.8169, "favf": 0.8059,
    "vitreous_debris": 0.7337, "pavf": 0.6231, "ez_disruption": 0.4897, "shrm": 0.2530,
    "preretinal_tissue": 0.2259, "ir_hemorrhages": 0.1552, "rpe_disruption": 0.0822,
    "atrophy_thinning": 0.0406,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fusion_longitudinal.yaml")
    parser.add_argument("--folds", action="store_true", help="analyse all CV folds, not the holdout")
    parser.add_argument("--feature", default="cst", help="clinical feature to test the association of")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--min-eyes", type=int, default=10)
    parser.add_argument(
        "--headroom-below",
        type=float,
        default=0.70,
        help="exclude labels whose OCT AUPRC is at or above this",
    )
    parser.add_argument("--no-headroom-filter", action="store_true")
    parser.add_argument("--output", default="outputs/reports")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = OlivesPipeline.from_config(REPO_ROOT / args.config, repo_root=REPO_ROOT)
    config = pipeline.config

    manifest = pipeline.get_manifest()
    frame = pipeline.modelling_frame(manifest, attach_cache=False)
    labels = manifest.label_columns
    analyser = WithinEyeAssociation(labels)

    if args.folds:
        assignments = list(
            GroupedCrossValidator(n_folds=config.split.n_folds, seed=config.split.seed).split(frame)
        )
        names = [f"fold_{i}" for i in range(len(assignments))]
    else:
        assignments = [PatientGroupedSplitter(seed=config.split.seed).split(frame)]
        names = ["holdout"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = None if args.no_headroom_filter else OCT_BASELINE
    summary: list[dict[str, object]] = []

    for name, assignment in zip(names, assignments):
        LOGGER.info("=" * 78)
        LOGGER.info("split '%s': %d training patients", name, len(assignment.train))
        table = analyser.for_training_fold(frame, assignment)

        display = table[["label", "eyes_with_change", "visits",
                         f"r_{args.feature}", f"p_{args.feature}"]]
        LOGGER.info("within-eye association (training patients only)\n%s",
                    display.round(4).to_string(index=False))

        registration = analyser.preregister(
            table,
            oct_baseline=baseline,
            min_eyes=args.min_eyes,
            alpha=args.alpha,
            headroom_below=args.headroom_below,
            feature=args.feature,
            fit_patients=list(assignment.train),
        )
        path = registration.save(output_dir / f"preregistered_targets_{name}.json")
        table.to_csv(output_dir / f"within_eye_association_{name}.csv", index=False)
        LOGGER.info("targets for '%s': %s -> %s", name, registration.labels or "none", path)
        summary.append({"split": name, "n_targets": len(registration.labels),
                        "targets": ", ".join(registration.labels)})

    LOGGER.info("=" * 78)
    LOGGER.info("summary\n%s", pd.DataFrame(summary).to_string(index=False))
    if args.folds:
        stable = set.intersection(
            *[set(row["targets"].split(", ")) - {""} for row in summary]
        ) if summary else set()
        LOGGER.info(
            "labels qualifying in EVERY fold: %s\n"
            "Those are the defensible confirmatory set; a label that only qualifies in one "
            "fold is a fold-specific artefact.",
            sorted(stable) or "none",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
