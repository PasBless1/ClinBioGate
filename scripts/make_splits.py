"""Create and verify patient-grouped splits.

Usage:
    python scripts/make_splits.py --config configs/data.yaml
    python scripts/make_splits.py --config configs/data.yaml --folds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from olives_biomarkers.data.splits import SplitManifestWriter, SplitValidator  # noqa: E402
from olives_biomarkers.pipeline import OlivesPipeline  # noqa: E402
from olives_biomarkers.utils.logging import LoggerFactory  # noqa: E402

LOGGER = LoggerFactory.get("olives.make_splits")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create patient-grouped splits.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--folds", action="store_true", help="Also write k-fold assignments.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = OlivesPipeline.from_config(args.config, repo_root=REPO_ROOT)
    manifest = pipeline.get_manifest()
    modelling = manifest.modelling_frame(
        policy=pipeline.config.duplicates.policy, labelled_only=True
    )
    LOGGER.info(
        "modelling frame: %d rows, %d patients", len(modelling), modelling["patient_id"].nunique()
    )

    assignments = [pipeline.make_holdout_split(manifest, write=True)]
    if args.folds:
        assignments.extend(pipeline.make_folds(manifest, write=True))

    validator = SplitValidator(pipeline.config.data.group_key)
    for assignment in assignments:
        validator.validate(modelling, assignment)
    LOGGER.info("all %d splits verified leakage-free", len(assignments))

    print()
    print(
        pd.DataFrame(
            [
                {"split": a.name, **{k: len(v) for k, v in a.partitions.items()}}
                for a in assignments
            ]
        ).to_string(index=False)
    )

    writer = SplitManifestWriter(pipeline.split_dir, pipeline.config.data.group_key)
    print()
    print("Label prevalence by partition (holdout):")
    print(
        writer.prevalence_by_partition(
            modelling, assignments[0], manifest.label_columns
        ).to_string(index=False)
    )
    print()
    print(f"Split manifests written to {pipeline.split_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
