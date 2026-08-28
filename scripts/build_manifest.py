"""Build the metadata-only sample manifest from the OLIVES parquet shards.

Usage:
    python scripts/build_manifest.py --config configs/data.yaml
    python scripts/build_manifest.py --config configs/data.yaml --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from olives_biomarkers.pipeline import OlivesPipeline  # noqa: E402
from olives_biomarkers.utils.logging import LoggerFactory  # noqa: E402

LOGGER = LoggerFactory.get("olives.build_manifest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the OLIVES sample manifest.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--split", default="train", help="Parquet split, or 'all'.")
    parser.add_argument("--no-hashes", action="store_true", help="Skip image hashing (faster, but disables duplicate detection).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = OlivesPipeline.from_config(args.config, repo_root=REPO_ROOT)

    if not pipeline.data_root.exists():
        LOGGER.error("OLIVES data not found at %s", pipeline.data_root)
        return 2

    if args.no_hashes:
        pipeline.config.manifest.compute_image_hashes = False
        LOGGER.warning("image hashing disabled; duplicate detection will be unavailable")

    split = None if args.split == "all" else args.split
    manifest = pipeline.build_manifest(split=split, save=True)

    print()
    for key, value in manifest.summary().items():
        print(f"{key:>26}: {value}")
    print()
    print(f"Manifest written to {pipeline.manifest_path(split)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
