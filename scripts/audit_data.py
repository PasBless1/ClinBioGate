"""Phase 0 entry point: build the manifest and run the data audit.

Usage:
    python scripts/audit_data.py --config configs/data.yaml
    python scripts/audit_data.py --config configs/data.yaml --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from olives_biomarkers.pipeline import OlivesPipeline  # noqa: E402
from olives_biomarkers.utils.logging import LoggerFactory  # noqa: E402

LOGGER = LoggerFactory.get("olives.audit_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the local OLIVES dataset.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to the data config.")
    parser.add_argument("--split", default="train", help="Parquet split to audit.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the manifest from parquet.")
    parser.add_argument(
        "--fail-on-blocker",
        action="store_true",
        help="Exit non-zero when the audit reports a blocker.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = OlivesPipeline.from_config(args.config, repo_root=REPO_ROOT)

    print(pipeline.describe())
    print()

    if not pipeline.data_root.exists():
        LOGGER.error(
            "OLIVES data not found at %s\n"
            "Download the Hugging Face dataset 'gOLIVES/OLIVES_Dataset' to that path, "
            "or edit data.root in %s.",
            pipeline.data_root,
            args.config,
        )
        return 2

    manifest = pipeline.get_manifest(split=args.split, rebuild=args.rebuild)
    report = pipeline.run_audit(manifest, write=True)

    print()
    print("=" * 72)
    print(f"AUDIT: {len(report.findings)} findings ({len(report.blockers)} blockers)")
    print("=" * 72)
    for finding in report.findings:
        print(f"  {finding}")
    print()
    print(f"Reports written to {pipeline.output_dir / 'reports'}")

    if args.fail_on_blocker and report.blockers:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
