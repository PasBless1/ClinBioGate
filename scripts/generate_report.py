"""Collect finished runs into one comparison report.

Usage:
    python scripts/generate_report.py --experiment-dir outputs/runs
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from olives_biomarkers.utils.io import JsonIO  # noqa: E402
from olives_biomarkers.utils.logging import LoggerFactory  # noqa: E402

LOGGER = LoggerFactory.get("olives.report")

#: The comparison the project brief requires. Rows without a run are reported as
#: missing rather than silently omitted.
MANDATORY_COMPARISONS = {
    "A": ("baseline_clinical", "Clinical only - clinical-information baseline"),
    "B": ("baseline_oct", "OCT only - imaging baseline"),
    "C": ("fusion_concat", "OCT + clinical concatenation - simple multimodal baseline"),
    "D": ("fusion_gated", "OCT + clinically gated fusion - proposed method"),
}


class ReportBuilder:
    """Assembles per-run artefacts into a single Markdown report."""

    def __init__(self, experiment_dir: Path) -> None:
        self.experiment_dir = experiment_dir

    def discover_runs(self) -> list[Path]:
        """Directories that contain a completed run."""
        return sorted(
            path
            for path in self.experiment_dir.glob("*")
            if path.is_dir() and (path / "run_metadata.json").exists()
        )

    @staticmethod
    def _read(path: Path) -> pd.DataFrame | None:
        return pd.read_csv(path) if path.exists() else None

    def summarise_run(self, run_dir: Path) -> dict:
        """One row of the headline comparison table."""
        metadata = JsonIO.read(run_dir / "run_metadata.json")
        record: dict = {
            "run_id": run_dir.name,
            "experiment": metadata.get("experiment"),
            "seed": metadata.get("seed"),
            "git_commit": metadata.get("git_commit"),
            "manifest_hash": metadata.get("manifest_hash"),
            "target_set": metadata.get("config", {}).get("data", {}).get("target_set"),
        }

        bootstrap = self._read(run_dir / "bootstrap_ci.csv")
        if bootstrap is not None:
            for _, row in bootstrap.iterrows():
                record[row["metric"]] = (
                    f"{row['point_estimate']:.4f} "
                    f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                )

        calibration = self._read(run_dir / "calibration_comparison.csv")
        if calibration is not None:
            record["ece_before"] = round(float(calibration["ece_pre"].mean()), 4)
            record["ece_after"] = round(float(calibration["ece_post"].mean()), 4)

        per_label = self._read(run_dir / "test_per_label.csv")
        if per_label is not None:
            record["n_labels_evaluated"] = int((~per_label["degenerate"]).sum())
            record["n_labels_undefined"] = int(per_label["degenerate"].sum())
        return record

    def build(self) -> str:
        """Render the report."""
        runs = self.discover_runs()
        lines = [
            "# OLIVES biomarker detection - results",
            "",
            f"_Generated {datetime.now(timezone.utc).isoformat()}_",
            "",
            f"Runs found: **{len(runs)}** in `{self.experiment_dir}`",
            "",
        ]

        if not runs:
            lines += [
                "No completed runs yet. Train a model first:",
                "",
                "```bash",
                "python scripts/train.py --config configs/baseline_oct.yaml",
                "python scripts/evaluate.py --run-dir outputs/runs/<run_id>",
                "```",
            ]
            return "\n".join(lines)

        summary = pd.DataFrame([self.summarise_run(run) for run in runs])
        lines += ["## Headline comparison", "", summary.to_markdown(index=False), ""]

        lines += ["## Mandatory comparisons", ""]
        completed = set(summary["experiment"].dropna())
        rows = [
            {
                "id": key,
                "model": name,
                "purpose": purpose,
                "status": "done" if name in completed else "NOT RUN",
            }
            for key, (name, purpose) in MANDATORY_COMPARISONS.items()
        ]
        lines += [pd.DataFrame(rows).to_markdown(index=False), ""]

        missing = [r["model"] for r in rows if r["status"] == "NOT RUN"]
        if missing:
            lines += [
                f"> **Incomplete.** {len(missing)} of {len(MANDATORY_COMPARISONS)} required "
                f"comparisons have no run: {', '.join(missing)}. No superiority claim can be made "
                "until every arm is trained on the same folds and budget.",
                "",
            ]

        for run in runs:
            lines += [f"## {run.name}", ""]
            per_label = self._read(run / "test_per_label.csv")
            if per_label is not None:
                display = per_label[
                    ["label", "n_positive", "threshold", "auroc", "auprc", "f1", "degenerate"]
                ].round(4)
                lines += ["### Per-label test metrics", "", display.to_markdown(index=False), ""]
                if per_label["degenerate"].any():
                    undefined = per_label.loc[per_label["degenerate"], "label"].tolist()
                    lines += [
                        f"> Metrics are undefined for {', '.join(undefined)}: no positive examples "
                        "in the test partition. Reported as undefined, not as zero.",
                        "",
                    ]
            coverage = self._read(run / "coverage_curve.csv")
            if coverage is not None:
                lines += [
                    "### Selective prediction",
                    "",
                    coverage.round(4).to_markdown(index=False),
                    "",
                ]

        lines += [
            "## Reading these results",
            "",
            "- Every interval is a **patient-level** bootstrap; scan-level resampling would be "
            "several times too narrow.",
            "- Compare arms only within the same split, seed and target set.",
            "- A difference whose intervals overlap heavily is not a finding.",
            "- With 87 patients (~13 in test), only large effects are detectable. A negative "
            "result is a legitimate outcome.",
            "",
        ]
        return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the results report.")
    parser.add_argument("--experiment-dir", default="outputs/runs")
    parser.add_argument("--output", default="outputs/reports/results.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.is_absolute():
        experiment_dir = REPO_ROOT / experiment_dir

    report = ReportBuilder(experiment_dir).build()
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print(report)
    print()
    print(f"Report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
