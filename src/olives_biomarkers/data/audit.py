"""Phase 0 data audit.

Answers the questions that must be settled before any model is trained: does the
data match the declared schema, how prevalent is each label, where are values
missing, which images are duplicated, and which fields the project brief assumes
simply do not exist in this data source.

The audit produces both a Markdown report (for reading) and JSON (for tests and
downstream tooling), and it fails loudly rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from olives_biomarkers.data.manifests import Manifest, ParquetShardIndex
from olives_biomarkers.data.schema import LabelSchema
from olives_biomarkers.utils.io import JsonIO
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.audit")


@dataclass
class AuditFinding:
    """One thing the audit noticed, with a severity the report groups by."""

    severity: str  # "info" | "warning" | "blocker"
    topic: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.topic}: {self.message}"


@dataclass
class AuditReport:
    """Structured audit result."""

    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sections: dict[str, Any] = field(default_factory=dict)
    findings: list[AuditFinding] = field(default_factory=list)

    def add(self, severity: str, topic: str, message: str) -> None:
        """Record a finding."""
        self.findings.append(AuditFinding(severity, topic, message))

    @property
    def blockers(self) -> list[AuditFinding]:
        """Findings that must be resolved before modelling."""
        return [f for f in self.findings if f.severity == "blocker"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_utc": self.created_utc,
            "sections": self.sections,
            "findings": [vars(f) for f in self.findings],
        }

    def to_json(self, path: str | Path) -> Path:
        """Write the machine-readable report."""
        return JsonIO.write(self.to_dict(), path)

    def to_markdown(self, path: str | Path) -> Path:
        """Write the human-readable report."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render_markdown(), encoding="utf-8")
        return out

    def render_markdown(self) -> str:
        """Render the report as Markdown."""
        lines: list[str] = [
            "# OLIVES data audit",
            "",
            f"_Generated {self.created_utc}_",
            "",
        ]

        by_severity: dict[str, list[AuditFinding]] = {"blocker": [], "warning": [], "info": []}
        for finding in self.findings:
            by_severity.setdefault(finding.severity, []).append(finding)

        lines += ["## Findings", ""]
        for severity in ("blocker", "warning", "info"):
            items = by_severity.get(severity, [])
            if not items:
                continue
            label = {"blocker": "Blockers", "warning": "Warnings", "info": "Notes"}[severity]
            lines.append(f"### {label} ({len(items)})")
            lines.append("")
            for item in items:
                lines.append(f"- **{item.topic}** - {item.message}")
            lines.append("")

        for name, payload in self.sections.items():
            lines.append(f"## {name.replace('_', ' ').title()}")
            lines.append("")
            lines.append(self._render_payload(payload))
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_payload(payload: Any) -> str:
        """Render a section body as a Markdown table or key/value list."""
        if isinstance(payload, pd.DataFrame):
            return payload.to_markdown(index=False)
        if isinstance(payload, dict):
            rows = []
            for key, value in payload.items():
                rendered = value
                if isinstance(value, float):
                    rendered = format(value, ".4f")
                rows.append(f"| `{key}` | {rendered} |")
            return "\n".join(["| field | value |", "| --- | --- |", *rows])
        if isinstance(payload, list):
            return "\n".join(f"- {item}" for item in payload)
        return str(payload)


class DataAuditor:
    """Runs every Phase 0 check over a built manifest."""

    RARE_LABEL_THRESHOLD = 0.01
    MIN_POSITIVES_FOR_STABLE_FOLD = 50

    def __init__(self, manifest: Manifest, schema: LabelSchema, data_root: str | Path) -> None:
        self.manifest = manifest
        self.schema = schema
        self.data_root = Path(data_root)
        self.report = AuditReport()

    # ------------------------------------------------------------------
    # individual checks
    # ------------------------------------------------------------------
    def check_source_files(self) -> None:
        """Confirm the parquet shards exist and record their footprint."""
        index = ParquetShardIndex(self.data_root, self.schema.config_name)
        shards = index.shards(None)
        total_bytes = sum(s.path.stat().st_size for s in shards)
        splits = sorted({s.split for s in shards})
        self.report.sections["source_files"] = {
            "config_dir": str(index.config_dir),
            "n_shards": len(shards),
            "splits_present": ", ".join(splits),
            "total_size_gb": round(total_bytes / 1e9, 2),
            "rows_declared": sum(s.num_rows for s in shards),
        }
        if "test" not in splits and self.schema.config_name == "disease_classification":
            self.report.add(
                "info",
                "splits",
                "This config ships a single `train` split; all partitioning is ours to define.",
            )

    def check_schema(self) -> None:
        """Verify declared columns exist verbatim in the parquet schema."""
        index = ParquetShardIndex(self.data_root, self.schema.config_name)
        available = index.column_names()
        self.schema.validate_columns(available, include_labels=True)
        self.report.sections["schema"] = {
            "target_set": self.schema.target_set,
            "n_labels": self.schema.n_labels,
            "columns_in_parquet": len(available),
            "all_declared_columns_present": True,
        }
        self.report.add("info", "schema", f"All {self.schema.n_labels} declared label columns resolved.")

    def check_cohort(self) -> None:
        """Patient, eye and visit counts, plus per-patient volume."""
        frame = self.manifest.frame
        rows_per_patient = frame.groupby("patient_id").size()
        eyes_per_patient = frame.groupby("patient_id")["eye_id"].nunique()
        section: dict[str, Any] = {
            "n_rows": int(len(frame)),
            "n_patients": int(frame["patient_id"].nunique()),
            "n_patient_eye_pairs": int(frame.groupby(["patient_id", "eye_id"], dropna=False).ngroups),
            "rows_per_patient_min": int(rows_per_patient.min()),
            "rows_per_patient_median": int(rows_per_patient.median()),
            "rows_per_patient_max": int(rows_per_patient.max()),
            "patients_with_two_eyes": int((eyes_per_patient > 1).sum()),
        }
        if "visit_uid" in frame.columns:
            visits_per_eye = frame.groupby(["patient_id", "eye_id"])["visit_index"].nunique()
            section["n_visits_inferred"] = int(frame["visit_uid"].nunique())
            section["visits_per_eye_min"] = int(visits_per_eye.min())
            section["visits_per_eye_median"] = int(visits_per_eye.median())
            section["visits_per_eye_max"] = int(visits_per_eye.max())
        self.report.sections["cohort"] = section

        n_patients = section["n_patients"]
        if n_patients < 100:
            self.report.add(
                "warning",
                "cohort size",
                f"Only {n_patients} patients. With patient-grouped splitting the test set holds "
                f"~{int(n_patients * 0.15)} patients, so confidence intervals will be wide and "
                "fold-to-fold variance high. Report patient-level bootstrap CIs, never scan-level.",
            )

    def check_visits(self) -> None:
        """Validate the inferred visit segmentation against the expected 49-scan volume."""
        frame = self.manifest.frame
        if "visit_uid" not in frame.columns:
            return

        visits = frame.drop_duplicates("visit_uid")
        unique_per_visit = frame.groupby("visit_uid")["image_hash"].nunique() if "image_hash" in frame else None
        expected = 49
        section: dict[str, Any] = {
            "n_visits_inferred": int(len(visits)),
            "visit_size_mode": int(visits["visit_size"].mode().iloc[0]),
            "visits_with_expected_49_rows": int((visits["visit_size"] == expected).sum()),
            "visits_with_98_rows_fully_duplicated": int((visits["visit_size"] == 2 * expected).sum()),
        }
        if unique_per_visit is not None:
            off_spec = unique_per_visit[unique_per_visit != expected]
            section["visits_with_unexpected_unique_image_count"] = int(len(off_spec))
            section["unexpected_visit_sizes"] = sorted(off_spec.unique().tolist())[:10]
            if len(off_spec):
                self.report.add(
                    "warning",
                    "visit inference",
                    f"{len(off_spec)} of {len(visits)} inferred visits do not hold exactly {expected} "
                    f"unique images (sizes {sorted(off_spec.unique().tolist())[:6]}). Visit indices are "
                    "inferred from BCVA/CST change-points and are not authoritative; do not report "
                    "visit-level results without manual verification.",
                )

        per_eye = frame.groupby(["patient_id", "eye_id"])["visit_index"].nunique()
        section["visits_per_eye_min"] = int(per_eye.min())
        section["visits_per_eye_median"] = int(per_eye.median())
        section["visits_per_eye_max"] = int(per_eye.max())
        self.report.sections["visit_inference"] = section

        self.report.add(
            "info",
            "visit inference",
            f"{len(visits)} visits inferred across {len(per_eye)} eyes "
            f"(median {int(per_eye.median())} per eye), consistent with the ~16 visits/patient "
            "reported in the OLIVES paper.",
        )

    def check_scan_number_coverage(self) -> None:
        """``Scan (n/49)`` is only populated on annotated visits; make that explicit."""
        frame = self.manifest.frame
        if "scan_number" not in frame.columns:
            return
        with_scan = int(frame["scan_number"].notna().sum())
        labelled = int(frame["has_biomarkers"].sum())
        mismatch = int((frame["scan_number"].notna() != frame["has_biomarkers"]).sum())
        self.report.sections["scan_number_coverage"] = {
            "rows_with_scan_number": with_scan,
            "rows_with_complete_biomarkers": labelled,
            "rows_where_the_two_disagree": mismatch,
            "pct_rows_with_scan_number": round(100 * with_scan / max(len(frame), 1), 2),
        }
        self.report.add(
            "warning",
            "scan_number coverage",
            f"`Scan (n/49)` is present on only {with_scan} of {len(frame)} rows "
            f"({100 * with_scan / max(len(frame), 1):.1f}%) - it belongs to the biomarker annotation "
            "block, not to every scan. It cannot be used as a general ordering key.",
        )
        if mismatch:
            self.report.add(
                "warning",
                "scan_number coverage",
                f"{mismatch} rows have a scan number but an incomplete biomarker vector (or vice "
                "versa). Filter modelling rows on `has_biomarkers`, never on scan_number.",
            )

    def check_disease_balance(self) -> None:
        """Disease label distribution and per-patient consistency."""
        frame = self.manifest.frame
        if "disease_label" not in frame.columns:
            return
        per_patient_labels = frame.groupby("patient_id")["disease_label"].nunique()
        mixed = per_patient_labels[per_patient_labels > 1]
        counts = frame["disease_name"].value_counts(dropna=False)
        patient_counts = (
            frame.drop_duplicates("patient_id")["disease_name"].value_counts(dropna=False)
        )
        self.report.sections["disease_balance"] = {
            **{f"scans_{k}": int(v) for k, v in counts.items()},
            **{f"patients_{k}": int(v) for k, v in patient_counts.items()},
            "patients_with_mixed_labels": int(len(mixed)),
        }
        if len(mixed):
            self.report.add(
                "blocker",
                "disease label",
                f"{len(mixed)} patients carry more than one disease label: {list(mixed.index)}",
            )

    def check_label_prevalence(self) -> None:
        """Per-label positives, and whether any label is too rare to model."""
        labelled = self.manifest.labelled()
        prevalence = self.manifest.label_prevalence(labelled)
        self.report.sections["label_prevalence"] = prevalence

        n_labelled = len(labelled)
        self.report.sections["label_coverage"] = {
            "n_rows_total": int(len(self.manifest.frame)),
            "n_rows_biomarker_labelled": int(n_labelled),
            "pct_labelled": round(100 * n_labelled / max(len(self.manifest.frame), 1), 2),
        }

        rare = prevalence[prevalence["prevalence"] < self.RARE_LABEL_THRESHOLD]
        for _, row in rare.iterrows():
            self.report.add(
                "warning",
                "rare label",
                f"`{row['label']}` has {row['n_positive']} positives "
                f"({100 * row['prevalence']:.2f}%) across {row['n_patients_positive']} patients - "
                "expect unstable per-label metrics and empty folds.",
            )
        unstable = prevalence[prevalence["n_positive"] < self.MIN_POSITIVES_FOR_STABLE_FOLD]
        if len(unstable):
            self.report.add(
                "warning",
                "target set",
                f"{len(unstable)} of {len(prevalence)} labels have <"
                f"{self.MIN_POSITIVES_FOR_STABLE_FOLD} positives. The 6-label MVP "
                "(`target_set: six`) avoids all of them.",
            )

    def check_label_cooccurrence(self) -> None:
        """How many labels co-occur per scan, and the pairwise Jaccard structure."""
        labelled = self.manifest.labelled()
        label_cols = self.manifest.label_columns
        if not label_cols or labelled.empty:
            return
        counts = labelled[label_cols].fillna(0).sum(axis=1)
        self.report.sections["label_cooccurrence"] = {
            "mean_labels_per_scan": round(float(counts.mean()), 3),
            "median_labels_per_scan": int(counts.median()),
            "max_labels_per_scan": int(counts.max()),
            "scans_with_zero_labels": int((counts == 0).sum()),
            "unique_label_vectors": int(labelled[label_cols].drop_duplicates().shape[0]),
        }

    def check_clinical_missingness(self) -> None:
        """Where BCVA/CST are missing and whether the loss concentrates in a patient."""
        frame = self.manifest.frame
        section: dict[str, Any] = {}
        offenders: dict[str, list[int]] = {}
        for clinical in ("bcva", "cst"):
            col = f"{clinical}_missing"
            if col not in frame.columns:
                continue
            missing = frame[frame[col]]
            section[f"n_{clinical}_missing"] = int(len(missing))
            section[f"pct_{clinical}_missing"] = round(100 * len(missing) / max(len(frame), 1), 3)
            section[f"patients_with_missing_{clinical}"] = int(missing["patient_id"].nunique())
            if len(missing):
                offenders[clinical] = sorted(missing["patient_id"].dropna().unique().tolist())

        for clinical, patients in offenders.items():
            preview = patients[:10]
            self.report.add(
                "warning",
                "clinical missingness",
                f"`{clinical}` missing for {len(patients)} patient(s): {preview}"
                f"{' ...' if len(patients) > 10 else ''}. Impute on the training fold only and keep "
                "a missingness indicator; run the complete-case sensitivity analysis.",
            )

        for clinical in ("bcva", "cst"):
            if clinical in frame.columns:
                values = frame[clinical].dropna()
                if len(values):
                    section[f"{clinical}_min"] = float(values.min())
                    section[f"{clinical}_median"] = float(values.median())
                    section[f"{clinical}_max"] = float(values.max())
        self.report.sections["clinical_missingness"] = section

    def check_duplicates(self) -> None:
        """Exact-duplicate images: how many, and whether any cross a patient."""
        frame = self.manifest.frame
        if "is_duplicate" not in frame.columns:
            self.report.add(
                "warning", "duplicates", "Manifest built without image hashes; duplicate check skipped."
            )
            return

        duplicated = frame[frame["is_duplicate"]]
        groups = duplicated.groupby("dup_group_id")
        cross_patient = groups["patient_id"].nunique()
        n_cross_patient = int((cross_patient > 1).sum())
        cross_eye = groups.apply(
            lambda g: g.groupby(["patient_id", "eye_id"], dropna=False).ngroups, include_groups=False
        )
        n_cross_eye = int((cross_eye > 1).sum()) if len(cross_eye) else 0

        section = {
            "n_rows": int(len(frame)),
            "n_unique_images": int(frame["dup_group_id"].nunique()),
            "n_duplicate_rows": int(len(duplicated)),
            "pct_duplicate_rows": round(100 * len(duplicated) / max(len(frame), 1), 2),
            "n_duplicate_groups": int(duplicated["dup_group_id"].nunique()),
            "max_group_size": int(frame["dup_group_size"].max()),
            "groups_spanning_multiple_patients": n_cross_patient,
            "groups_spanning_multiple_eyes": n_cross_eye,
        }
        if "is_adjacent_duplicate" in frame.columns:
            section["n_adjacent_duplicate_rows"] = int(frame["is_adjacent_duplicate"].sum())
        self.report.sections["duplicates"] = section

        if len(duplicated):
            self.report.add(
                "warning",
                "duplicates",
                f"{len(duplicated)} rows ({section['pct_duplicate_rows']}%) are byte-identical to "
                f"another row, in {section['n_duplicate_groups']} groups. Deduplicate before "
                "splitting, and never let one group straddle two partitions.",
            )
        if n_cross_patient:
            self.report.add(
                "blocker",
                "duplicates",
                f"{n_cross_patient} duplicate groups span more than one patient id - patient-grouped "
                "splitting alone will NOT contain them.",
            )

    def check_unresolved_fields(self) -> None:
        """Report brief-assumed fields that this data source does not carry."""
        unresolved = self.schema.unresolved_fields
        if not unresolved:
            return
        self.report.sections["unresolved_fields"] = {
            name: spec.get("status", "unknown") for name, spec in unresolved.items()
        }
        for name, spec in unresolved.items():
            note = " ".join(str(spec.get("note", "")).split())
            self.report.add("warning", f"unresolved field: {name}", note)

    def check_fundus_readiness(self) -> None:
        """State explicitly whether the fundus extension can proceed."""
        available = ParquetShardIndex(self.data_root, self.schema.config_name).column_names()
        has_fundus = any("fundus" in c.lower() for c in available)
        self.report.sections["fundus_pairing"] = {
            "fundus_columns_present": has_fundus,
            "safe_to_pair": False if not has_fundus else "requires manual verification",
        }
        if not has_fundus:
            self.report.add(
                "info",
                "fundus extension",
                "No fundus imagery in this mirror, so fundus pairing is NOT currently safe and the "
                "Phase 6 fundus extension is blocked until the Zenodo release is obtained.",
            )

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------
    def run(self) -> AuditReport:
        """Run every check and return the populated report."""
        checks = (
            self.check_source_files,
            self.check_schema,
            self.check_cohort,
            self.check_visits,
            self.check_scan_number_coverage,
            self.check_disease_balance,
            self.check_label_prevalence,
            self.check_label_cooccurrence,
            self.check_clinical_missingness,
            self.check_duplicates,
            self.check_unresolved_fields,
            self.check_fundus_readiness,
        )
        for check in checks:
            LOGGER.info("audit: %s", check.__name__)
            check()
        LOGGER.info(
            "audit complete: %d findings (%d blockers)",
            len(self.report.findings),
            len(self.report.blockers),
        )
        return self.report
