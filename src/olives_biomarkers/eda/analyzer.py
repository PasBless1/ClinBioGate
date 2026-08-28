"""Exploratory data analysis over the OLIVES manifest.

Every method returns a DataFrame or dict rather than printing, so the notebook
controls presentation and the same numbers can be reused in the written report.

The analyses are grouped to answer the questions that decide the experimental
design: how much independent evidence exists (cohort), what is being predicted
(labels), what the auxiliary modality actually contains (clinical), how the
repeated-measures structure constrains splitting (longitudinal, leakage), and
whether the images themselves are clean (image).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from olives_biomarkers.data.manifests import Manifest
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.eda")


class OlivesEDA:
    """Descriptive analyses of a built manifest.

    Args:
        manifest: A manifest produced by :class:`ManifestBuilder`.
        dedup_policy: Duplicate policy applied to the "analysis" view.

    Example:
        >>> eda = OlivesEDA(manifest)
        >>> eda.cohort_overview()
        >>> eda.label_prevalence()
    """

    def __init__(self, manifest: Manifest, dedup_policy: str = "keep_first") -> None:
        self.manifest = manifest
        self.schema = manifest.schema
        self.dedup_policy = dedup_policy
        self.frame = manifest.frame
        self.label_columns = manifest.label_columns

    # ==================================================================
    # views
    # ==================================================================
    @property
    def unique(self) -> pd.DataFrame:
        """All rows with exact duplicates removed."""
        return self.manifest.deduplicated(self.dedup_policy)

    @property
    def labelled(self) -> pd.DataFrame:
        """Deduplicated rows carrying a complete biomarker vector."""
        return self.manifest.modelling_frame(policy=self.dedup_policy, labelled_only=True)

    def display_names(self) -> dict[str, str]:
        """Canonical key to human-readable label name."""
        return self.schema.key_to_display

    # ==================================================================
    # 1. cohort structure
    # ==================================================================
    def cohort_overview(self) -> pd.DataFrame:
        """Headline counts across the raw, deduplicated and modelling views."""
        views = {
            "all rows": self.frame,
            f"unique images ({self.dedup_policy})": self.unique,
            "biomarker-labelled": self.labelled,
        }
        rows = []
        for name, frame in views.items():
            record = {
                "view": name,
                "n_scans": len(frame),
                "n_patients": frame["patient_id"].nunique(),
                "n_eyes": frame.groupby(["patient_id", "eye_id"], dropna=False).ngroups,
            }
            if "visit_uid" in frame.columns:
                record["n_visits"] = frame["visit_uid"].nunique()
            rows.append(record)
        return pd.DataFrame(rows)

    def per_patient_summary(self) -> pd.DataFrame:
        """One row per patient: eyes, visits, scans, disease, clinical range."""
        frame = self.unique
        grouped = frame.groupby("patient_id")
        summary = pd.DataFrame(
            {
                "n_scans": grouped.size(),
                "n_eyes": grouped["eye_id"].nunique(),
                "n_visits": grouped["visit_uid"].nunique()
                if "visit_uid" in frame.columns
                else np.nan,
                "disease": grouped["disease_name"].first()
                if "disease_name" in frame.columns
                else None,
                "n_labelled_scans": grouped["has_biomarkers"].sum(),
                "bcva_first": grouped["bcva"].first(),
                "bcva_last": grouped["bcva"].last(),
                "bcva_mean": grouped["bcva"].mean(),
                "cst_first": grouped["cst"].first(),
                "cst_last": grouped["cst"].last(),
                "cst_mean": grouped["cst"].mean(),
                "bcva_missing": grouped["bcva_missing"].sum()
                if "bcva_missing" in frame.columns
                else 0,
            }
        )
        summary["bcva_change"] = summary["bcva_last"] - summary["bcva_first"]
        summary["cst_change"] = summary["cst_last"] - summary["cst_first"]
        return summary.reset_index()

    def visits_per_eye(self) -> pd.DataFrame:
        """Distribution of inferred visit counts per eye."""
        frame = self.unique
        if "visit_uid" not in frame.columns:
            return pd.DataFrame()
        counts = (
            frame.groupby(["patient_id", "eye_id"])["visit_index"].nunique().rename("n_visits")
        )
        return counts.reset_index()

    def disease_distribution(self) -> pd.DataFrame:
        """DR versus DME at scan, eye and patient level."""
        frame = self.unique
        if "disease_name" not in frame.columns:
            return pd.DataFrame()
        by_scan = frame["disease_name"].value_counts().rename("n_scans")
        by_patient = (
            frame.drop_duplicates("patient_id")["disease_name"].value_counts().rename("n_patients")
        )
        by_eye = (
            frame.drop_duplicates(["patient_id", "eye_id"])["disease_name"]
            .value_counts()
            .rename("n_eyes")
        )
        out = pd.concat([by_scan, by_eye, by_patient], axis=1).fillna(0).astype(int)
        out["pct_patients"] = (100 * out["n_patients"] / out["n_patients"].sum()).round(1)
        return out.reset_index(names="disease")

    # ==================================================================
    # 2. labels
    # ==================================================================
    def label_prevalence(self) -> pd.DataFrame:
        """Positive counts, rates and patient spread per biomarker."""
        prevalence = self.manifest.label_prevalence(self.labelled)
        prevalence["pct"] = (100 * prevalence["prevalence"]).round(2)
        prevalence["rarity"] = pd.cut(
            prevalence["prevalence"],
            bins=[-0.001, 0.01, 0.05, 0.20, 1.0],
            labels=["very rare (<1%)", "rare (1-5%)", "uncommon (5-20%)", "common (>20%)"],
        )
        return prevalence

    def label_prevalence_by_disease(self) -> pd.DataFrame:
        """Prevalence of each biomarker split by DR versus DME.

        The paper notes that biomarkers overlap heavily across the two diseases,
        so wide separation here would be surprising.
        """
        frame = self.labelled
        if "disease_name" not in frame.columns:
            return pd.DataFrame()
        rows = []
        for label in self.label_columns:
            record: dict[str, Any] = {"label": label}
            for disease, group in frame.groupby("disease_name"):
                values = group[label].dropna()
                record[f"{disease}_prevalence"] = round(float((values == 1).mean()), 4) if len(values) else np.nan
                record[f"{disease}_n_positive"] = int((values == 1).sum())
            rows.append(record)
        out = pd.DataFrame(rows)
        if "DR_prevalence" in out.columns and "DME_prevalence" in out.columns:
            out["difference"] = (out["DME_prevalence"] - out["DR_prevalence"]).round(4)
        return out

    def label_cardinality(self) -> dict[str, Any]:
        """How many biomarkers co-occur on one scan."""
        frame = self.labelled
        counts = frame[self.label_columns].sum(axis=1)
        return {
            "mean_labels_per_scan": round(float(counts.mean()), 3),
            "median_labels_per_scan": float(counts.median()),
            "std_labels_per_scan": round(float(counts.std()), 3),
            "min_labels_per_scan": int(counts.min()),
            "max_labels_per_scan": int(counts.max()),
            "scans_with_no_label": int((counts == 0).sum()),
            "pct_scans_with_no_label": round(100 * float((counts == 0).mean()), 2),
            "n_unique_label_vectors": int(frame[self.label_columns].drop_duplicates().shape[0]),
            "n_possible_label_vectors": int(2 ** len(self.label_columns)),
            "label_density": round(float(counts.mean() / len(self.label_columns)), 4),
        }

    def label_cardinality_distribution(self) -> pd.DataFrame:
        """Histogram of the number of positive labels per scan."""
        counts = self.labelled[self.label_columns].sum(axis=1)
        out = counts.value_counts().sort_index().rename("n_scans").reset_index()
        out.columns = ["n_labels_present", "n_scans"]
        out["pct"] = (100 * out["n_scans"] / out["n_scans"].sum()).round(2)
        return out

    def label_cooccurrence_matrix(self, normalize: str = "jaccard") -> pd.DataFrame:
        """Pairwise co-occurrence between biomarkers.

        Args:
            normalize: ``"jaccard"``, ``"conditional"`` (P(col|row)), or ``"count"``.
        """
        matrix = self.labelled[self.label_columns].to_numpy(dtype=float)
        intersection = matrix.T @ matrix
        totals = matrix.sum(axis=0)

        if normalize == "count":
            result = intersection
        elif normalize == "conditional":
            with np.errstate(divide="ignore", invalid="ignore"):
                result = intersection / np.maximum(totals[:, None], 1)
        else:
            union = totals[:, None] + totals[None, :] - intersection
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.where(union > 0, intersection / np.maximum(union, 1), 0.0)
        return pd.DataFrame(result, index=self.label_columns, columns=self.label_columns)

    def label_correlation(self, method: str = "spearman") -> pd.DataFrame:
        """Correlation between biomarker indicator variables."""
        return self.labelled[self.label_columns].corr(method=method)

    def top_label_combinations(self, top_n: int = 15) -> pd.DataFrame:
        """The most frequent biomarker co-occurrence patterns."""
        frame = self.labelled[self.label_columns].astype(int)
        combos = frame.apply(
            lambda row: ", ".join([c for c in self.label_columns if row[c] == 1]) or "(none)",
            axis=1,
        )
        out = combos.value_counts().head(top_n).rename("n_scans").reset_index()
        out.columns = ["combination", "n_scans"]
        out["pct"] = (100 * out["n_scans"] / len(frame)).round(2)
        return out

    def label_prevalence_by_patient(self) -> pd.DataFrame:
        """Per-patient prevalence of each label.

        Reveals whether a rare biomarker lives in a handful of patients, which is
        what makes patient-grouped folds unstable for that label.
        """
        frame = self.labelled
        return frame.groupby("patient_id")[self.label_columns].mean().round(4)

    def label_patient_concentration(self) -> pd.DataFrame:
        """How concentrated each label's positives are within few patients."""
        frame = self.labelled
        rows = []
        for label in self.label_columns:
            positives = frame[frame[label] == 1]
            n_positive = len(positives)
            n_patients = positives["patient_id"].nunique()
            if n_positive:
                per_patient = positives.groupby("patient_id").size().sort_values(ascending=False)
                top_share = float(per_patient.iloc[0] / n_positive)
            else:
                top_share = np.nan
            rows.append(
                {
                    "label": label,
                    "n_positive": n_positive,
                    "n_patients_with_positive": n_patients,
                    "pct_patients": round(100 * n_patients / frame["patient_id"].nunique(), 1),
                    "largest_patient_share": round(top_share, 3) if n_positive else np.nan,
                    "at_risk_of_empty_fold": n_patients < 5,
                }
            )
        return pd.DataFrame(rows).sort_values("n_positive")

    # ==================================================================
    # 3. clinical variables
    # ==================================================================
    def clinical_summary(self) -> pd.DataFrame:
        """Descriptive statistics for BCVA and CST."""
        frame = self.unique
        rows = []
        for feature in ("bcva", "cst"):
            if feature not in frame.columns:
                continue
            values = frame[feature].dropna()
            rows.append(
                {
                    "feature": feature,
                    "n": int(len(values)),
                    "n_missing": int(frame[feature].isna().sum()),
                    "pct_missing": round(100 * float(frame[feature].isna().mean()), 2),
                    "mean": round(float(values.mean()), 2),
                    "std": round(float(values.std()), 2),
                    "min": float(values.min()),
                    "q25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                    "n_unique": int(values.nunique()),
                    "skew": round(float(values.skew()), 3),
                }
            )
        return pd.DataFrame(rows)

    def clinical_by_disease(self) -> pd.DataFrame:
        """BCVA and CST distributions split by disease."""
        frame = self.unique
        if "disease_name" not in frame.columns:
            return pd.DataFrame()
        return (
            frame.groupby("disease_name")[["bcva", "cst"]]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .round(2)
        )

    def missingness_profile(self) -> pd.DataFrame:
        """Where clinical values are missing, by patient."""
        frame = self.frame
        columns = [c for c in ("bcva_missing", "cst_missing") if c in frame.columns]
        if not columns:
            return pd.DataFrame()
        grouped = frame.groupby("patient_id")[columns].agg(["sum", "mean"])
        grouped.columns = ["_".join(c) for c in grouped.columns]
        affected = grouped[grouped.filter(like="_sum").sum(axis=1) > 0]
        return affected.round(4).reset_index()

    def clinical_label_association(self) -> pd.DataFrame:
        """Mean BCVA and CST for scans with versus without each biomarker.

        The paper reports that CST tracks fluid-related biomarkers (IRF, DRT/ME),
        which is the association most likely to make clinical fusion pay off.
        """
        frame = self.labelled
        rows = []
        for label in self.label_columns:
            present = frame[frame[label] == 1]
            absent = frame[frame[label] == 0]
            record: dict[str, Any] = {
                "label": label,
                "n_present": len(present),
                "n_absent": len(absent),
            }
            for feature in ("bcva", "cst"):
                if feature not in frame.columns:
                    continue
                mean_present = present[feature].mean()
                mean_absent = absent[feature].mean()
                record[f"{feature}_present"] = round(float(mean_present), 2)
                record[f"{feature}_absent"] = round(float(mean_absent), 2)
                record[f"{feature}_delta"] = round(float(mean_present - mean_absent), 2)
                pooled = frame[feature].std()
                record[f"{feature}_cohens_d"] = (
                    round(float((mean_present - mean_absent) / pooled), 3) if pooled else np.nan
                )
            rows.append(record)
        out = pd.DataFrame(rows)
        if "cst_cohens_d" in out.columns:
            out = out.reindex(out["cst_cohens_d"].abs().sort_values(ascending=False).index)
        return out.reset_index(drop=True)

    def clinical_correlation(self) -> pd.DataFrame:
        """Correlation between BCVA, CST and the biomarker indicators."""
        frame = self.labelled
        columns = [c for c in ("bcva", "cst") if c in frame.columns] + self.label_columns
        return frame[columns].corr(method="spearman").loc[["bcva", "cst"], self.label_columns].round(3)

    # ==================================================================
    # 4. longitudinal structure
    # ==================================================================
    def visit_trajectories(self) -> pd.DataFrame:
        """One row per inferred visit, with its clinical values and ordering."""
        frame = self.unique
        if "visit_uid" not in frame.columns:
            return pd.DataFrame()
        grouped = frame.groupby(["patient_id", "eye_id", "visit_index"], dropna=False)
        out = grouped.agg(
            n_scans=("row_uid", "size"),
            bcva=("bcva", "first"),
            cst=("cst", "first"),
            disease=("disease_name", "first"),
            has_biomarkers=("has_biomarkers", "max"),
        ).reset_index()
        out = out.sort_values(["patient_id", "eye_id", "visit_index"])
        out["bcva_change_from_first"] = out.groupby(["patient_id", "eye_id"])["bcva"].transform(
            lambda s: s - s.iloc[0]
        )
        out["cst_change_from_first"] = out.groupby(["patient_id", "eye_id"])["cst"].transform(
            lambda s: s - s.iloc[0]
        )
        out["bcva_change_from_previous"] = out.groupby(["patient_id", "eye_id"])["bcva"].diff()
        return out

    def treatment_progression(self) -> pd.DataFrame:
        """Mean BCVA change per visit index, mirroring Figure 5 of the paper.

        Also reports how many eyes remain at each visit, since the cohort thins
        out and later visits are dominated by the harder cases.
        """
        trajectories = self.visit_trajectories()
        if trajectories.empty:
            return pd.DataFrame()
        grouped = trajectories.groupby("visit_index")
        out = pd.DataFrame(
            {
                "n_eyes": grouped.size(),
                "mean_bcva": grouped["bcva"].mean().round(2),
                "mean_cst": grouped["cst"].mean().round(1),
                "mean_bcva_change_from_first": grouped["bcva_change_from_first"].mean().round(2),
                "std_bcva_change_from_first": grouped["bcva_change_from_first"].std().round(2),
                "mean_visit_to_visit_change": grouped["bcva_change_from_previous"].mean().round(3),
                "pct_improved_vs_previous": (
                    100 * grouped["bcva_change_from_previous"].apply(lambda s: (s > 0).mean())
                ).round(1),
            }
        ).reset_index()
        return out

    def within_eye_variability(self) -> pd.DataFrame:
        """How much BCVA and CST move within one eye across its visits.

        Low within-eye variability means repeated visits of the same eye are near
        duplicates, which is precisely why splitting must group by patient.
        """
        trajectories = self.visit_trajectories()
        if trajectories.empty:
            return pd.DataFrame()
        grouped = trajectories.groupby(["patient_id", "eye_id"])
        out = pd.DataFrame(
            {
                "n_visits": grouped.size(),
                "bcva_std": grouped["bcva"].std().round(2),
                "bcva_range": (grouped["bcva"].max() - grouped["bcva"].min()).round(1),
                "cst_std": grouped["cst"].std().round(2),
                "cst_range": (grouped["cst"].max() - grouped["cst"].min()).round(1),
            }
        ).reset_index()
        return out

    def first_last_visit_comparison(self) -> pd.DataFrame:
        """Biomarker prevalence at the first versus the last annotated visit.

        Biomarkers are annotated only at these two visits, and the paper shows
        treatment itself induces a domain shift between them.
        """
        frame = self.labelled
        if "visit_index" not in frame.columns:
            return pd.DataFrame()
        ranked = frame.copy()
        extremes = ranked.groupby(["patient_id", "eye_id"])["visit_index"].agg(["min", "max"])
        ranked = ranked.merge(extremes, on=["patient_id", "eye_id"], how="left")
        ranked["visit_position"] = np.select(
            [ranked["visit_index"] == ranked["min"], ranked["visit_index"] == ranked["max"]],
            ["first", "last"],
            default="middle",
        )
        rows = []
        for label in self.label_columns:
            record: dict[str, Any] = {"label": label}
            for position in ("first", "last"):
                subset = ranked[ranked["visit_position"] == position]
                record[f"{position}_prevalence"] = (
                    round(float(subset[label].mean()), 4) if len(subset) else np.nan
                )
                record[f"{position}_n"] = int(len(subset))
            record["change"] = round(
                record.get("last_prevalence", np.nan) - record.get("first_prevalence", np.nan), 4
            )
            rows.append(record)
        return pd.DataFrame(rows)

    # ==================================================================
    # 5. duplicates and leakage risk
    # ==================================================================
    def duplicate_analysis(self) -> dict[str, Any]:
        """Scale and containment of exact-duplicate images."""
        frame = self.frame
        if "is_duplicate" not in frame.columns:
            return {"available": False}
        duplicated = frame[frame["is_duplicate"]]
        groups = duplicated.groupby("dup_group_id")
        cross_patient = int((groups["patient_id"].nunique() > 1).sum())
        adjacent = int(frame["is_adjacent_duplicate"].sum()) if "is_adjacent_duplicate" in frame else 0
        return {
            "available": True,
            "n_rows": len(frame),
            "n_unique_images": int(frame["dup_group_id"].nunique()),
            "n_duplicate_rows": len(duplicated),
            "pct_duplicate_rows": round(100 * len(duplicated) / len(frame), 2),
            "n_duplicate_groups": int(duplicated["dup_group_id"].nunique()),
            "max_group_size": int(frame["dup_group_size"].max()),
            "n_adjacent_duplicate_rows": adjacent,
            "n_non_adjacent_duplicate_rows": len(duplicated) - adjacent,
            "groups_spanning_multiple_patients": cross_patient,
            "cross_patient_leakage_risk": cross_patient > 0,
        }

    def duplicate_visits(self) -> pd.DataFrame:
        """Visits whose images repeat an earlier visit of the same eye.

        The dataset card cites patient 61 (W8 identical to W12); this locates
        every such case.
        """
        frame = self.frame
        if "dup_group_id" not in frame.columns or "visit_uid" not in frame.columns:
            return pd.DataFrame()
        non_adjacent = frame[frame["is_duplicate"] & ~frame.get("is_adjacent_duplicate", False)]
        if non_adjacent.empty:
            return pd.DataFrame()
        spread = non_adjacent.groupby("dup_group_id")["visit_uid"].nunique()
        multi_visit = spread[spread > 1].index
        subset = non_adjacent[non_adjacent["dup_group_id"].isin(multi_visit)]
        out = (
            subset.groupby(["patient_id", "eye_id"])
            .agg(
                n_repeated_images=("dup_group_id", "nunique"),
                visits_involved=("visit_uid", lambda s: sorted(s.unique())),
            )
            .reset_index()
        )
        return out.sort_values("n_repeated_images", ascending=False)

    def leakage_risk_summary(self) -> pd.DataFrame:
        """Nested structure that makes random image-level splitting invalid."""
        frame = self.unique
        rows = [
            {
                "grouping": "patient",
                "n_groups": frame["patient_id"].nunique(),
                "mean_scans_per_group": round(len(frame) / frame["patient_id"].nunique(), 1),
            },
            {
                "grouping": "patient x eye",
                "n_groups": frame.groupby(["patient_id", "eye_id"], dropna=False).ngroups,
                "mean_scans_per_group": round(
                    len(frame) / frame.groupby(["patient_id", "eye_id"], dropna=False).ngroups, 1
                ),
            },
        ]
        if "visit_uid" in frame.columns:
            rows.append(
                {
                    "grouping": "patient x eye x visit",
                    "n_groups": frame["visit_uid"].nunique(),
                    "mean_scans_per_group": round(len(frame) / frame["visit_uid"].nunique(), 1),
                }
            )
        rows.append({"grouping": "individual scan", "n_groups": len(frame), "mean_scans_per_group": 1.0})
        out = pd.DataFrame(rows)
        out["effective_sample_size_if_split_here"] = out["n_groups"]
        return out

    def split_feasibility(self, test_fraction: float = 0.15) -> pd.DataFrame:
        """Expected positives per label in a test partition of this size.

        Any label expected to land below a handful of positives cannot support a
        stable per-label metric, let alone a confidence interval.
        """
        frame = self.labelled
        n_patients = frame["patient_id"].nunique()
        n_test_patients = max(1, int(round(test_fraction * n_patients)))
        rows = []
        for label in self.label_columns:
            positives = frame[frame[label] == 1]
            patients_with = positives["patient_id"].nunique()
            expected_patients = patients_with * test_fraction
            expected_positives = len(positives) * test_fraction
            rows.append(
                {
                    "label": label,
                    "n_positive_total": len(positives),
                    "n_patients_with_positive": patients_with,
                    "expected_test_patients_with_positive": round(expected_patients, 1),
                    "expected_test_positives": round(expected_positives, 1),
                    "probability_zero_in_test": round(
                        float(max(0.0, 1 - patients_with / n_patients) ** n_test_patients), 3
                    ),
                    "usable_in_test": expected_positives >= 10 and expected_patients >= 2,
                }
            )
        return pd.DataFrame(rows).sort_values("expected_test_positives")

    # ==================================================================
    # 6. images
    # ==================================================================
    def image_statistics(self, reader: Any, n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
        """Decode a random sample of scans and summarise size and intensity.

        Args:
            reader: A :class:`ParquetImageReader`.
            n_samples: How many scans to decode.
        """
        rng = np.random.default_rng(seed)
        frame = self.unique
        positions = rng.choice(len(frame), size=min(n_samples, len(frame)), replace=False)

        rows = []
        for position in positions:
            row = frame.iloc[position]
            image = reader.read_image(int(row["shard_index"]), int(row["row_in_shard"]))
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            rows.append(
                {
                    "row_uid": int(row["row_uid"]),
                    "patient_id": int(row["patient_id"]),
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "aspect_ratio": round(image.width / image.height, 3),
                    "mean_intensity": round(float(array.mean()), 4),
                    "std_intensity": round(float(array.std()), 4),
                    "min_intensity": round(float(array.min()), 4),
                    "max_intensity": round(float(array.max()), 4),
                    "pct_near_black": round(float((array < 0.05).mean()), 4),
                    "pct_saturated": round(float((array > 0.95).mean()), 4),
                }
            )
        return pd.DataFrame(rows)

    def image_statistics_summary(self, stats: pd.DataFrame) -> dict[str, Any]:
        """Condense :meth:`image_statistics` and compare to the paper's constants."""
        return {
            "n_sampled": len(stats),
            "unique_resolutions": sorted(
                {f"{w}x{h}" for w, h in zip(stats["width"], stats["height"])}
            ),
            "unique_modes": sorted(stats["mode"].unique().tolist()),
            "mean_intensity": round(float(stats["mean_intensity"].mean()), 4),
            "mean_intensity_std_across_scans": round(float(stats["mean_intensity"].std()), 4),
            "mean_within_scan_std": round(float(stats["std_intensity"].mean()), 4),
            "paper_reported_mean": 0.482,
            "paper_reported_std": 0.037,
            "mean_pct_near_black": round(float(stats["pct_near_black"].mean()), 4),
        }

    def sample_images(
        self,
        reader: Any,
        n: int = 8,
        label: str | None = None,
        present: bool = True,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Sample decoded scans, optionally conditioned on one biomarker.

        Args:
            label: Restrict to scans where this biomarker is present or absent.
            present: Which side of that condition to sample.
        """
        frame = self.labelled if label else self.unique
        if label:
            frame = frame[frame[label] == (1 if present else 0)]
        if frame.empty:
            return []

        rng = np.random.default_rng(seed)
        positions = rng.choice(len(frame), size=min(n, len(frame)), replace=False)
        out = []
        for position in positions:
            row = frame.iloc[position]
            image = reader.read_image(int(row["shard_index"]), int(row["row_in_shard"]))
            out.append(
                {
                    "image": image.convert("L"),
                    "row_uid": int(row["row_uid"]),
                    "patient_id": int(row["patient_id"]),
                    "disease": row.get("disease_name"),
                    "bcva": row.get("bcva"),
                    "cst": row.get("cst"),
                    "labels": [c for c in self.label_columns if row.get(c) == 1]
                    if "has_biomarkers" in row and row["has_biomarkers"]
                    else [],
                }
            )
        return out

    # ==================================================================
    # 7. report
    # ==================================================================
    def key_findings(self) -> list[str]:
        """Plain-language conclusions the experimental design has to respect."""
        findings: list[str] = []
        overview = self.cohort_overview()
        modelling = overview.iloc[-1]
        findings.append(
            f"Modelling set: {modelling['n_scans']:,} labelled scans from only "
            f"{modelling['n_patients']} patients - the effective sample size is the patient count, "
            "not the scan count."
        )

        leakage = self.leakage_risk_summary()
        scans_per_patient = leakage.loc[leakage["grouping"] == "patient", "mean_scans_per_group"].iloc[0]
        findings.append(
            f"Each patient contributes ~{scans_per_patient:.0f} labelled scans, so a random "
            "image-level split would put near-identical scans on both sides."
        )

        duplicates = self.duplicate_analysis()
        if duplicates.get("available"):
            findings.append(
                f"{duplicates['pct_duplicate_rows']}% of rows are byte-identical duplicates "
                f"({duplicates['n_duplicate_groups']:,} groups); "
                f"{'no' if not duplicates['cross_patient_leakage_risk'] else 'SOME'} groups cross a patient."
            )

        prevalence = self.label_prevalence()
        rare = prevalence[prevalence["prevalence"] < 0.01]
        if len(rare):
            findings.append(
                f"{len(rare)} of {len(prevalence)} biomarkers occur in under 1% of scans "
                f"({', '.join(rare['label'].tolist())}) - per-label metrics for these will be unstable."
            )

        feasibility = self.split_feasibility()
        unusable = feasibility[~feasibility["usable_in_test"]]
        if len(unusable):
            findings.append(
                f"{len(unusable)} labels are expected to yield too few test positives to evaluate "
                "at a 15% patient holdout; the 6-label MVP avoids all of them."
            )

        cardinality = self.label_cardinality()
        findings.append(
            f"Scans carry {cardinality['mean_labels_per_scan']} biomarkers on average across "
            f"{cardinality['n_unique_label_vectors']} distinct label vectors - genuinely multilabel, "
            "so per-label thresholds are needed rather than one global threshold."
        )

        association = self.clinical_label_association()
        if "cst_cohens_d" in association.columns and len(association):
            top = association.iloc[0]
            findings.append(
                f"CST separates '{top['label']}' most strongly (Cohen's d = {top['cst_cohens_d']}), "
                "which is where clinical fusion is most likely to help."
            )
        return findings
