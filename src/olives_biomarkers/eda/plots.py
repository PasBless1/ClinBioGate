"""Plot helpers for the EDA notebook.

Every method takes a DataFrame produced by :class:`OlivesEDA` and returns a
matplotlib Figure, so figures can be shown inline or written to
``outputs/figures`` without changing the call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


class EDAPlotter:
    """Consistent, colour-blind-safe figures for the OLIVES EDA.

    Args:
        style: Matplotlib style name.
        palette: Ordered categorical colours.
        figsize: Default figure size.
        dpi: Figure resolution.
    """

    PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860")
    SEQUENTIAL = "viridis"
    DIVERGING = "RdBu_r"

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        palette: Sequence[str] | None = None,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 110,
    ) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.palette = list(palette or self.PALETTE)
        self.figsize = figsize
        self.dpi = dpi
        available = plt.style.available
        self.plt.style.use(style if style in available else "default")
        self.plt.rcParams.update(
            {
                "figure.dpi": dpi,
                "axes.titlesize": 11,
                "axes.titleweight": "bold",
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.autolayout": False,
            }
        )

    # ------------------------------------------------------------------
    def save(self, figure: Any, path: str | Path) -> Path:
        """Write a figure to disk, creating parent directories."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, bbox_inches="tight", dpi=self.dpi)
        return out

    @staticmethod
    def _annotate_bars(axis: Any, bars: Any, fmt: str = "{:.0f}", offset: float = 0.01) -> None:
        """Write the value at the end of each horizontal bar."""
        span = axis.get_xlim()[1]
        for bar in bars:
            width = bar.get_width()
            axis.text(
                width + offset * span,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(width),
                va="center",
                fontsize=8,
            )

    # ------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------
    def label_prevalence(self, prevalence: pd.DataFrame, log_scale: bool = True) -> Any:
        """Horizontal bars of positive counts per biomarker."""
        data = prevalence.sort_values("n_positive")
        figure, axes = self.plt.subplots(1, 2, figsize=(13, max(4, 0.38 * len(data))))

        bars = axes[0].barh(data["label"], data["n_positive"], color=self.palette[0])
        axes[0].set_xlabel("positive scans" + (" (log scale)" if log_scale else ""))
        axes[0].set_title("Positive scans per biomarker")
        if log_scale:
            axes[0].set_xscale("log")
        self._annotate_bars(axes[0], bars, "{:.0f}")

        colors = [
            self.palette[3] if p < 0.01 else self.palette[1] if p < 0.05 else self.palette[2]
            for p in data["prevalence"]
        ]
        axes[1].barh(data["label"], 100 * data["prevalence"], color=colors)
        axes[1].axvline(1.0, color="grey", ls="--", lw=1, label="1% (very rare)")
        axes[1].axvline(5.0, color="grey", ls=":", lw=1, label="5% (rare)")
        axes[1].set_xlabel("prevalence (%)")
        axes[1].set_title("Prevalence among labelled scans")
        axes[1].legend(loc="lower right")
        figure.tight_layout()
        return figure

    def label_cardinality(self, distribution: pd.DataFrame) -> Any:
        """Histogram of how many biomarkers appear on one scan."""
        figure, axis = self.plt.subplots(figsize=self.figsize)
        axis.bar(distribution["n_labels_present"], distribution["n_scans"], color=self.palette[0])
        for _, row in distribution.iterrows():
            axis.text(
                row["n_labels_present"],
                row["n_scans"],
                f"{row['pct']:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_xlabel("biomarkers present on one scan")
        axis.set_ylabel("number of scans")
        axis.set_title("Label cardinality per B-scan")
        figure.tight_layout()
        return figure

    def cooccurrence_heatmap(
        self, matrix: pd.DataFrame, title: str = "Biomarker co-occurrence (Jaccard)"
    ) -> Any:
        """Heatmap of pairwise label co-occurrence."""
        figure, axis = self.plt.subplots(figsize=(9, 7.5))
        image = axis.imshow(matrix.to_numpy(), cmap=self.SEQUENTIAL, vmin=0, vmax=1)
        axis.set_xticks(range(len(matrix.columns)))
        axis.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
        axis.set_yticks(range(len(matrix.index)))
        axis.set_yticklabels(matrix.index, fontsize=8)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        return figure

    def correlation_heatmap(self, matrix: pd.DataFrame, title: str = "Label correlation") -> Any:
        """Diverging heatmap for a correlation matrix."""
        figure, axis = self.plt.subplots(figsize=(9, 7.5))
        image = axis.imshow(matrix.to_numpy(), cmap=self.DIVERGING, vmin=-1, vmax=1)
        axis.set_xticks(range(len(matrix.columns)))
        axis.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
        axis.set_yticks(range(len(matrix.index)))
        axis.set_yticklabels(matrix.index, fontsize=8)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        return figure

    def prevalence_by_disease(self, table: pd.DataFrame) -> Any:
        """Grouped bars comparing biomarker prevalence in DR versus DME."""
        columns = [c for c in table.columns if c.endswith("_prevalence")]
        if not columns:
            return None
        data = table.sort_values(columns[0])
        y = np.arange(len(data))
        height = 0.38
        figure, axis = self.plt.subplots(figsize=(10, max(4, 0.4 * len(data))))
        for offset, (column, color) in enumerate(zip(columns, self.palette)):
            axis.barh(
                y + (offset - 0.5) * height,
                100 * data[column],
                height=height,
                label=column.replace("_prevalence", ""),
                color=color,
            )
        axis.set_yticks(y)
        axis.set_yticklabels(data["label"], fontsize=8)
        axis.set_xlabel("prevalence (%)")
        axis.set_title("Biomarker prevalence by disease")
        axis.legend()
        figure.tight_layout()
        return figure

    # ------------------------------------------------------------------
    # clinical
    # ------------------------------------------------------------------
    def clinical_distributions(self, frame: pd.DataFrame, by_disease: bool = True) -> Any:
        """Histograms of BCVA and CST, optionally split by disease."""
        features = [f for f in ("bcva", "cst") if f in frame.columns]
        figure, axes = self.plt.subplots(1, len(features), figsize=(6 * len(features), 4.2))
        axes = np.atleast_1d(axes)

        for axis, feature in zip(axes, features):
            if by_disease and "disease_name" in frame.columns:
                for color, (disease, group) in zip(
                    self.palette, frame.groupby("disease_name")
                ):
                    axis.hist(
                        group[feature].dropna(),
                        bins=40,
                        alpha=0.6,
                        label=str(disease),
                        color=color,
                    )
                axis.legend()
            else:
                axis.hist(frame[feature].dropna(), bins=40, color=self.palette[0])
            axis.set_xlabel(feature.upper())
            axis.set_ylabel("scans")
            axis.set_title(f"{feature.upper()} distribution")
        figure.tight_layout()
        return figure

    def clinical_scatter(self, frame: pd.DataFrame, label: str | None = None) -> Any:
        """BCVA against CST, optionally coloured by one biomarker."""
        figure, axis = self.plt.subplots(figsize=(7, 5.5))
        if label and label in frame.columns:
            for value, color, name in [(0, self.palette[0], "absent"), (1, self.palette[3], "present")]:
                subset = frame[frame[label] == value]
                axis.scatter(
                    subset["cst"], subset["bcva"], s=6, alpha=0.35, color=color, label=f"{label} {name}"
                )
            axis.legend()
        else:
            axis.scatter(frame["cst"], frame["bcva"], s=6, alpha=0.3, color=self.palette[0])
        axis.set_xlabel("CST (central subfield thickness, um)")
        axis.set_ylabel("BCVA (ETDRS letters)")
        axis.set_title("Clinical feature space")
        figure.tight_layout()
        return figure

    def clinical_label_association(self, association: pd.DataFrame, feature: str = "cst") -> Any:
        """Effect size of one clinical feature across biomarkers."""
        column = f"{feature}_cohens_d"
        if column not in association.columns:
            return None
        data = association.dropna(subset=[column]).sort_values(column)
        colors = [self.palette[3] if v < 0 else self.palette[2] for v in data[column]]
        figure, axis = self.plt.subplots(figsize=(9, max(4, 0.38 * len(data))))
        axis.barh(data["label"], data[column], color=colors)
        axis.axvline(0, color="black", lw=1)
        for threshold in (-0.5, 0.5):
            axis.axvline(threshold, color="grey", ls="--", lw=0.8)
        axis.set_xlabel(f"Cohen's d ({feature.upper()}: present minus absent)")
        axis.set_title(f"How well {feature.upper()} separates each biomarker")
        figure.tight_layout()
        return figure

    # ------------------------------------------------------------------
    # cohort and longitudinal
    # ------------------------------------------------------------------
    def per_patient_distribution(self, summary: pd.DataFrame) -> Any:
        """Scans, visits and labelled scans per patient."""
        figure, axes = self.plt.subplots(1, 3, figsize=(15, 4))
        panels = [
            ("n_scans", "unique scans per patient"),
            ("n_visits", "inferred visits per patient"),
            ("n_labelled_scans", "labelled scans per patient"),
        ]
        for axis, (column, title) in zip(axes, panels):
            if column not in summary.columns:
                continue
            axis.hist(summary[column].dropna(), bins=25, color=self.palette[0])
            axis.axvline(
                summary[column].median(),
                color=self.palette[3],
                ls="--",
                label=f"median {summary[column].median():.0f}",
            )
            axis.set_xlabel(column)
            axis.set_ylabel("patients")
            axis.set_title(title)
            axis.legend()
        figure.tight_layout()
        return figure

    def treatment_progression(self, progression: pd.DataFrame) -> Any:
        """Mean BCVA change and cohort attrition across visits (paper Figure 5)."""
        figure, axes = self.plt.subplots(1, 3, figsize=(15, 4))

        axes[0].bar(progression["visit_index"], progression["n_eyes"], color=self.palette[0])
        axes[0].set_xlabel("visit index (inferred)")
        axes[0].set_ylabel("eyes remaining")
        axes[0].set_title("Cohort attrition across visits")

        axes[1].plot(
            progression["visit_index"],
            progression["mean_bcva_change_from_first"],
            marker="o",
            color=self.palette[2],
        )
        axes[1].fill_between(
            progression["visit_index"],
            progression["mean_bcva_change_from_first"] - progression["std_bcva_change_from_first"],
            progression["mean_bcva_change_from_first"] + progression["std_bcva_change_from_first"],
            alpha=0.2,
            color=self.palette[2],
        )
        axes[1].axhline(0, color="black", lw=1)
        axes[1].set_xlabel("visit index (inferred)")
        axes[1].set_ylabel("BCVA change vs first visit")
        axes[1].set_title("Mean BCVA improvement")

        axes[2].bar(
            progression["visit_index"],
            progression["pct_improved_vs_previous"],
            color=self.palette[1],
        )
        axes[2].axhline(50, color="black", ls="--", lw=1, label="50%")
        axes[2].set_xlabel("visit index (inferred)")
        axes[2].set_ylabel("% eyes improved vs previous")
        axes[2].set_title("Visit-to-visit improvement")
        axes[2].legend()
        figure.tight_layout()
        return figure

    def patient_label_heatmap(self, prevalence_by_patient: pd.DataFrame) -> Any:
        """Per-patient label prevalence, showing how concentrated rare labels are."""
        figure, axis = self.plt.subplots(figsize=(12, 7))
        image = axis.imshow(
            prevalence_by_patient.to_numpy().T, aspect="auto", cmap=self.SEQUENTIAL, vmin=0, vmax=1
        )
        axis.set_yticks(range(len(prevalence_by_patient.columns)))
        axis.set_yticklabels(prevalence_by_patient.columns, fontsize=8)
        axis.set_xlabel("patient (ordered by id)")
        axis.set_title("Per-patient biomarker prevalence")
        figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02, label="prevalence within patient")
        figure.tight_layout()
        return figure

    def split_feasibility(self, feasibility: pd.DataFrame) -> Any:
        """Expected test-set positives per label at the planned holdout size."""
        data = feasibility.sort_values("expected_test_positives")
        colors = [self.palette[2] if ok else self.palette[3] for ok in data["usable_in_test"]]
        figure, axis = self.plt.subplots(figsize=(9, max(4, 0.38 * len(data))))
        axis.barh(data["label"], data["expected_test_positives"], color=colors)
        axis.axvline(10, color="black", ls="--", lw=1, label="10 positives (minimum usable)")
        axis.set_xscale("symlog")
        axis.set_xlabel("expected positives in a 15% patient holdout")
        axis.set_title("Can each label be evaluated on the test set?")
        axis.legend()
        figure.tight_layout()
        return figure

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------
    def image_grid(
        self, samples: list[dict[str, Any]], n_cols: int = 4, title: str = "Sample OCT B-scans"
    ) -> Any:
        """Grid of decoded scans annotated with their clinical context."""
        if not samples:
            return None
        n_rows = int(np.ceil(len(samples) / n_cols))
        figure, axes = self.plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.6 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for axis, sample in zip(axes, samples):
            axis.imshow(np.asarray(sample["image"]), cmap="gray")
            caption = f"P{sample['patient_id']} · {sample.get('disease', '?')}"
            if sample.get("bcva") is not None and not pd.isna(sample.get("bcva")):
                caption += f"\nBCVA {sample['bcva']:.0f} · CST {sample['cst']:.0f}"
            if sample.get("labels"):
                shown = ", ".join(sample["labels"][:3])
                caption += f"\n{shown}"
            axis.set_title(caption, fontsize=7)
            axis.axis("off")
        for axis in axes[len(samples) :]:
            axis.axis("off")
        figure.suptitle(title, fontsize=12, fontweight="bold")
        figure.tight_layout()
        return figure

    def image_statistics(self, stats: pd.DataFrame) -> Any:
        """Intensity and geometry summaries over a sample of scans."""
        figure, axes = self.plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(stats["mean_intensity"], bins=30, color=self.palette[0])
        axes[0].axvline(0.482, color=self.palette[3], ls="--", label="paper mean 0.482")
        axes[0].set_xlabel("mean intensity")
        axes[0].set_ylabel("scans")
        axes[0].set_title("Per-scan mean intensity")
        axes[0].legend()

        axes[1].hist(stats["std_intensity"], bins=30, color=self.palette[1])
        axes[1].set_xlabel("within-scan intensity std")
        axes[1].set_title("Per-scan contrast")

        axes[2].scatter(stats["width"], stats["height"], s=24, alpha=0.6, color=self.palette[2])
        axes[2].set_xlabel("width (px)")
        axes[2].set_ylabel("height (px)")
        axes[2].set_title("Image resolutions")
        figure.tight_layout()
        return figure

    def duplicate_summary(self, duplicates: dict[str, Any]) -> Any:
        """Composition of the dataset by duplication status."""
        if not duplicates.get("available"):
            return None
        unique_only = duplicates["n_rows"] - duplicates["n_duplicate_rows"]
        adjacent = duplicates["n_adjacent_duplicate_rows"]
        other = duplicates["n_duplicate_rows"] - adjacent

        figure, axis = self.plt.subplots(figsize=(7, 4.5))
        labels = ["unique images", "adjacent duplicates", "repeated-visit duplicates"]
        values = [unique_only, adjacent, other]
        bars = axis.barh(labels, values, color=[self.palette[2], self.palette[1], self.palette[3]])
        for bar, value in zip(bars, values):
            axis.text(
                value + 0.01 * duplicates["n_rows"],
                bar.get_y() + bar.get_height() / 2,
                f"{value:,} ({100 * value / duplicates['n_rows']:.1f}%)",
                va="center",
                fontsize=9,
            )
        axis.set_xlabel("rows")
        axis.set_title("Dataset composition by duplication status")
        figure.tight_layout()
        return figure
