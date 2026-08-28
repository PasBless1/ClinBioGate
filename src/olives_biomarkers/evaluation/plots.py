"""Figures for the modelling phases: curves, comparisons, calibration, Grad-CAM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


class ResultsPlotter:
    """Consistent figures for training, comparison, calibration and explanation.

    Mirrors :class:`~olives_biomarkers.eda.plots.EDAPlotter` so figures across the
    project share one visual language.
    """

    PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860")
    MODEL_COLORS = {
        "clinical_only": "#8172B3",
        "oct_only": "#4C72B0",
        "concat_fusion": "#DD8452",
        "gated_fusion": "#55A868",
    }

    def __init__(self, figsize: tuple[float, float] = (10, 5), dpi: int = 110) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.figsize = figsize
        self.dpi = dpi
        style = "seaborn-v0_8-whitegrid"
        self.plt.style.use(style if style in plt.style.available else "default")
        self.plt.rcParams.update(
            {
                "figure.dpi": dpi,
                "axes.titlesize": 11,
                "axes.titleweight": "bold",
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            }
        )

    def save(self, figure: Any, path: str | Path) -> Path:
        """Write a figure to disk."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, bbox_inches="tight", dpi=self.dpi)
        return out

    def _color(self, model: str, index: int = 0) -> str:
        return self.MODEL_COLORS.get(model, self.PALETTE[index % len(self.PALETTE)])

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def training_curves(self, histories: dict[str, pd.DataFrame], monitor: str = "val_macro_auprc") -> Any:
        """Loss and monitored metric per epoch, one line per model."""
        figure, axes = self.plt.subplots(1, 2, figsize=(13, 4.4))
        for index, (name, history) in enumerate(histories.items()):
            color = self._color(name, index)
            axes[0].plot(history["epoch"], history["train_loss"], color=color, label=f"{name} train")
            axes[0].plot(
                history["epoch"], history["val_loss"], color=color, ls="--", label=f"{name} val"
            )
            if monitor in history.columns:
                axes[1].plot(history["epoch"], history[monitor], color=color, marker="o", label=name)
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Training and validation loss")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel(monitor)
        axes[1].set_title(f"Validation {monitor}")
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        return figure

    # ------------------------------------------------------------------
    # comparison
    # ------------------------------------------------------------------
    def model_comparison(
        self, table: pd.DataFrame, metrics: Sequence[str] = ("macro_f1", "macro_auroc", "macro_auprc")
    ) -> Any:
        """Grouped bars comparing models on the headline metrics."""
        available = [m for m in metrics if m in table.columns]
        figure, axes = self.plt.subplots(1, len(available), figsize=(5 * len(available), 4.2))
        axes = np.atleast_1d(axes)

        for axis, metric in zip(axes, available):
            grouped = table.groupby("model")[metric].agg(["mean", "std", "count"])
            colors = [self._color(m, i) for i, m in enumerate(grouped.index)]
            bars = axis.bar(
                grouped.index,
                grouped["mean"],
                yerr=grouped["std"].fillna(0),
                capsize=4,
                color=colors,
            )
            for bar, value in zip(bars, grouped["mean"]):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            axis.set_ylabel(metric)
            axis.set_title(metric)
            axis.tick_params(axis="x", rotation=30)
        figure.suptitle("Model comparison (error bars: std across seeds/folds)", fontweight="bold")
        figure.tight_layout()
        return figure

    def bootstrap_intervals(self, bootstrap: pd.DataFrame, metric: str = "macro_auprc") -> Any:
        """Point estimates with patient-level bootstrap intervals."""
        scoped = bootstrap[bootstrap["metric"] == metric].copy()
        if scoped.empty:
            return None
        scoped = scoped.sort_values("point_estimate")
        y = np.arange(len(scoped))
        figure, axis = self.plt.subplots(figsize=(9, max(3, 0.7 * len(scoped))))
        for offset, (_, row) in enumerate(scoped.iterrows()):
            color = self._color(str(row.get("model", "")), offset)
            axis.plot(
                [row["ci_lower"], row["ci_upper"]], [offset, offset], color=color, lw=3, alpha=0.6
            )
            axis.plot(row["point_estimate"], offset, "o", color=color, markersize=9)
        axis.set_yticks(y)
        axis.set_yticklabels(scoped.get("model", scoped["run_id"]))
        axis.set_xlabel(metric)
        axis.set_title(f"{metric} with 95% patient-level bootstrap CI")
        figure.tight_layout()
        return figure

    def per_label_comparison(self, pivot: pd.DataFrame, metric: str = "auprc") -> Any:
        """Per-label metric across models, ordered by label frequency."""
        models = [c for c in pivot.columns if c != "n_positive"]
        if not models:
            return None
        data = pivot.copy()
        y = np.arange(len(data))
        height = 0.8 / len(models)
        figure, axis = self.plt.subplots(figsize=(10, max(4, 0.45 * len(data))))
        for index, model in enumerate(models):
            axis.barh(
                y + (index - (len(models) - 1) / 2) * height,
                data[model],
                height=height,
                label=model,
                color=self._color(model, index),
            )
        axis.set_yticks(y)
        axis.set_yticklabels([f"{i}  (n={int(n)})" for i, n in zip(data.index, data["n_positive"])])
        axis.set_xlabel(metric)
        axis.set_title(f"Per-label {metric} by model")
        axis.legend()
        figure.tight_layout()
        return figure

    # ------------------------------------------------------------------
    # calibration and uncertainty
    # ------------------------------------------------------------------
    def reliability_diagram(
        self, curves: dict[str, pd.DataFrame], title: str = "Reliability"
    ) -> Any:
        """Observed frequency against predicted confidence, before and after."""
        figure, axis = self.plt.subplots(figsize=(6, 5.6))
        axis.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for index, (name, curve) in enumerate(curves.items()):
            valid = curve.dropna(subset=["mean_confidence", "observed_frequency"])
            axis.plot(
                valid["mean_confidence"],
                valid["observed_frequency"],
                marker="o",
                color=self.PALETTE[index % len(self.PALETTE)],
                label=name,
            )
        axis.set_xlabel("mean predicted probability")
        axis.set_ylabel("observed frequency")
        axis.set_title(title)
        axis.legend()
        figure.tight_layout()
        return figure

    def calibration_comparison(self, comparison: pd.DataFrame) -> Any:
        """Per-label ECE and Brier before versus after temperature scaling."""
        data = comparison[comparison["reliable"]].sort_values("ece_pre") if "reliable" in comparison else comparison
        if data.empty:
            data = comparison
        y = np.arange(len(data))
        figure, axes = self.plt.subplots(1, 2, figsize=(13, max(4, 0.42 * len(data))))
        for axis, (pre, post, name) in zip(
            axes, [("ece_pre", "ece_post", "ECE"), ("brier_pre", "brier_post", "Brier")]
        ):
            axis.barh(y - 0.2, data[pre], height=0.4, label="before", color=self.PALETTE[3])
            axis.barh(y + 0.2, data[post], height=0.4, label="after", color=self.PALETTE[2])
            axis.set_yticks(y)
            axis.set_yticklabels(data["label"], fontsize=8)
            axis.set_xlabel(name)
            axis.set_title(f"{name}: temperature scaling")
            axis.legend()
        figure.tight_layout()
        return figure

    def coverage_curve(self, curves: dict[str, pd.DataFrame], metric: str = "macro_f1") -> Any:
        """Retained-set performance as coverage decreases."""
        figure, axes = self.plt.subplots(1, 2, figsize=(13, 4.4))
        for index, (name, curve) in enumerate(curves.items()):
            color = self._color(name, index)
            axes[0].plot(100 * curve["coverage"], curve[metric], marker="o", color=color, label=name)
            axes[1].plot(
                100 * curve["coverage"], curve["hamming_loss"], marker="o", color=color, label=name
            )
        axes[0].set_xlabel("coverage (% retained)")
        axes[0].set_ylabel(metric)
        axes[0].set_title(f"{metric} on the retained set")
        axes[0].invert_xaxis()
        axes[0].legend()
        axes[1].set_xlabel("coverage (% retained)")
        axes[1].set_ylabel("hamming loss")
        axes[1].set_title("Error on the retained set")
        axes[1].invert_xaxis()
        axes[1].legend()
        figure.tight_layout()
        return figure

    def uncertainty_distribution(
        self, uncertainty: np.ndarray, error_rate: np.ndarray, n_bins: int = 20
    ) -> Any:
        """Uncertainty histogram split by whether the prediction was wrong."""
        any_wrong = error_rate > 0
        figure, axes = self.plt.subplots(1, 2, figsize=(13, 4.2))

        axes[0].hist(
            [uncertainty[~any_wrong], uncertainty[any_wrong]],
            bins=n_bins,
            label=["all labels correct", "at least one wrong"],
            color=[self.PALETTE[2], self.PALETTE[3]],
            stacked=False,
        )
        axes[0].set_xlabel("predictive uncertainty")
        axes[0].set_ylabel("scans")
        axes[0].set_title("Uncertainty by correctness")
        axes[0].legend()

        order = np.argsort(uncertainty)
        binned = np.array_split(order, n_bins)
        axes[1].plot(
            [uncertainty[b].mean() for b in binned],
            [error_rate[b].mean() for b in binned],
            marker="o",
            color=self.PALETTE[0],
        )
        axes[1].set_xlabel("mean uncertainty in bin")
        axes[1].set_ylabel("mean per-label error rate")
        axes[1].set_title("Does uncertainty track error?")
        figure.tight_layout()
        return figure

    # ------------------------------------------------------------------
    # gating
    # ------------------------------------------------------------------
    def gate_distribution(self, collected: dict[str, np.ndarray]) -> Any:
        """Where the gate sits and how far it moved from identity."""
        gate = collected["gate"]
        scale = collected["scale"]
        figure, axes = self.plt.subplots(1, 3, figsize=(15, 4.2))

        axes[0].hist(gate.ravel(), bins=50, color=self.PALETTE[0])
        axes[0].axvline(0.5, color="black", ls="--", label="identity (residual)")
        axes[0].set_xlabel("gate value")
        axes[0].set_title("Gate activation distribution")
        axes[0].legend()

        axes[1].hist(scale.ravel(), bins=50, color=self.PALETTE[1])
        axes[1].axvline(1.0, color="black", ls="--", label="no modulation")
        axes[1].set_xlabel("applied scale")
        axes[1].set_title("Modulation applied to image features")
        axes[1].legend()

        per_channel = gate.std(axis=0)
        axes[2].hist(per_channel, bins=40, color=self.PALETTE[2])
        axes[2].axvline(0.01, color=self.PALETTE[3], ls="--", label="effectively constant")
        axes[2].set_xlabel("per-channel std across samples")
        axes[2].set_title("Does the gate vary with input?")
        axes[2].legend()
        figure.tight_layout()
        return figure

    # ------------------------------------------------------------------
    # explainability
    # ------------------------------------------------------------------
    def gradcam_overlay(
        self,
        image: np.ndarray,
        results: Sequence[Any],
        n_cols: int = 4,
        title: str = "Grad-CAM per biomarker",
    ) -> Any:
        """One overlay per biomarker for a single scan."""
        if len(results) == 0:
            return None
        n_panels = len(results) + 1
        n_rows = int(np.ceil(n_panels / n_cols))
        figure, axes = self.plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.6 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("OCT B-scan", fontsize=9)
        axes[0].axis("off")

        for axis, result in zip(axes[1:], results):
            axis.imshow(image, cmap="gray")
            axis.imshow(result.heatmap, cmap="jet", alpha=0.45)
            axis.set_title(
                f"{result.label_name}\np={result.probability:.2f} · {result.outcome}", fontsize=8
            )
            axis.axis("off")
        for axis in axes[n_panels:]:
            axis.axis("off")
        figure.suptitle(title, fontweight="bold")
        figure.tight_layout()
        return figure

    def outcome_gallery(
        self, panels: dict[str, list[dict[str, Any]]], label: str, n_cols: int = 4
    ) -> Any:
        """A row per outcome category (TP / TN / FP / FN) for one biomarker."""
        categories = [k for k, v in panels.items() if v]
        if not categories:
            return None
        figure, axes = self.plt.subplots(
            len(categories), n_cols, figsize=(3.4 * n_cols, 2.6 * len(categories))
        )
        axes = np.atleast_2d(axes)

        for row, category in enumerate(categories):
            for col in range(n_cols):
                axis = axes[row, col]
                axis.axis("off")
                if col >= len(panels[category]):
                    continue
                panel = panels[category][col]
                axis.imshow(panel["image"], cmap="gray")
                if panel.get("heatmap") is not None:
                    axis.imshow(panel["heatmap"], cmap="jet", alpha=0.45)
                axis.set_title(
                    f"{category}\np={panel['probability']:.2f}", fontsize=8
                )
        figure.suptitle(f"Grad-CAM outcomes for {label}", fontweight="bold")
        figure.tight_layout()
        return figure
