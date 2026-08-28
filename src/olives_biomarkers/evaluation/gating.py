"""Analysis of what the clinical gate actually learned.

A gated model that beats concatenation is only interesting if the gate is doing
something interpretable. These tools answer three questions:

1. Did the gate move away from its identity initialisation at all?
2. Does the gate respond to the clinical inputs, or has it collapsed to a
   constant that the head could have absorbed into a bias?
3. Do the channels it modulates most relate to the biomarkers CST should inform?

A gate that stays at identity, or varies randomly with respect to BCVA and CST,
means the gating mechanism is inert and Model D is Model C with extra parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.gating")


class GateAnalyzer:
    """Collects and interprets gate activations from a trained gated model.

    Args:
        model: A trained :class:`GatedFusionModel`.
        device: Torch device string.
    """

    def __init__(self, model: Any, device: str = "cpu") -> None:
        if not hasattr(model, "gate"):
            raise TypeError(f"{type(model).__name__} has no clinical gate to analyse")
        self.model = model.to(device)
        self.device = device

    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect(self, loader: Any) -> dict[str, np.ndarray]:
        """Run the loader and return gate values, scales and the clinical inputs."""
        self.model.eval()
        gates, scales, clinicals, targets, patients = [], [], [], [], []

        for batch in loader:
            clinical = batch["clinical"].to(self.device)
            embedding = self.model.clinical_encoder(clinical)
            gate = self.model.gate(embedding)
            gates.append(gate.cpu().numpy())
            scales.append(self.model.gate.scale(gate).cpu().numpy())
            clinicals.append(batch["clinical"].numpy())
            targets.append(batch["target"].numpy())
            patients.append(batch["patient_id"].numpy())

        return {
            "gate": np.concatenate(gates),
            "scale": np.concatenate(scales),
            "clinical": np.concatenate(clinicals),
            "target": np.concatenate(targets),
            "patient_id": np.concatenate(patients),
        }

    # ------------------------------------------------------------------
    def summary(self, collected: dict[str, np.ndarray]) -> dict[str, float]:
        """Whether the gate moved off identity and how much it varies."""
        gate = collected["gate"]
        scale = collected["scale"]
        # Variation *across samples*, per channel: this is what distinguishes a
        # real gate from a learned constant.
        per_channel_std = gate.std(axis=0)
        return {
            "gate_mean": float(gate.mean()),
            "gate_std_overall": float(gate.std()),
            "gate_std_across_samples_mean": float(per_channel_std.mean()),
            "gate_std_across_samples_max": float(per_channel_std.max()),
            "scale_mean": float(scale.mean()),
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
            "fraction_channels_essentially_constant": float((per_channel_std < 0.01).mean()),
            "fraction_scale_suppressing": float((scale < 0.5).mean()),
            "fraction_scale_amplifying": float((scale > 1.5).mean()),
            "mean_absolute_deviation_from_identity": float(np.abs(scale - 1.0).mean()),
        }

    def interpret(self, summary: dict[str, float]) -> list[str]:
        """Plain-language reading of :meth:`summary`."""
        notes: list[str] = []
        if summary["mean_absolute_deviation_from_identity"] < 0.02:
            notes.append(
                "The gate never left its identity initialisation (mean |scale - 1| < 0.02). "
                "Model D is behaving as Model C with extra parameters."
            )
        if summary["fraction_channels_essentially_constant"] > 0.9:
            notes.append(
                f"{100 * summary['fraction_channels_essentially_constant']:.0f}% of gate channels "
                "barely vary across samples: the gate has largely collapsed to a constant the "
                "classification head could absorb, so it is not conditioning on clinical input."
            )
        elif summary["gate_std_across_samples_mean"] > 0.05:
            notes.append(
                "The gate varies meaningfully across samples, so it is genuinely conditioning "
                "on BCVA and CST rather than acting as a learned constant."
            )
        if summary["fraction_scale_suppressing"] > 0.2:
            notes.append(
                f"{100 * summary['fraction_scale_suppressing']:.0f}% of gated activations are "
                "damped below half strength - clinical context is actively suppressing image "
                "features, not merely amplifying them."
            )
        return notes

    # ------------------------------------------------------------------
    def clinical_response(
        self, collected: dict[str, np.ndarray], feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        """Correlate each clinical input with the mean gate value per sample.

        A gate that is genuinely clinically driven should show non-trivial
        correlation with at least one clinical feature.
        """
        from scipy import stats

        clinical = collected["clinical"]
        mean_gate = collected["gate"].mean(axis=1)
        names = feature_names or [f"clinical_{i}" for i in range(clinical.shape[1])]

        rows = []
        for index, name in enumerate(names):
            values = clinical[:, index]
            if np.std(values) < 1e-8:
                rows.append(
                    {"feature": name, "spearman_r": np.nan, "p_value": np.nan, "note": "constant"}
                )
                continue
            correlation, p_value = stats.spearmanr(values, mean_gate)
            rows.append(
                {
                    "feature": name,
                    "spearman_r": round(float(correlation), 4),
                    "p_value": float(p_value),
                    "note": "responds" if abs(correlation) > 0.1 else "weak",
                }
            )
        return pd.DataFrame(rows)

    def gate_by_label(
        self, collected: dict[str, np.ndarray], label_names: list[str]
    ) -> pd.DataFrame:
        """Mean gate value for scans with versus without each biomarker.

        If gating is doing clinically sensible work, the fluid-related biomarkers
        (which CST tracks) should separate more than the vitreous-face ones.
        """
        mean_gate = collected["gate"].mean(axis=1)
        targets = collected["target"]
        rows = []
        for index, label in enumerate(label_names):
            present = mean_gate[targets[:, index] == 1]
            absent = mean_gate[targets[:, index] == 0]
            if len(present) == 0 or len(absent) == 0:
                rows.append({"label": label, "n_present": len(present), "delta": np.nan})
                continue
            pooled = mean_gate.std()
            rows.append(
                {
                    "label": label,
                    "n_present": int(len(present)),
                    "gate_present": round(float(present.mean()), 4),
                    "gate_absent": round(float(absent.mean()), 4),
                    "delta": round(float(present.mean() - absent.mean()), 4),
                    "cohens_d": round(float((present.mean() - absent.mean()) / pooled), 3)
                    if pooled > 1e-8
                    else np.nan,
                }
            )
        table = pd.DataFrame(rows)
        if "cohens_d" in table.columns:
            table = table.reindex(table["cohens_d"].abs().sort_values(ascending=False).index)
        return table.reset_index(drop=True)

    def channel_activity(self, collected: dict[str, np.ndarray], top_n: int = 20) -> pd.DataFrame:
        """The gate channels that deviate most from identity."""
        scale = collected["scale"]
        deviation = np.abs(scale.mean(axis=0) - 1.0)
        order = np.argsort(deviation)[::-1][:top_n]
        return pd.DataFrame(
            {
                "channel": order,
                "mean_scale": scale[:, order].mean(axis=0).round(4),
                "std_scale": scale[:, order].std(axis=0).round(4),
                "abs_deviation_from_identity": deviation[order].round(4),
            }
        )
