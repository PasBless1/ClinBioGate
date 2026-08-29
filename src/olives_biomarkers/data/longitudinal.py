"""Within-eye clinical context and the controls that keep it honest.

The cross-sectional clinical features (absolute BCVA and CST at the current
visit) are largely redundant with the B-scan: CST is a thickness measurement of
the very volume being classified, and its between-patient variance dominates its
within-patient variance. That is why a clinical gate trained on absolute values
degenerates into global amplification -- the only thing absolute CST reliably
encodes across patients is overall severity.

The information a single B-scan does *not* contain is the **within-eye
contrast**: how this eye's thickness and acuity have moved relative to its own
baseline. On the labelled OLIVES subset every eye has exactly two graded visits,
and centring both the clinical value and the biomarker state within an eye --
which removes patient identity by construction -- leaves a measurable
association for several biomarkers.

Three classes here:

* :class:`LongitudinalClinicalFeatures` derives the contrast columns.
* :class:`ClinicalPerturbation` implements the control ladder that separates
  clinical signal from visit identity.
* :class:`WithinEyeAssociation` measures the association on the training fold
  only, so target labels can be pre-registered before any test data is touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from olives_biomarkers.utils.io import JsonIO
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.longitudinal")

EYE_KEYS = ("patient_id", "eye_id")


class LongitudinalClinicalFeatures:
    """Adds within-eye contrast columns to the modelling frame.

    For each eye the *baseline* visit is its earliest (lowest ``visit_index``).
    Every scan then carries the eye's baseline value and its signed change from
    that baseline. The baseline visit itself gets a delta of zero and a flag, so
    the model can tell "no change" apart from "no history".

    This is causal by construction: a visit's delta depends only on itself and on
    an earlier visit of the same eye. It also cannot leak across the split, since
    partitions are assigned by patient and an eye never spans two partitions.

    Args:
        features: Base clinical columns to derive contrasts for.
        visit_key: Column identifying a visit.
        order_key: Column giving temporal order within an eye.

    Example:
        >>> frame = LongitudinalClinicalFeatures().transform(frame)
        >>> LongitudinalClinicalFeatures().derived_names()
        ['bcva_baseline', 'bcva_delta', 'cst_baseline', 'cst_delta', 'is_baseline_visit']
    """

    def __init__(
        self,
        features: tuple[str, ...] = ("bcva", "cst"),
        visit_key: str = "visit_uid",
        order_key: str = "visit_index",
    ) -> None:
        self.features = tuple(features)
        self.visit_key = visit_key
        self.order_key = order_key

    # ------------------------------------------------------------------
    def derived_names(self) -> list[str]:
        """Columns this transform adds, in a stable order."""
        names: list[str] = []
        for feature in sorted(self.features):
            names += [f"{feature}_baseline", f"{feature}_delta"]
        return names + ["is_baseline_visit"]

    def required_columns(self) -> list[str]:
        """Columns the input frame must already provide."""
        return [*EYE_KEYS, self.visit_key, self.order_key, *self.features]

    # ------------------------------------------------------------------
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``frame`` with the contrast columns attached.

        Idempotent: calling it twice recomputes rather than compounding, so a
        notebook that re-runs a cell cannot silently build deltas of deltas.

        Args:
            frame: Modelling frame with visit structure and clinical columns.

        Raises:
            KeyError: A required column is absent.
        """
        missing = [c for c in self.required_columns() if c not in frame.columns]
        if missing:
            raise KeyError(
                f"longitudinal features need columns {missing}; build the manifest with "
                "visit inference enabled before requesting them"
            )

        out = frame.copy()
        groups = out.groupby(list(EYE_KEYS), sort=False)

        # Rank visits within the eye. `visit_index` is not contiguous after visit
        # inference, so order matters but its absolute value does not.
        visit_order = groups[self.order_key].rank(method="dense").astype(int) - 1
        out["visit_order"] = visit_order
        out["is_baseline_visit"] = (visit_order == 0).astype(np.float32)

        for feature in sorted(self.features):
            baseline = self._baseline_value(out, feature)
            out[f"{feature}_baseline"] = baseline
            out[f"{feature}_delta"] = (
                pd.to_numeric(out[feature], errors="coerce") - baseline
            ).astype(np.float32)

        n_eyes = groups.ngroups
        n_multi = int((groups[self.visit_key].transform("nunique") > 1).any())
        LOGGER.info(
            "longitudinal clinical features on %d eyes (%d visits); multi-visit eyes present: %s",
            n_eyes,
            out[self.visit_key].nunique(),
            bool(n_multi),
        )
        return out

    def _baseline_value(self, frame: pd.DataFrame, feature: str) -> pd.Series:
        """The eye's earliest-visit value of ``feature``, broadcast to its scans.

        Ties on ``order_key`` are impossible within an eye because the key is a
        visit ordinal, but a NaN at baseline propagates as NaN so the missingness
        indicator downstream can record it rather than an imputed zero.
        """
        values = pd.to_numeric(frame[feature], errors="coerce")
        ordered = frame[self.order_key].to_numpy()
        # idxmin over the visit ordinal picks the earliest row of each eye.
        helper = pd.DataFrame({"_order": ordered, "_value": values}, index=frame.index)
        first = helper.groupby([frame[k] for k in EYE_KEYS], sort=False)["_order"].transform("min")
        at_baseline = helper["_value"].where(helper["_order"] == first)
        return (
            at_baseline.groupby([frame[k] for k in EYE_KEYS], sort=False)
            .transform("first")
            .astype(np.float32)
        )


class ClinicalPerturbation:
    """The control ladder that separates clinical signal from visit identity.

    In OLIVES the pair ``(BCVA, CST)`` identifies the visit uniquely in 97-100%
    of cases in every fold. A fusion model can therefore reach a real improvement
    by memorising which visit it is looking at, which would not transfer to a new
    clinic and is not a clinical finding. Each mode below destroys one candidate
    explanation while leaving the others intact:

    ==========================  ===================================  ==========================
    mode                        what survives                        reading if score holds up
    ==========================  ===================================  ==========================
    ``none``                    everything                           reference arm
    ``patient_mean``            patient severity, no visit contrast  gain was patient-level
    ``within_patient_shuffle``  patient identity and marginals       gain was a fingerprint
    ``across_patient_shuffle``  nothing but the marginals            floor; anything here is a bug
    ``quantise``                clinical meaning, not exact values   gain is not exact-value recall
    ==========================  ===================================  ==========================

    ``quantise`` is both a diagnostic and a remedy: if the gain survives binning
    CST to 25 um and BCVA to 5 letters, it is a clinical effect and the binned
    features are the defensible way to ship it.

    Perturbation happens at visit granularity because BCVA and CST are constant
    within a visit, and it runs **before** longitudinal derivation so the deltas
    are computed from perturbed values.

    Args:
        mode: One of :attr:`MODES`.
        features: Clinical columns to perturb.
        seed: RNG seed; the shuffles are deterministic given it.
        bins: Bin width per feature for ``quantise``.
    """

    MODES = (
        "none",
        "patient_mean",
        "within_patient_shuffle",
        "across_patient_shuffle",
        "quantise",
    )
    DEFAULT_BINS = {"cst": 25.0, "bcva": 5.0}

    def __init__(
        self,
        mode: str = "none",
        features: tuple[str, ...] = ("bcva", "cst"),
        seed: int = 42,
        bins: dict[str, float] | None = None,
        visit_key: str = "visit_uid",
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown clinical perturbation {mode!r}; choose from {self.MODES}")
        self.mode = mode
        self.features = tuple(features)
        self.seed = seed
        self.bins = {**self.DEFAULT_BINS, **(bins or {})}
        self.visit_key = visit_key

    # ------------------------------------------------------------------
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``frame`` with the clinical columns perturbed."""
        if self.mode == "none":
            return frame
        missing = [c for c in (*EYE_KEYS, self.visit_key, *self.features) if c not in frame.columns]
        if missing:
            raise KeyError(f"clinical perturbation needs columns {missing}")

        out = frame.copy()
        handler = {
            "patient_mean": self._patient_mean,
            "within_patient_shuffle": self._within_patient_shuffle,
            "across_patient_shuffle": self._across_patient_shuffle,
            "quantise": self._quantise,
        }[self.mode]
        out = handler(out)
        LOGGER.warning(
            "clinical perturbation '%s' applied to %s; this is a CONTROL arm and its score "
            "is not a model result",
            self.mode,
            list(self.features),
        )
        return out

    # ------------------------------------------------------------------
    def _visit_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One row per visit, carrying the eye keys and clinical values."""
        return (
            frame.groupby([*EYE_KEYS, self.visit_key], sort=True)[list(self.features)]
            .first()
            .reset_index()
        )

    def _apply_visit_table(self, frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
        """Broadcast per-visit values back onto scans."""
        lookup = table.set_index(self.visit_key)[list(self.features)]
        for feature in self.features:
            frame[feature] = frame[self.visit_key].map(lookup[feature]).astype(np.float32)
        return frame

    def _patient_mean(self, frame: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            values = pd.to_numeric(frame[feature], errors="coerce")
            frame[feature] = values.groupby(frame["patient_id"]).transform("mean").astype(np.float32)
        return frame

    def _within_patient_shuffle(self, frame: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        table = self._visit_table(frame)
        for feature in self.features:
            table[feature] = (
                table.groupby("patient_id", sort=False)[feature]
                .transform(lambda s: s.to_numpy()[rng.permutation(len(s))])
            )
        return self._apply_visit_table(frame, table)

    def _across_patient_shuffle(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Permute whole visits' clinical values across all eyes.

        Permuting globally rather than within a partition is deliberate and safe:
        only the covariate moves, never a label, so a test patient receiving
        another eye's BCVA/CST learns nothing about its own biomarkers -- which
        is exactly the floor this arm is meant to establish.
        """
        rng = np.random.default_rng(self.seed)
        table = self._visit_table(frame)
        order = rng.permutation(len(table))
        for feature in self.features:
            table[feature] = table[feature].to_numpy()[order]
        return self._apply_visit_table(frame, table)

    def _quantise(self, frame: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            width = float(self.bins.get(feature, 0.0))
            if width <= 0:
                continue
            values = pd.to_numeric(frame[feature], errors="coerce")
            frame[feature] = (np.round(values / width) * width).astype(np.float32)
        return frame

    # ------------------------------------------------------------------
    def uniqueness_report(self, frame: pd.DataFrame) -> dict[str, float]:
        """How uniquely the clinical pair still identifies a visit or patient.

        Run this on a control arm to confirm the perturbation did what it claims:
        ``quantise`` should visibly reduce ``visits_uniquely_identified``.
        """
        pairs = frame.groupby(list(self.features), sort=False)
        return {
            "n_distinct_values": float(pairs.ngroups),
            "n_visits": float(frame[self.visit_key].nunique()),
            "n_patients": float(frame["patient_id"].nunique()),
            "visits_uniquely_identified": float((pairs[self.visit_key].nunique() == 1).mean()),
            "patients_uniquely_identified": float((pairs["patient_id"].nunique() == 1).mean()),
        }


@dataclass
class PreregisteredTargets:
    """Labels named in advance as where clinical context should help."""

    labels: list[str]
    rationale: str
    min_eyes: int
    alpha: float
    fit_patients: list[int] = field(default_factory=list)
    table: list[dict[str, Any]] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        """Write the registration so the analysis order is auditable."""
        return JsonIO.write(vars(self), path)


class WithinEyeAssociation:
    """Association between clinical change and biomarker change, within an eye.

    Both the clinical value and the label are centred within eye before
    correlating, so anything constant for a patient -- their identity included --
    contributes nothing. A non-zero correlation here is a longitudinal clinical
    effect, which is the only kind that a fusion model can exploit without
    memorising the cohort.

    The frame **must** be restricted to training patients. Choosing target labels
    from a table that has seen the test partition is selection on the outcome,
    and it is the first thing a reviewer will look for.

    Args:
        label_columns: Biomarker columns to test.
        features: Clinical columns to test against.
        visit_key: Column identifying a visit.
    """

    def __init__(
        self,
        label_columns: list[str],
        features: tuple[str, ...] = ("bcva", "cst"),
        visit_key: str = "visit_uid",
    ) -> None:
        self.label_columns = list(label_columns)
        self.features = tuple(features)
        self.visit_key = visit_key

    # ------------------------------------------------------------------
    def visit_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Collapse scans to one row per visit, labels by presence."""
        aggregation: dict[str, Any] = {f: (f, "mean") for f in self.features}
        aggregation.update({label: (label, "max") for label in self.label_columns})
        return (
            frame.groupby([*EYE_KEYS, self.visit_key], sort=True).agg(**aggregation).reset_index()
        )

    def for_training_fold(self, frame: pd.DataFrame, assignment: Any) -> pd.DataFrame:
        """Analyse using the fold's training patients only.

        Args:
            frame: Full modelling frame.
            assignment: A :class:`SplitAssignment` whose ``train`` list is used.
        """
        train_patients = set(getattr(assignment, "train"))
        if not train_patients:
            raise ValueError("assignment has no training patients")
        restricted = frame[frame["patient_id"].isin(train_patients)]
        held_out = set(frame["patient_id"]) - train_patients
        LOGGER.info(
            "within-eye association on %d training patients (%d held out and excluded)",
            restricted["patient_id"].nunique(),
            len(held_out),
        )
        return self.analyse(restricted)

    def analyse(self, frame: pd.DataFrame, min_visits: int = 4) -> pd.DataFrame:
        """Per-label within-eye correlations.

        Args:
            frame: Rows to analyse; restrict to the training fold first.
            min_visits: Skip labels with fewer usable visits than this.

        Returns:
            One row per label with the number of eyes in which the label
            changes, the within-eye correlation with each clinical feature, and
            its two-sided p-value.
        """
        from scipy.stats import pearsonr

        visits = self.visit_table(frame)
        rows: list[dict[str, Any]] = []
        for label in self.label_columns:
            varies = visits.groupby(list(EYE_KEYS), sort=False)[label].transform("nunique") > 1
            usable = visits[varies]
            record: dict[str, Any] = {
                "label": label,
                "eyes_with_change": int(usable.groupby(list(EYE_KEYS)).ngroups) if len(usable) else 0,
                "visits": int(len(usable)),
            }
            if len(usable) < min_visits:
                for feature in self.features:
                    record[f"r_{feature}"] = np.nan
                    record[f"p_{feature}"] = np.nan
                rows.append(record)
                continue

            centred_label = self._centre(usable, label)
            for feature in self.features:
                centred_feature = self._centre(usable, feature)
                if centred_feature.std() < 1e-12 or centred_label.std() < 1e-12:
                    record[f"r_{feature}"], record[f"p_{feature}"] = np.nan, np.nan
                    continue
                r, p = pearsonr(centred_feature, centred_label)
                record[f"r_{feature}"] = float(r)
                record[f"p_{feature}"] = float(p)
            rows.append(record)
        return pd.DataFrame(rows).sort_values("eyes_with_change", ascending=False)

    @staticmethod
    def _centre(frame: pd.DataFrame, column: str) -> np.ndarray:
        """Subtract the eye mean, so between-eye variation cannot contribute."""
        values = pd.to_numeric(frame[column], errors="coerce")
        centred = values - values.groupby([frame[k] for k in EYE_KEYS], sort=False).transform("mean")
        return centred.fillna(0.0).to_numpy(dtype=float)

    # ------------------------------------------------------------------
    def preregister(
        self,
        table: pd.DataFrame,
        oct_baseline: dict[str, float] | None = None,
        min_eyes: int = 10,
        alpha: float = 0.01,
        headroom_below: float = 0.70,
        feature: str = "cst",
        fit_patients: list[int] | None = None,
    ) -> PreregisteredTargets:
        """Name the labels where fusion is predicted to help, before running it.

        A label qualifies on two independent grounds: a within-eye clinical
        association strong enough to be a mechanism, and an OCT baseline low
        enough to leave room. Labels the image model already handles well are
        excluded even when their association is strong -- there is nothing there
        to win.

        Args:
            table: Output of :meth:`analyse` on training patients.
            oct_baseline: Per-label OCT AUPRC. Without it the headroom filter is
                skipped and every associated label qualifies.
            min_eyes: Minimum eyes in which the label must change.
            alpha: Two-sided significance threshold for the association.
            headroom_below: Exclude labels whose OCT AUPRC is at or above this.
            feature: Which clinical feature's association to test.
        """
        qualifying: list[str] = []
        for _, row in table.iterrows():
            p_value = row.get(f"p_{feature}", np.nan)
            if not np.isfinite(p_value) or p_value >= alpha:
                continue
            if row["eyes_with_change"] < min_eyes:
                continue
            if oct_baseline is not None:
                baseline = oct_baseline.get(row["label"])
                if baseline is None or baseline >= headroom_below:
                    continue
            qualifying.append(str(row["label"]))

        rationale = (
            f"within-eye |r| with {feature} significant at alpha={alpha} over at least "
            f"{min_eyes} eyes, and OCT AUPRC below {headroom_below}"
        )
        if oct_baseline is None:
            rationale = (
                f"within-eye |r| with {feature} significant at alpha={alpha} over at least "
                f"{min_eyes} eyes (no OCT baseline supplied, headroom filter skipped)"
            )
        LOGGER.info("pre-registered target labels: %s", qualifying or "none")
        return PreregisteredTargets(
            labels=qualifying,
            rationale=rationale,
            min_eyes=min_eyes,
            alpha=alpha,
            fit_patients=sorted(fit_patients or []),
            table=table.replace({np.nan: None}).to_dict(orient="records"),
        )
