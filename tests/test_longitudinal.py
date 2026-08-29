"""Within-eye clinical features, the control ladder, and pre-registration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olives_biomarkers.data.longitudinal import (
    ClinicalPerturbation,
    LongitudinalClinicalFeatures,
    WithinEyeAssociation,
)

LABELS = ["irf", "drt_me", "pavf"]


def _frame(
    n_patients: int = 6,
    visits_per_eye: int = 3,
    scans_per_visit: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """A small longitudinal frame with the columns the real manifest provides."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(n_patients):
        base_cst = 200.0 + 40.0 * patient  # strong between-patient variation
        for visit in range(visits_per_eye):
            cst = base_cst + 30.0 * visit
            bcva = 70.0 - 2.0 * visit
            for scan in range(scans_per_visit):
                rows.append(
                    {
                        "patient_id": patient,
                        "eye_id": float(patient),
                        "visit_uid": f"{patient}_E{patient}_V{visit}",
                        # deliberately non-contiguous, as visit inference leaves it
                        "visit_index": visit * 7,
                        "scan_number": scan,
                        "bcva": bcva,
                        "cst": cst,
                        "irf": float(visit > 0),          # changes within eye
                        "drt_me": float(patient % 2),     # constant within patient
                        "pavf": float(rng.integers(0, 2)),
                    }
                )
    return pd.DataFrame(rows)


class TestLongitudinalClinicalFeatures:
    """Derived contrast columns."""

    def test_adds_the_documented_columns(self) -> None:
        out = LongitudinalClinicalFeatures().transform(_frame())
        for column in LongitudinalClinicalFeatures().derived_names():
            assert column in out.columns

    def test_delta_is_measured_from_the_eye_baseline(self) -> None:
        out = LongitudinalClinicalFeatures().transform(_frame())
        assert np.allclose(out["cst"] - out["cst_baseline"], out["cst_delta"])

    def test_baseline_visit_has_zero_delta_and_the_flag(self) -> None:
        """The earliest visit has no history; its delta must be exactly zero."""
        out = LongitudinalClinicalFeatures().transform(_frame())
        baseline = out[out["is_baseline_visit"] == 1.0]
        assert len(baseline) > 0
        assert np.allclose(baseline["cst_delta"], 0.0)
        assert np.allclose(baseline["bcva_delta"], 0.0)
        assert (baseline["visit_order"] == 0).all()

    def test_baseline_is_the_earliest_visit_not_the_first_row(self) -> None:
        """Row order must not decide which visit counts as baseline."""
        frame = _frame()
        shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
        ordered = LongitudinalClinicalFeatures().transform(frame)
        out = LongitudinalClinicalFeatures().transform(shuffled)
        expected = dict(zip(ordered["visit_uid"], ordered["cst_delta"]))
        got = dict(zip(out["visit_uid"], out["cst_delta"]))
        assert got == pytest.approx(expected)

    def test_non_contiguous_visit_index_still_orders_correctly(self) -> None:
        out = LongitudinalClinicalFeatures().transform(_frame())
        per_visit = out.groupby("visit_uid")[["visit_order", "visit_index"]].first()
        assert set(per_visit["visit_order"]) == {0, 1, 2}
        assert per_visit.sort_values("visit_index")["visit_order"].is_monotonic_increasing

    def test_transform_is_idempotent(self) -> None:
        """A notebook re-running a cell must not build deltas of deltas."""
        transformer = LongitudinalClinicalFeatures()
        once = transformer.transform(_frame())
        twice = transformer.transform(once)
        assert np.allclose(once["cst_delta"], twice["cst_delta"])

    def test_single_visit_eye_gets_a_zero_delta(self) -> None:
        out = LongitudinalClinicalFeatures().transform(_frame(visits_per_eye=1))
        assert np.allclose(out["cst_delta"], 0.0)
        assert (out["is_baseline_visit"] == 1.0).all()

    def test_missing_visit_structure_is_reported(self) -> None:
        frame = _frame().drop(columns=["visit_uid"])
        with pytest.raises(KeyError, match="visit_uid"):
            LongitudinalClinicalFeatures().transform(frame)

    def test_missing_clinical_value_propagates_rather_than_becoming_zero(self) -> None:
        """A NaN baseline must stay NaN so the missingness indicator can fire."""
        frame = _frame()
        frame.loc[frame["visit_uid"] == "0_E0_V0", "cst"] = np.nan
        out = LongitudinalClinicalFeatures().transform(frame)
        patient_zero = out[out["patient_id"] == 0]
        assert patient_zero["cst_delta"].isna().all()


class TestClinicalPerturbation:
    """Each control must destroy exactly what it claims to."""

    def test_none_returns_the_frame_untouched(self) -> None:
        frame = _frame()
        assert ClinicalPerturbation(mode="none").transform(frame) is frame

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown clinical perturbation"):
            ClinicalPerturbation(mode="scramble")

    def test_patient_mean_removes_the_visit_contrast(self) -> None:
        out = ClinicalPerturbation(mode="patient_mean").transform(_frame())
        per_patient = out.groupby("patient_id")["cst"].nunique()
        assert (per_patient == 1).all()

    def test_patient_mean_keeps_between_patient_variation(self) -> None:
        """It must remove the visit contrast only, not all clinical signal."""
        out = ClinicalPerturbation(mode="patient_mean").transform(_frame())
        assert out.groupby("patient_id")["cst"].first().nunique() > 1

    def test_within_patient_shuffle_preserves_each_patient_values(self) -> None:
        frame = _frame()
        out = ClinicalPerturbation(mode="within_patient_shuffle", seed=1).transform(frame)
        for patient, group in out.groupby("patient_id"):
            before = sorted(frame[frame["patient_id"] == patient].groupby("visit_uid")["cst"].first())
            after = sorted(group.groupby("visit_uid")["cst"].first())
            assert before == pytest.approx(after)

    def test_within_patient_shuffle_keeps_the_fingerprint(self) -> None:
        """Visit identity must survive; only the value-to-label alignment breaks."""
        frame = _frame()
        perturbation = ClinicalPerturbation(mode="within_patient_shuffle", seed=1)
        before = perturbation.uniqueness_report(frame)
        after = perturbation.uniqueness_report(perturbation.transform(frame))
        assert after["visits_uniquely_identified"] == pytest.approx(
            before["visits_uniquely_identified"], abs=0.05
        )

    def test_across_patient_shuffle_preserves_the_global_multiset(self) -> None:
        frame = _frame()
        out = ClinicalPerturbation(mode="across_patient_shuffle", seed=3).transform(frame)
        before = sorted(frame.groupby("visit_uid")["cst"].first())
        after = sorted(out.groupby("visit_uid")["cst"].first())
        assert before == pytest.approx(after)

    def test_across_patient_shuffle_actually_moves_values(self) -> None:
        frame = _frame()
        out = ClinicalPerturbation(mode="across_patient_shuffle", seed=3).transform(frame)
        assert not np.allclose(frame["cst"].to_numpy(), out["cst"].to_numpy())

    def test_quantise_reduces_distinct_values(self) -> None:
        frame = _frame()
        perturbation = ClinicalPerturbation(mode="quantise", bins={"cst": 100.0, "bcva": 10.0})
        after = perturbation.uniqueness_report(perturbation.transform(frame))
        before = perturbation.uniqueness_report(frame)
        assert after["n_distinct_values"] < before["n_distinct_values"]

    def test_quantise_keeps_values_within_half_a_bin(self) -> None:
        frame = _frame()
        out = ClinicalPerturbation(mode="quantise", bins={"cst": 25.0}).transform(frame)
        assert (np.abs(out["cst"] - frame["cst"]) <= 12.5 + 1e-6).all()

    def test_shuffles_are_deterministic_given_the_seed(self) -> None:
        frame = _frame()
        a = ClinicalPerturbation(mode="across_patient_shuffle", seed=11).transform(frame)
        b = ClinicalPerturbation(mode="across_patient_shuffle", seed=11).transform(frame)
        assert np.allclose(a["cst"], b["cst"])

    def test_different_seeds_give_different_shuffles(self) -> None:
        frame = _frame(n_patients=12)
        a = ClinicalPerturbation(mode="across_patient_shuffle", seed=1).transform(frame)
        b = ClinicalPerturbation(mode="across_patient_shuffle", seed=2).transform(frame)
        assert not np.allclose(a["cst"], b["cst"])

    def test_perturbation_runs_before_deltas_are_derived(self) -> None:
        """A control arm must not keep honest deltas beside perturbed absolutes."""
        frame = _frame()
        perturbed = ClinicalPerturbation(mode="patient_mean").transform(frame)
        out = LongitudinalClinicalFeatures().transform(perturbed)
        assert np.allclose(out["cst_delta"], 0.0)


class TestWithinEyeAssociation:
    """The measurement that decides whether fusion has a mechanism at all."""

    def test_row_per_label(self) -> None:
        table = WithinEyeAssociation(LABELS).analyse(_frame())
        assert set(table["label"]) == set(LABELS)

    def test_detects_a_within_eye_association(self) -> None:
        """`irf` turns on at visit 1 exactly as CST rises, so r must be high.

        The ceiling here is sqrt(3)/2 = 0.866, not 1.0: CST rises linearly across
        the three visits while the label is a step, so the relationship is
        perfectly monotone but not perfectly linear.
        """
        table = WithinEyeAssociation(LABELS).analyse(_frame()).set_index("label")
        assert table.loc["irf", "r_cst"] == pytest.approx(np.sqrt(3) / 2, abs=0.01)
        assert table.loc["irf", "p_cst"] < 0.01

    def test_patient_constant_label_yields_no_association(self) -> None:
        """`drt_me` is perfectly predicted by patient identity and never changes.

        Between patients its correlation with CST is large; within eye there is
        nothing to correlate, which is exactly the confound being removed.
        """
        frame = _frame()
        between = frame[["cst", "drt_me"]].corr().iloc[0, 1]
        assert abs(between) > 0.1

        table = WithinEyeAssociation(LABELS).analyse(frame).set_index("label")
        assert table.loc["drt_me", "eyes_with_change"] == 0
        assert np.isnan(table.loc["drt_me", "r_cst"])

    def test_counts_only_eyes_where_the_label_changes(self) -> None:
        table = WithinEyeAssociation(LABELS).analyse(_frame(n_patients=6)).set_index("label")
        assert table.loc["irf", "eyes_with_change"] == 6

    def test_for_training_fold_excludes_held_out_patients(self) -> None:
        class _Assignment:
            train = [0, 1, 2]
            val = [3]
            test = [4, 5]

        analyser = WithinEyeAssociation(LABELS)
        table = analyser.for_training_fold(_frame(), _Assignment())
        # Six patients total, three in training: only those eyes may be counted.
        assert table.set_index("label").loc["irf", "eyes_with_change"] == 3

    def test_empty_training_set_is_rejected(self) -> None:
        class _Assignment:
            train: list[int] = []

        with pytest.raises(ValueError, match="no training patients"):
            WithinEyeAssociation(LABELS).for_training_fold(_frame(), _Assignment())


class TestPreregistration:
    """Targets must be named before the test partition is touched."""

    @pytest.fixture
    def table(self) -> pd.DataFrame:
        return WithinEyeAssociation(LABELS).analyse(_frame(n_patients=20))

    def test_significant_and_low_baseline_labels_qualify(self, table) -> None:
        registration = WithinEyeAssociation(LABELS).preregister(
            table, oct_baseline={"irf": 0.20, "drt_me": 0.5, "pavf": 0.5}, min_eyes=5
        )
        assert registration.labels == ["irf"]

    def test_a_label_the_image_model_already_handles_is_excluded(self) -> None:
        """A strong association is not an opportunity if OCT is already at 0.97."""
        table = WithinEyeAssociation(LABELS).analyse(_frame(n_patients=20))
        registration = WithinEyeAssociation(LABELS).preregister(
            table, oct_baseline={"irf": 0.97}, min_eyes=5, headroom_below=0.70
        )
        assert registration.labels == []

    def test_too_few_eyes_disqualifies(self, table) -> None:
        registration = WithinEyeAssociation(LABELS).preregister(
            table, oct_baseline={"irf": 0.20}, min_eyes=999
        )
        assert registration.labels == []

    def test_registration_records_the_patients_it_saw(self, table, tmp_path) -> None:
        registration = WithinEyeAssociation(LABELS).preregister(
            table, oct_baseline={"irf": 0.2}, min_eyes=5, fit_patients=[3, 1, 2]
        )
        assert registration.fit_patients == [1, 2, 3]
        path = registration.save(tmp_path / "targets.json")
        assert path.exists()

    def test_rationale_states_the_criteria(self, table) -> None:
        registration = WithinEyeAssociation(LABELS).preregister(
            table, oct_baseline={"irf": 0.2}, min_eyes=5, alpha=0.01
        )
        assert "0.01" in registration.rationale
        assert "OCT AUPRC" in registration.rationale
