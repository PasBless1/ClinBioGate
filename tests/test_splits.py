"""Leakage tests.

The most important tests in the suite. Several deliberately construct a leaky
split and assert that the validator rejects it -- a leakage check that cannot
fail is not a check.
"""

from __future__ import annotations

import pandas as pd
import pytest

from olives_biomarkers.data.splits import (
    GroupedCrossValidator,
    LeakageError,
    PatientGroupedSplitter,
    SplitAssignment,
    SplitManifestWriter,
    SplitValidator,
)


class TestPatientGroupedSplitter:
    """Holdout splitting."""

    def test_partitions_are_patient_disjoint(self, modelling_frame: pd.DataFrame) -> None:
        assignment = PatientGroupedSplitter(seed=42).split(modelling_frame)
        seen: set[int] = set()
        for patients in assignment.partitions.values():
            assert not (seen & set(patients)), "a patient appears in two partitions"
            seen |= set(patients)

    def test_every_partition_is_non_empty(self, modelling_frame: pd.DataFrame) -> None:
        assignment = PatientGroupedSplitter(seed=42).split(modelling_frame)
        for name, patients in assignment.partitions.items():
            assert patients, f"partition {name} is empty"

    def test_deterministic_for_a_fixed_seed(self, modelling_frame: pd.DataFrame) -> None:
        first = PatientGroupedSplitter(seed=7).split(modelling_frame)
        second = PatientGroupedSplitter(seed=7).split(modelling_frame)
        assert sorted(first.train) == sorted(second.train)
        assert sorted(first.test) == sorted(second.test)

    def test_different_seeds_give_different_splits(self, modelling_frame: pd.DataFrame) -> None:
        first = PatientGroupedSplitter(seed=1).split(modelling_frame)
        second = PatientGroupedSplitter(seed=99).split(modelling_frame)
        assert sorted(first.test) != sorted(second.test)

    def test_no_scan_of_a_test_patient_reaches_training(
        self, modelling_frame: pd.DataFrame
    ) -> None:
        assignment = PatientGroupedSplitter(seed=42).split(modelling_frame)
        train_rows = modelling_frame[modelling_frame["patient_id"].isin(assignment.train)]
        assert not set(train_rows["patient_id"]) & set(assignment.test)

    def test_both_eyes_of_a_patient_stay_together(self, modelling_frame: pd.DataFrame) -> None:
        assignment = PatientGroupedSplitter(seed=42).split(modelling_frame)
        partition = assignment.label_rows(modelling_frame)
        scoped = modelling_frame.assign(partition=partition).dropna(subset=["partition"])
        per_patient = scoped.groupby("patient_id")["partition"].nunique()
        assert (per_patient == 1).all(), "a patient's eyes were split across partitions"

    def test_calibration_is_carved_out_of_train(self, modelling_frame: pd.DataFrame) -> None:
        assignment = PatientGroupedSplitter(
            calibration_fraction_of_train=0.2, seed=42
        ).split(modelling_frame)
        assert assignment.calibration
        assert not set(assignment.calibration) & set(assignment.train)
        assert not set(assignment.calibration) & set(assignment.test)

    def test_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            PatientGroupedSplitter(train_fraction=0.8, val_fraction=0.3, test_fraction=0.3)


class TestSplitValidatorRejectsLeakage:
    """The validator must actually fail on bad splits."""

    def test_rejects_a_patient_in_two_partitions(self, modelling_frame: pd.DataFrame) -> None:
        patients = modelling_frame["patient_id"].unique().tolist()
        leaky = SplitAssignment(
            name="leaky", train=[patients[0], patients[1]], val=[patients[0]], test=[patients[2]]
        )
        with pytest.raises(LeakageError, match="appears in both"):
            SplitValidator().check_group_disjoint(leaky)

    def test_rejects_an_empty_partition(self, modelling_frame: pd.DataFrame) -> None:
        patients = modelling_frame["patient_id"].unique().tolist()
        empty = SplitAssignment(name="empty", train=[patients[0]], val=[], test=[patients[1]])
        with pytest.raises(LeakageError, match="is empty"):
            SplitValidator().check_non_empty(empty)

    def test_rejects_a_duplicate_group_spanning_partitions(self) -> None:
        # Two byte-identical images that belong to different patients.
        frame = pd.DataFrame(
            {
                "patient_id": [1, 2, 1, 2],
                "dup_group_id": [10, 10, 11, 12],
                "row_uid": [0, 1, 2, 3],
            }
        )
        assignment = SplitAssignment(name="x", train=[1], val=[2], test=[2])
        # patient 2 in two partitions would trip the first check, so test the
        # duplicate check in isolation with a clean assignment.
        assignment = SplitAssignment(name="x", train=[1], val=[2], test=[])
        with pytest.raises(LeakageError, match="duplicate-image groups span"):
            SplitValidator().check_duplicates_contained(frame, assignment)

    def test_accepts_a_contained_duplicate_group(self) -> None:
        frame = pd.DataFrame(
            {"patient_id": [1, 1, 2], "dup_group_id": [10, 10, 11], "row_uid": [0, 1, 2]}
        )
        assignment = SplitAssignment(name="ok", train=[1], val=[2], test=[])
        SplitValidator().check_duplicates_contained(frame, assignment)

    def test_real_splitter_output_passes_full_validation(
        self, modelling_frame: pd.DataFrame
    ) -> None:
        assignment = PatientGroupedSplitter(seed=42).split(modelling_frame)
        result = SplitValidator().validate(modelling_frame, assignment)
        assert result["assignment"]["name"] == "holdout"


class TestGroupedCrossValidator:
    """k-fold at patient level."""

    def test_produces_the_requested_number_of_folds(self, modelling_frame: pd.DataFrame) -> None:
        folds = GroupedCrossValidator(n_folds=3, seed=42).split(modelling_frame)
        assert len(folds) == 3

    def test_every_fold_is_internally_disjoint(self, modelling_frame: pd.DataFrame) -> None:
        for fold in GroupedCrossValidator(n_folds=3, seed=42).split(modelling_frame):
            SplitValidator().validate(modelling_frame, fold)

    def test_test_partitions_cover_patients_without_overlap(
        self, modelling_frame: pd.DataFrame
    ) -> None:
        folds = GroupedCrossValidator(n_folds=3, seed=42).split(modelling_frame)
        collected: list[int] = []
        for fold in folds:
            collected.extend(fold.test)
        assert len(collected) == len(set(collected)), "a patient is tested in two folds"
        assert set(collected) == set(modelling_frame["patient_id"].unique())


class TestSplitManifestWriter:
    """Persistence and round-tripping."""

    def test_round_trips_through_json(self, tmp_path, modelling_frame: pd.DataFrame) -> None:
        assignment = PatientGroupedSplitter(seed=42).split(modelling_frame)
        writer = SplitManifestWriter(tmp_path)
        paths = writer.write(assignment, modelling_frame)
        restored = SplitManifestWriter.load(paths["json"])
        assert sorted(restored.train) == sorted(assignment.train)
        assert sorted(restored.test) == sorted(assignment.test)

    def test_prevalence_table_covers_every_partition(
        self, tmp_path, manifest, modelling_frame: pd.DataFrame
    ) -> None:
        assignment = PatientGroupedSplitter(seed=42).split(modelling_frame)
        table = SplitManifestWriter(tmp_path).prevalence_by_partition(
            modelling_frame, assignment, manifest.label_columns
        )
        assert set(table["partition"]) == set(assignment.partitions)
        assert (table["n_patients"] > 0).all()
