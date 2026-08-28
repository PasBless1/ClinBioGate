"""Leakage-safe, patient-grouped partitioning.

The dominant risk in OLIVES is not model choice, it is leakage: 49 B-scans per
volume, repeated visits, two eyes per patient, and byte-identical duplicate
images. Every splitter here partitions **patients**, never scans, and every
split is verified before it is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from olives_biomarkers.utils.io import JsonIO
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.splits")


class LeakageError(RuntimeError):
    """Raised when a split would place one patient or duplicate group in two partitions."""


@dataclass
class SplitAssignment:
    """Patient ids assigned to each partition of one split."""

    name: str
    train: list[int] = field(default_factory=list)
    val: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)
    calibration: list[int] = field(default_factory=list)
    seed: int = 42

    @property
    def partitions(self) -> dict[str, list[int]]:
        """All non-empty partitions keyed by name."""
        out = {"train": self.train, "val": self.val, "test": self.test}
        if self.calibration:
            out["calibration"] = self.calibration
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "partitions": {k: sorted(int(p) for p in v) for k, v in self.partitions.items()},
            "n_patients": {k: len(v) for k, v in self.partitions.items()},
        }

    def partition_of(self, patient_id: int) -> str | None:
        """Which partition a patient belongs to, or None."""
        for name, patients in self.partitions.items():
            if patient_id in patients:
                return name
        return None

    def label_rows(self, frame: pd.DataFrame, group_key: str = "patient_id") -> pd.Series:
        """Map every manifest row to its partition name."""
        lookup: dict[int, str] = {}
        for name, patients in self.partitions.items():
            for patient in patients:
                lookup[patient] = name
        return frame[group_key].map(lookup)


class SplitValidator:
    """Independent verification that a split is leakage-free.

    Deliberately separate from the splitters so the tests can construct a
    deliberately-leaky split and confirm this raises.
    """

    def __init__(self, group_key: str = "patient_id") -> None:
        self.group_key = group_key

    def check_group_disjoint(self, assignment: SplitAssignment) -> None:
        """Every patient appears in at most one partition."""
        seen: dict[int, str] = {}
        for name, patients in assignment.partitions.items():
            for patient in patients:
                if patient in seen:
                    raise LeakageError(
                        f"patient {patient} appears in both '{seen[patient]}' and '{name}'"
                    )
                seen[patient] = name

    def check_non_empty(self, assignment: SplitAssignment) -> None:
        """No partition is empty."""
        for name, patients in assignment.partitions.items():
            if not patients:
                raise LeakageError(f"partition '{name}' is empty")

    def check_duplicates_contained(
        self, frame: pd.DataFrame, assignment: SplitAssignment
    ) -> None:
        """No duplicate-image group straddles two partitions."""
        if "dup_group_id" not in frame.columns:
            return
        partition = assignment.label_rows(frame, self.group_key)
        scoped = frame.assign(_partition=partition).dropna(subset=["_partition"])
        spread = scoped.groupby("dup_group_id")["_partition"].nunique()
        offenders = spread[spread > 1]
        if len(offenders):
            raise LeakageError(
                f"{len(offenders)} duplicate-image groups span multiple partitions "
                f"(e.g. group {offenders.index[0]})"
            )

    def check_coverage(self, frame: pd.DataFrame, assignment: SplitAssignment) -> dict[str, int]:
        """Report how many rows land in each partition and how many are unassigned."""
        partition = assignment.label_rows(frame, self.group_key)
        counts = partition.value_counts(dropna=False).to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def validate(self, frame: pd.DataFrame, assignment: SplitAssignment) -> dict[str, Any]:
        """Run every check; raises :class:`LeakageError` on the first failure."""
        self.check_group_disjoint(assignment)
        self.check_non_empty(assignment)
        self.check_duplicates_contained(frame, assignment)
        return {
            "assignment": assignment.to_dict(),
            "row_counts": self.check_coverage(frame, assignment),
        }


class PatientGroupedSplitter:
    """Single grouped train / val / test holdout, with an inner calibration split.

    Patients are stratified on a patient-level attribute (disease label by
    default) so partitions keep a comparable DR/DME mix, then allocated
    largest-stratum-first so small strata still reach every partition.
    """

    def __init__(
        self,
        train_fraction: float = 0.70,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        calibration_fraction_of_train: float = 0.15,
        stratify_on: str | None = "disease_label",
        group_key: str = "patient_id",
        seed: int = 42,
    ) -> None:
        total = train_fraction + val_fraction + test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1.0, got {total}")
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.calibration_fraction_of_train = calibration_fraction_of_train
        self.stratify_on = stratify_on
        self.group_key = group_key
        self.seed = seed

    def _patient_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One row per patient carrying the stratification attribute."""
        columns = [self.group_key]
        if self.stratify_on and self.stratify_on in frame.columns:
            columns.append(self.stratify_on)
        table = frame[columns].drop_duplicates(subset=[self.group_key]).reset_index(drop=True)
        if self.stratify_on and self.stratify_on in table.columns:
            table["_stratum"] = table[self.stratify_on].astype(str)
        else:
            table["_stratum"] = "all"
        return table

    @staticmethod
    def _allocate(patients: np.ndarray, fractions: dict[str, float], rng: np.random.Generator) -> dict[str, list[int]]:
        """Shuffle then slice one stratum into the requested fractions.

        Uses largest-remainder apportionment, then guarantees every partition at
        least one patient when the stratum is big enough to allow it. Handing the
        remainder to the largest partition instead would starve val and test in
        small strata: with four patients and 70/15/15, naive rounding gives
        train=3, val=1, **test=0**.
        """
        shuffled = patients.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        names = list(fractions)
        out: dict[str, list[int]] = {name: [] for name in names}
        if n == 0:
            return out

        exact = {name: fractions[name] * n for name in names}
        counts = {name: int(np.floor(exact[name])) for name in names}

        # Largest-remainder: the partitions whose exact share was cut most get the
        # leftover seats, which keeps the realised fractions closest to requested.
        remainder = n - sum(counts.values())
        by_fraction_lost = sorted(names, key=lambda k: exact[k] - counts[k], reverse=True)
        for name in by_fraction_lost[:remainder]:
            counts[name] += 1

        # Guarantee coverage: an empty val or test partition is unusable, so take
        # from the largest partition that can spare a patient.
        if n >= len(names):
            for name in names:
                if counts[name] > 0:
                    continue
                donor = max(names, key=lambda k: counts[k])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[name] += 1

        cursor = 0
        for name in names:
            out[name] = shuffled[cursor : cursor + counts[name]].tolist()
            cursor += counts[name]
        return out

    def split(self, frame: pd.DataFrame, name: str = "holdout") -> SplitAssignment:
        """Produce a validated train/val/test/calibration assignment."""
        rng = np.random.default_rng(self.seed)
        table = self._patient_table(frame)

        fractions = {
            "train": self.train_fraction,
            "val": self.val_fraction,
            "test": self.test_fraction,
        }
        buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        for stratum in sorted(table["_stratum"].unique()):
            patients = table.loc[table["_stratum"] == stratum, self.group_key].to_numpy()
            allocated = self._allocate(patients, fractions, rng)
            for key, values in allocated.items():
                buckets[key].extend(values)

        calibration: list[int] = []
        if self.calibration_fraction_of_train > 0 and buckets["train"]:
            train_array = np.array(buckets["train"])
            rng.shuffle(train_array)
            n_cal = max(1, int(round(self.calibration_fraction_of_train * len(train_array))))
            n_cal = min(n_cal, len(train_array) - 1)
            calibration = train_array[:n_cal].tolist()
            buckets["train"] = train_array[n_cal:].tolist()

        assignment = SplitAssignment(
            name=name,
            train=buckets["train"],
            val=buckets["val"],
            test=buckets["test"],
            calibration=calibration,
            seed=self.seed,
        )
        SplitValidator(self.group_key).validate(frame, assignment)
        LOGGER.info(
            "holdout split '%s': train=%d val=%d test=%d calibration=%d patients",
            name,
            len(assignment.train),
            len(assignment.val),
            len(assignment.test),
            len(assignment.calibration),
        )
        return assignment


class GroupedCrossValidator:
    """Patient-level k-fold cross-validation with an inner val/calibration split."""

    def __init__(
        self,
        n_folds: int = 5,
        val_fraction_of_train: float = 0.15,
        calibration_fraction_of_train: float = 0.15,
        stratify_on: str | None = "disease_label",
        group_key: str = "patient_id",
        seed: int = 42,
    ) -> None:
        self.n_folds = n_folds
        self.val_fraction_of_train = val_fraction_of_train
        self.calibration_fraction_of_train = calibration_fraction_of_train
        self.stratify_on = stratify_on
        self.group_key = group_key
        self.seed = seed

    def split(self, frame: pd.DataFrame) -> list[SplitAssignment]:
        """Return one :class:`SplitAssignment` per fold."""
        rng = np.random.default_rng(self.seed)
        splitter = PatientGroupedSplitter(
            stratify_on=self.stratify_on, group_key=self.group_key, seed=self.seed
        )
        table = splitter._patient_table(frame)

        # Round-robin patients within each stratum across folds so fold-wise
        # disease balance stays comparable.
        fold_of: dict[int, int] = {}
        for stratum in sorted(table["_stratum"].unique()):
            patients = table.loc[table["_stratum"] == stratum, self.group_key].to_numpy()
            rng.shuffle(patients)
            for position, patient in enumerate(patients):
                fold_of[patient] = position % self.n_folds

        assignments: list[SplitAssignment] = []
        for fold in range(self.n_folds):
            test = [p for p, f in fold_of.items() if f == fold]
            remaining = np.array([p for p, f in fold_of.items() if f != fold])
            fold_rng = np.random.default_rng(self.seed + fold)
            fold_rng.shuffle(remaining)

            n_val = max(1, int(round(self.val_fraction_of_train * len(remaining))))
            n_cal = max(1, int(round(self.calibration_fraction_of_train * len(remaining))))
            val = remaining[:n_val].tolist()
            calibration = remaining[n_val : n_val + n_cal].tolist()
            train = remaining[n_val + n_cal :].tolist()

            assignment = SplitAssignment(
                name=f"fold_{fold}",
                train=train,
                val=val,
                test=test,
                calibration=calibration,
                seed=self.seed + fold,
            )
            SplitValidator(self.group_key).validate(frame, assignment)
            assignments.append(assignment)

        LOGGER.info("built %d patient-grouped folds", len(assignments))
        return assignments

    def iter_folds(self, frame: pd.DataFrame) -> Iterator[SplitAssignment]:
        """Iterate folds lazily."""
        yield from self.split(frame)


class SplitManifestWriter:
    """Persists split assignments and their prevalence summaries."""

    def __init__(self, output_dir: str | Path, group_key: str = "patient_id") -> None:
        self.output_dir = Path(output_dir)
        self.group_key = group_key

    def prevalence_by_partition(
        self, frame: pd.DataFrame, assignment: SplitAssignment, label_columns: list[str]
    ) -> pd.DataFrame:
        """Per-partition label prevalence, used to document stratification limits."""
        partition = assignment.label_rows(frame, self.group_key)
        scoped = frame.assign(partition=partition).dropna(subset=["partition"])
        rows = []
        for name, group in scoped.groupby("partition"):
            labelled = group[group["has_biomarkers"]] if "has_biomarkers" in group else group
            record: dict[str, Any] = {
                "partition": name,
                "n_patients": int(group[self.group_key].nunique()),
                "n_rows": int(len(group)),
                "n_labelled_rows": int(len(labelled)),
            }
            for column in label_columns:
                values = labelled[column].dropna()
                record[column] = round(float((values == 1).mean()), 4) if len(values) else np.nan
            rows.append(record)
        return pd.DataFrame(rows)

    def write(
        self,
        assignment: SplitAssignment,
        frame: pd.DataFrame,
        label_columns: list[str] | None = None,
    ) -> dict[str, Path]:
        """Write the JSON assignment, a per-row CSV and the prevalence table."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        paths["json"] = JsonIO.write(
            assignment.to_dict(), self.output_dir / f"split_{assignment.name}.json"
        )

        partition = assignment.label_rows(frame, self.group_key)
        rows = frame.loc[partition.notna(), [self.group_key]].copy()
        rows["partition"] = partition[partition.notna()]
        row_csv = self.output_dir / f"split_{assignment.name}_patients.csv"
        rows.drop_duplicates().sort_values([self.group_key]).to_csv(row_csv, index=False)
        paths["patients_csv"] = row_csv

        if label_columns:
            prevalence = self.prevalence_by_partition(frame, assignment, label_columns)
            prevalence_csv = self.output_dir / f"split_{assignment.name}_prevalence.csv"
            prevalence.to_csv(prevalence_csv, index=False)
            paths["prevalence_csv"] = prevalence_csv

        LOGGER.info("split manifests written to %s", self.output_dir)
        return paths

    @staticmethod
    def load(path: str | Path) -> SplitAssignment:
        """Read back a split assignment written by :meth:`write`."""
        payload = JsonIO.read(path)
        partitions = payload["partitions"]
        return SplitAssignment(
            name=payload["name"],
            train=partitions.get("train", []),
            val=partitions.get("val", []),
            test=partitions.get("test", []),
            calibration=partitions.get("calibration", []),
            seed=payload.get("seed", 42),
        )
