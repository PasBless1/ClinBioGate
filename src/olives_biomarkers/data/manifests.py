"""Build and query the metadata-only sample manifest.

The OLIVES mirror stores image bytes inline in ~30 GB of parquet. Almost every
analysis we need (prevalence, missingness, grouping, splitting, leakage checks)
touches only the scalar columns, so the manifest strips the bytes out once and
everything downstream reads a few megabytes instead.

A manifest row is addressed by ``(shard_index, row_in_shard)``, which is a stable
handle the dataset class later uses to fetch the actual image.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from olives_biomarkers.data.schema import LabelSchema, SchemaValidationError
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.manifest")


@dataclass(frozen=True)
class ShardRef:
    """One parquet shard belonging to a split."""

    index: int
    path: Path
    split: str
    num_rows: int

    @property
    def name(self) -> str:
        return self.path.name


class ParquetShardIndex:
    """Discovers the parquet shards for one Hugging Face config."""

    def __init__(self, data_root: str | Path, config_name: str) -> None:
        self.data_root = Path(data_root)
        self.config_name = config_name
        self.config_dir = self.data_root / config_name

    def require_exists(self) -> None:
        """Raise an actionable error if the expected data directory is absent."""
        if not self.config_dir.is_dir():
            raise FileNotFoundError(
                "OLIVES parquet data not found.\n"
                f"  expected directory : {self.config_dir}\n"
                "  fix: download the Hugging Face dataset 'gOLIVES/OLIVES_Dataset' into that\n"
                "       location, or point data.root in configs/data.yaml at the right place."
            )

    def shards(self, split: str | None = None) -> list[ShardRef]:
        """List shards, optionally filtered to one split (``train`` / ``test``)."""
        self.require_exists()
        refs: list[ShardRef] = []
        for path in sorted(self.config_dir.glob("*.parquet")):
            shard_split = path.name.split("-")[0]
            if split is not None and shard_split != split:
                continue
            refs.append(
                ShardRef(
                    index=len(refs),
                    path=path,
                    split=shard_split,
                    num_rows=pq.ParquetFile(path).metadata.num_rows,
                )
            )
        if not refs:
            raise FileNotFoundError(f"no parquet shards for split={split!r} in {self.config_dir}")
        return refs

    def available_splits(self) -> list[str]:
        """Distinct split prefixes present on disk."""
        self.require_exists()
        return sorted({p.name.split("-")[0] for p in self.config_dir.glob("*.parquet")})

    def column_names(self) -> list[str]:
        """Column names of the first shard."""
        first = next(iter(sorted(self.config_dir.glob("*.parquet"))), None)
        if first is None:
            raise FileNotFoundError(f"no parquet shards in {self.config_dir}")
        return list(pq.ParquetFile(first).schema_arrow.names)

    def total_rows(self, split: str | None = None) -> int:
        """Sum of row counts across the selected shards."""
        return sum(s.num_rows for s in self.shards(split))


class VisitInferencer:
    """Infers visit blocks, which this data source does not label explicitly.

    A visit is one 49-B-scan OCT volume of one eye at one time point. This mirror
    carries no visit or week column, and ``Scan (n/49)`` is populated only on the
    biomarker-annotated visits, so neither can drive the segmentation. What *is*
    universal is that BCVA and CST are measured once per visit and therefore stay
    constant across a visit's 49 rows.

    The rule is: within one eye, in file order, open a new visit when the
    ``(bcva, cst)`` pair changes, or when the current visit already holds 49
    distinct scans. The size cap matters because consecutive visits occasionally
    record identical BCVA and CST, which the change-point alone would merge.

    Two duplication modes are handled separately:

    * **Adjacent row duplication** - the same image emitted twice in succession.
      Flagged as ``is_adjacent_duplicate`` and folded into the current visit.
    * **Repeated visits** - a later visit whose images are byte-identical to an
      earlier one (the dataset card cites patient 61, W8 vs W12). These open a new
      visit, because they are a distinct clinical time point.

    Visit indices are inferred, never authoritative; the audit reports how many
    blocks deviate from the expected 49 rows.
    """

    SCANS_PER_VOLUME = 49

    def __init__(self, scans_per_volume: int = SCANS_PER_VOLUME) -> None:
        self.scans_per_volume = scans_per_volume

    def assign(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add visit columns to a manifest.

        Args:
            frame: Manifest rows in original file order, carrying ``patient_id``,
                ``eye_id``, ``bcva``, ``cst`` and ideally ``image_hash``.

        Returns:
            A copy with ``visit_index``, ``visit_uid``, ``is_adjacent_duplicate``
            and ``visit_size`` added.
        """
        required = {"patient_id", "eye_id"}
        missing = required - set(frame.columns)
        if missing:
            raise SchemaValidationError(f"visit inference needs columns {sorted(missing)}")

        out = frame.copy()
        n = len(out)
        visit_index = np.zeros(n, dtype=np.int64)
        adjacent_dup = np.zeros(n, dtype=bool)

        hashes = out["image_hash"].to_numpy() if "image_hash" in out.columns else None
        bcva = out["bcva"].to_numpy() if "bcva" in out.columns else np.zeros(n)
        cst = out["cst"].to_numpy() if "cst" in out.columns else np.zeros(n)

        for _, positions in out.groupby(["patient_id", "eye_id"], dropna=False, sort=False).indices.items():
            ordered = np.sort(positions)
            current_visit = 0
            unique_in_visit = 0
            prev_key: tuple[Any, Any] | None = None
            prev_hash: Any = None

            for pos in ordered:
                img_hash = hashes[pos] if hashes is not None else None
                if prev_hash is not None and img_hash is not None and img_hash == prev_hash:
                    # Same image twice in a row: a duplicated record, not a new scan.
                    adjacent_dup[pos] = True
                    visit_index[pos] = current_visit
                    continue

                key = (bcva[pos], cst[pos])
                key_changed = prev_key is not None and not self._same_key(key, prev_key)
                if key_changed or unique_in_visit >= self.scans_per_volume:
                    current_visit += 1
                    unique_in_visit = 0

                visit_index[pos] = current_visit
                unique_in_visit += 1
                prev_key = key
                prev_hash = img_hash

        out["visit_index"] = visit_index
        out["is_adjacent_duplicate"] = adjacent_dup
        out["visit_uid"] = (
            out["patient_id"].astype("Int64").astype(str)
            + "_E"
            + out["eye_id"].astype("Int64").astype(str)
            + "_V"
            + out["visit_index"].astype(str)
        )
        out["visit_size"] = out.groupby("visit_uid")["visit_uid"].transform("size").astype(int)
        return out

    @staticmethod
    def _same_key(left: tuple[Any, Any], right: tuple[Any, Any]) -> bool:
        """Compare (bcva, cst) pairs treating NaN as equal to NaN."""
        for a, b in zip(left, right):
            a_nan = a is None or (isinstance(a, float) and np.isnan(a))
            b_nan = b is None or (isinstance(b, float) and np.isnan(b))
            if a_nan and b_nan:
                continue
            if a_nan != b_nan or a != b:
                return False
        return True


class DuplicateFlagger:
    """Groups byte-identical images and assigns stable duplicate group ids."""

    def flag(self, frame: pd.DataFrame, hash_column: str = "image_hash") -> pd.DataFrame:
        """Add ``dup_group_id``, ``dup_group_size`` and ``dup_rank`` columns.

        ``dup_rank == 0`` marks the first occurrence, which is what the
        ``keep_first`` duplicate policy retains.
        """
        if hash_column not in frame.columns:
            raise SchemaValidationError(
                f"duplicate flagging needs '{hash_column}'; rebuild the manifest with "
                "manifest.compute_image_hashes = true"
            )
        out = frame.copy()
        codes, _ = pd.factorize(out[hash_column], sort=False)
        out["dup_group_id"] = codes
        sizes = out.groupby("dup_group_id")["dup_group_id"].transform("size")
        out["dup_group_size"] = sizes.astype(int)
        out["dup_rank"] = out.groupby("dup_group_id").cumcount().astype(int)
        out["is_duplicate"] = out["dup_group_size"] > 1
        return out


class ManifestBuilder:
    """Streams parquet shards into a compact, image-free sample manifest."""

    def __init__(
        self,
        data_root: str | Path,
        schema: LabelSchema,
        compute_image_hashes: bool = True,
        hash_digest_size: int = 12,
    ) -> None:
        self.data_root = Path(data_root)
        self.schema = schema
        self.compute_image_hashes = compute_image_hashes
        self.hash_digest_size = hash_digest_size
        self.index = ParquetShardIndex(self.data_root, schema.config_name)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _columns_to_read(self, available: list[str]) -> list[str]:
        """Scalar columns plus the cheapest image projection that still works."""
        wanted = [c for c in self.schema.required_source_columns() if c in available]
        image_col = self.schema.image_column
        if self.compute_image_hashes:
            wanted.append(image_col)
        elif image_col in available:
            wanted.append(f"{image_col}.path")
        return wanted

    def _hash_bytes(self, payload: bytes) -> str:
        return hashlib.blake2b(payload, digest_size=self.hash_digest_size).hexdigest()

    def _shard_frame(self, shard: ShardRef, columns: list[str]) -> pd.DataFrame:
        """Read one shard and reduce it to manifest rows."""
        table = pq.read_table(shard.path, columns=columns)
        frame = table.to_pandas()

        image_col = self.schema.image_column
        if self.compute_image_hashes and image_col in frame.columns:
            images = frame.pop(image_col)
            frame["image_path"] = [rec["path"] for rec in images]
            frame["image_hash"] = [self._hash_bytes(rec["bytes"]) for rec in images]
        elif "path" in frame.columns:
            frame = frame.rename(columns={"path": "image_path"})

        frame = frame.rename(columns=self.schema.rename_map())
        frame.insert(0, "row_in_shard", np.arange(len(frame), dtype=np.int64))
        frame.insert(0, "shard_index", shard.index)
        frame.insert(0, "shard_name", shard.name)
        frame.insert(0, "split_source", shard.split)
        return frame

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def iter_shard_frames(self, split: str | None = "train") -> Iterator[tuple[ShardRef, pd.DataFrame]]:
        """Yield ``(shard, frame)`` pairs one shard at a time (bounded memory)."""
        available = self.index.column_names()
        self.schema.validate_columns(available, include_labels=False)
        columns = self._columns_to_read(available)
        for shard in self.index.shards(split):
            yield shard, self._shard_frame(shard, columns)

    def build(self, split: str | None = "train", progress: bool = True) -> Manifest:
        """Build the full manifest for one split.

        Args:
            split: ``"train"``, ``"test"``, or None for every shard.
            progress: Log per-shard progress (the hashing pass takes ~90 s).

        Returns:
            A :class:`Manifest` with visit, duplicate and derived columns filled in.
        """
        frames: list[pd.DataFrame] = []
        shards = self.index.shards(split)
        LOGGER.info(
            "Building manifest: config=%s split=%s shards=%d hashes=%s",
            self.schema.config_name,
            split,
            len(shards),
            self.compute_image_hashes,
        )
        for shard, frame in self.iter_shard_frames(split):
            frames.append(frame)
            if progress:
                LOGGER.info("  shard %2d/%d %-32s rows=%d", shard.index + 1, len(shards), shard.name, len(frame))

        combined = pd.concat(frames, ignore_index=True)
        combined.insert(0, "row_uid", np.arange(len(combined), dtype=np.int64))
        combined = self._add_derived_columns(combined)

        if self.compute_image_hashes:
            combined = DuplicateFlagger().flag(combined)
        if "scan_number" in combined.columns:
            combined = VisitInferencer().assign(combined)

        LOGGER.info("Manifest built: %d rows x %d columns", len(combined), combined.shape[1])
        return Manifest(combined, schema=self.schema, split=split)

    def _add_derived_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add missingness flags, disease names and the biomarker-labelled flag."""
        out = frame.copy()
        for clinical in ("bcva", "cst"):
            if clinical in out.columns:
                out[f"{clinical}_missing"] = out[clinical].isna()

        if "disease_label" in out.columns:
            out["disease_name"] = out["disease_label"].map(self.schema.disease_label_map)

        label_keys = [k for k in self.schema.label_keys if k in out.columns]
        if label_keys:
            out["has_biomarkers"] = out[label_keys].notna().all(axis=1)
            out["n_positive_labels"] = out[label_keys].fillna(0).sum(axis=1).astype("Int64")
            out.loc[~out["has_biomarkers"], "n_positive_labels"] = pd.NA
        else:
            out["has_biomarkers"] = False
        return out


class Manifest:
    """A sample manifest with query helpers, save/load and integrity checks."""

    def __init__(self, frame: pd.DataFrame, schema: LabelSchema, split: str | None = None) -> None:
        self.frame = frame
        self.schema = schema
        self.split = split

    # -- dunder ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:
        return (
            f"Manifest(rows={len(self.frame)}, labels={self.schema.n_labels}, "
            f"target_set={self.schema.target_set!r}, split={self.split!r})"
        )

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the manifest to parquet (no image bytes, safe to keep locally)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(out, index=False)
        LOGGER.info("Manifest saved: %s (%.1f KB)", out, out.stat().st_size / 1024)
        return out

    @classmethod
    def load(cls, path: str | Path, schema: LabelSchema, split: str | None = None) -> Manifest:
        """Read a manifest previously written by :meth:`save`."""
        frame = pd.read_parquet(path)
        return cls(frame, schema=schema, split=split)

    # -- views -------------------------------------------------------------
    @property
    def label_columns(self) -> list[str]:
        """Label columns present in the frame, in schema order."""
        return [k for k in self.schema.label_keys if k in self.frame.columns]

    def labelled(self) -> pd.DataFrame:
        """Rows that carry a complete biomarker vector."""
        return self.frame[self.frame["has_biomarkers"]].copy()

    def deduplicated(self, policy: str = "keep_first") -> pd.DataFrame:
        """Apply a duplicate policy and return the surviving rows."""
        if policy == "keep_all":
            return self.frame.copy()
        if "dup_rank" not in self.frame.columns:
            raise SchemaValidationError("no duplicate columns; rebuild manifest with hashes enabled")
        if policy == "keep_first":
            return self.frame[self.frame["dup_rank"] == 0].copy()
        if policy == "drop_all":
            return self.frame[~self.frame["is_duplicate"]].copy()
        raise ValueError(f"unknown duplicate policy {policy!r}")

    def modelling_frame(self, policy: str = "keep_first", labelled_only: bool = True) -> pd.DataFrame:
        """The rows a model actually trains on: deduplicated and label-complete."""
        frame = self.deduplicated(policy)
        if labelled_only:
            frame = frame[frame["has_biomarkers"]]
        return frame.reset_index(drop=True)

    def label_matrix(self, frame: pd.DataFrame | None = None) -> np.ndarray:
        """Return the ``(n_samples, n_labels)`` float array of targets."""
        source = self.frame if frame is None else frame
        return source[self.label_columns].to_numpy(dtype=np.float32)

    # -- summaries ---------------------------------------------------------
    def patients(self) -> np.ndarray:
        """Unique patient ids present."""
        return np.sort(self.frame["patient_id"].dropna().unique())

    def label_prevalence(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """Positive counts and rates per label over label-complete rows."""
        source = self.labelled() if frame is None else frame
        rows = []
        for key in self.label_columns:
            values = source[key].dropna()
            positives = int((values == 1).sum())
            n = int(len(values))
            rows.append(
                {
                    "label": key,
                    "display": self.schema.key_to_display.get(key, key),
                    "n_labelled": n,
                    "n_positive": positives,
                    "prevalence": positives / n if n else np.nan,
                    "n_patients_positive": int(
                        source.loc[source[key] == 1, "patient_id"].nunique()
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values("prevalence", ascending=False).reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        """Compact dictionary summary used by the audit report."""
        frame = self.frame
        summary: dict[str, Any] = {
            "n_rows": int(len(frame)),
            "n_patients": int(frame["patient_id"].nunique()),
            "n_eyes": int(frame.groupby(["patient_id", "eye_id"], dropna=False).ngroups),
            "n_labelled_rows": int(frame["has_biomarkers"].sum()),
            "target_set": self.schema.target_set,
            "config_name": self.schema.config_name,
            "split": self.split,
        }
        if "visit_uid" in frame.columns:
            summary["n_visits_inferred"] = int(frame["visit_uid"].nunique())
        if "is_duplicate" in frame.columns:
            summary["n_duplicate_rows"] = int(frame["is_duplicate"].sum())
            summary["n_duplicate_groups"] = int(
                frame.loc[frame["is_duplicate"], "dup_group_id"].nunique()
            )
        for clinical in ("bcva", "cst"):
            col = f"{clinical}_missing"
            if col in frame.columns:
                summary[f"n_{clinical}_missing"] = int(frame[col].sum())
        if "disease_name" in frame.columns:
            summary["disease_counts"] = frame["disease_name"].value_counts().to_dict()
        return summary
