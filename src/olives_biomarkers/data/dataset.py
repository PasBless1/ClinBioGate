"""Torch datasets over the OLIVES parquet mirror.

Images live inline in ~30 GB of parquet, which is fine for a local scan but
impractical to random-access during training and impossible to upload to Colab.
Two access paths are therefore provided:

* :class:`ParquetImageReader` - random access straight into the shards, with
  row-group caching. Good for inspection and small jobs.
* :class:`ImageCacheExporter` - writes just the modelling subset (9,396 PNGs,
  2.3 GB) to a folder once. That folder is what you upload to Drive and train
  from on Colab.

:class:`OlivesDataset` reads from whichever source is available.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from olives_biomarkers.data.manifests import ParquetShardIndex
from olives_biomarkers.data.preprocessing import ClinicalPreprocessor
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.dataset")


class ParquetImageReader:
    """Random access to image bytes inside the parquet shards.

    Caches the most recently used row group per shard, so iterating a manifest in
    ``(shard_index, row_in_shard)`` order costs roughly one decode per row group
    rather than one file read per row.
    """

    def __init__(
        self,
        data_root: str | Path,
        config_name: str,
        split: str | None = "train",
        image_column: str = "Image",
        cache_size: int = 2,
    ) -> None:
        self.index = ParquetShardIndex(data_root, config_name)
        self.shards = {s.index: s for s in self.index.shards(split)}
        self.image_column = image_column
        self.cache_size = cache_size
        self._files: dict[int, pq.ParquetFile] = {}
        self._row_group_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self._cache_order: list[tuple[int, int]] = []

    def _parquet_file(self, shard_index: int) -> pq.ParquetFile:
        if shard_index not in self._files:
            self._files[shard_index] = pq.ParquetFile(self.shards[shard_index].path)
        return self._files[shard_index]

    def _locate(self, shard_index: int, row_in_shard: int) -> tuple[int, int]:
        """Map a shard row index to ``(row_group, offset_within_group)``."""
        parquet = self._parquet_file(shard_index)
        remaining = row_in_shard
        for group in range(parquet.num_row_groups):
            rows = parquet.metadata.row_group(group).num_rows
            if remaining < rows:
                return group, remaining
            remaining -= rows
        raise IndexError(f"row {row_in_shard} out of range for shard {shard_index}")

    def _row_group(self, shard_index: int, row_group: int) -> list[dict[str, Any]]:
        key = (shard_index, row_group)
        if key in self._row_group_cache:
            return self._row_group_cache[key]
        table = self._parquet_file(shard_index).read_row_group(
            row_group, columns=[self.image_column]
        )
        rows = table.column(self.image_column).to_pylist()
        self._row_group_cache[key] = rows
        self._cache_order.append(key)
        while len(self._cache_order) > self.cache_size:
            self._row_group_cache.pop(self._cache_order.pop(0), None)
        return rows

    def read_bytes(self, shard_index: int, row_in_shard: int) -> bytes:
        """Return the raw encoded image bytes for one manifest row."""
        group, offset = self._locate(shard_index, row_in_shard)
        return self._row_group(shard_index, group)[offset]["bytes"]

    def read_image(self, shard_index: int, row_in_shard: int) -> Image.Image:
        """Return the decoded PIL image for one manifest row."""
        return Image.open(io.BytesIO(self.read_bytes(shard_index, row_in_shard)))

    def close(self) -> None:
        """Release cached readers and decoded row groups."""
        self._files.clear()
        self._row_group_cache.clear()
        self._cache_order.clear()


class ImageCacheExporter:
    """Extracts a manifest subset to individual PNG files on disk.

    This is the step that makes Colab practical: export the 9,396
    biomarker-labelled, deduplicated scans once (2.3 GB), upload that folder to
    Drive, and train from it without the 30 GB of parquet.
    """

    def __init__(
        self,
        data_root: str | Path,
        config_name: str,
        output_dir: str | Path,
        split: str | None = "train",
        image_column: str = "Image",
    ) -> None:
        self.reader_args = (data_root, config_name, split, image_column)
        self.output_dir = Path(output_dir)
        self.index = ParquetShardIndex(data_root, config_name)
        self.split = split
        self.image_column = image_column

    @staticmethod
    def cache_filename(row: pd.Series) -> str:
        """Deterministic, collision-free filename for one manifest row."""
        return f"s{int(row['shard_index']):03d}_r{int(row['row_in_shard']):05d}.png"

    def export(
        self, frame: pd.DataFrame, overwrite: bool = False, log_every: int = 1000
    ) -> pd.DataFrame:
        """Write one PNG per manifest row and return the frame with ``cache_path``.

        Args:
            frame: Manifest rows to export (typically the modelling subset).
            overwrite: Re-write files that already exist.
            log_every: Progress logging interval.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = frame.copy()
        filenames = out.apply(self.cache_filename, axis=1)
        out["cache_path"] = [str(self.output_dir / name) for name in filenames]

        # Group by shard so each parquet file is opened once and read in order.
        pending = out[["shard_index", "row_in_shard", "cache_path"]].copy()
        if not overwrite:
            pending = pending[~pending["cache_path"].map(lambda p: Path(p).exists())]
        if pending.empty:
            LOGGER.info("image cache already complete: %s", self.output_dir)
            return out

        LOGGER.info("exporting %d images to %s", len(pending), self.output_dir)
        written = 0
        for shard_index, group in pending.groupby("shard_index", sort=True):
            shard = {s.index: s for s in self.index.shards(self.split)}[int(shard_index)]
            parquet = pq.ParquetFile(shard.path)
            wanted = dict(zip(group["row_in_shard"].to_numpy(), group["cache_path"]))
            cursor = 0
            for rg in range(parquet.num_row_groups):
                n_rows = parquet.metadata.row_group(rg).num_rows
                hits = [r for r in wanted if cursor <= r < cursor + n_rows]
                if hits:
                    rows = parquet.read_row_group(rg, columns=[self.image_column])
                    payload = rows.column(self.image_column).to_pylist()
                    for row_index in hits:
                        Path(wanted[row_index]).write_bytes(payload[row_index - cursor]["bytes"])
                        written += 1
                        if written % log_every == 0:
                            LOGGER.info("  exported %d/%d", written, len(pending))
                cursor += n_rows
        LOGGER.info("image cache written: %d files in %s", written, self.output_dir)
        return out

    def cache_size_gb(self) -> float:
        """Total size of the exported cache in GB."""
        if not self.output_dir.exists():
            return 0.0
        return sum(p.stat().st_size for p in self.output_dir.glob("*.png")) / 1e9


@dataclass
class OlivesSample:
    """One dataset item, before collation."""

    image: Any
    clinical: Any
    target: Any
    row_uid: int
    patient_id: int


class GroupBalancedSampler:
    """Sample patients or visits uniformly, then choose one of their scans.

    Optional rare-positive weighting is computed after aggregating labels at the
    selected group level, so repeated B-scans never multiply a patient's weight.
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        targets: np.ndarray | None = None,
        num_samples: int | None = None,
        rare_positive_power: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.group_ids = np.asarray(group_ids)
        self.num_samples = int(num_samples or len(self.group_ids))
        self.seed = seed
        self.epoch = 0
        self.groups = np.unique(self.group_ids)
        self.indices = {
            group: np.flatnonzero(self.group_ids == group) for group in self.groups
        }
        weights = np.ones(len(self.groups), dtype=np.float64)
        if targets is not None and rare_positive_power > 0:
            group_targets = np.stack(
                [np.asarray(targets)[self.indices[group]].max(axis=0) for group in self.groups]
            )
            prevalence = np.clip(group_targets.mean(axis=0), 0.02, 1.0)
            rarity = group_targets / prevalence
            group_score = np.maximum(1.0, rarity.max(axis=1))
            weights *= np.power(np.clip(group_score, 1.0, 10.0), rare_positive_power)
        self.probabilities = weights / weights.sum()

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        selected = rng.choice(
            len(self.groups), size=self.num_samples, replace=True, p=self.probabilities
        )
        for group_position in selected:
            candidates = self.indices[self.groups[group_position]]
            yield int(rng.choice(candidates))


class OlivesDataset:
    """Multilabel OCT dataset over a manifest partition.

    Args:
        frame: Manifest rows for exactly one partition.
        label_columns: Canonical label column names, in a fixed order.
        clinical_preprocessor: Fitted on the training fold; supplies clinical features.
        transform: Torchvision transform applied to the PIL image.
        reader: Parquet reader, used when ``cache_path`` is absent.
        return_image: Set False for the clinical-only baseline to skip image I/O.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        label_columns: list[str],
        clinical_preprocessor: ClinicalPreprocessor | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        reader: ParquetImageReader | None = None,
        return_image: bool = True,
        image_mode: str = "repeat",
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_columns = label_columns
        self.transform = transform
        self.reader = reader
        self.return_image = return_image
        self.image_mode = image_mode
        if image_mode not in {"repeat", "grayscale", "adjacent"}:
            raise ValueError(f"unknown image_mode {image_mode!r}")

        self.targets = self.frame[label_columns].to_numpy(dtype=np.float32)
        self.row_uids = self.frame["row_uid"].to_numpy()
        self.patient_ids = self.frame["patient_id"].to_numpy()

        if clinical_preprocessor is not None:
            self.clinical = clinical_preprocessor.transform(self.frame)
        else:
            self.clinical = np.zeros((len(self.frame), 0), dtype=np.float32)

        self._has_cache = "cache_path" in self.frame.columns
        if return_image and not self._has_cache and reader is None:
            raise ValueError(
                "return_image=True needs either a 'cache_path' column (run ImageCacheExporter) "
                "or a ParquetImageReader"
            )
        self.neighbor_indices = self._build_neighbor_indices()

    def _build_neighbor_indices(self) -> np.ndarray:
        """Previous/current/next indices within an inferred OCT visit."""
        neighbors = np.repeat(np.arange(len(self.frame))[:, None], 3, axis=1)
        if self.image_mode != "adjacent" or self.frame.empty:
            return neighbors
        if "visit_uid" in self.frame.columns:
            grouped = self.frame.groupby("visit_uid", sort=False).indices.values()
        else:
            keys = [c for c in ("patient_id", "eye_id", "visit_index") if c in self.frame]
            grouped = self.frame.groupby(keys, sort=False).indices.values()
        for positions in grouped:
            positions = np.asarray(positions, dtype=int)
            if "scan_number" in self.frame.columns:
                order = pd.to_numeric(
                    self.frame.iloc[positions]["scan_number"], errors="coerce"
                ).fillna(self.frame.iloc[positions]["row_uid"])
                positions = positions[np.argsort(order.to_numpy(), kind="stable")]
            for offset, current in enumerate(positions):
                previous = positions[max(0, offset - 1)]
                following = positions[min(len(positions) - 1, offset + 1)]
                neighbors[current] = (previous, current, following)
        return neighbors

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def n_labels(self) -> int:
        """Number of target labels."""
        return len(self.label_columns)

    @property
    def clinical_dim(self) -> int:
        """Width of the clinical feature vector."""
        return int(self.clinical.shape[1])

    def load_image(self, index: int) -> Image.Image:
        """Load one image from the PNG cache, falling back to parquet."""
        row = self.frame.iloc[index]
        if self._has_cache:
            path = row["cache_path"]
            if isinstance(path, str) and Path(path).exists():
                return Image.open(path)
        if self.reader is None:
            raise FileNotFoundError(f"no image source for row {index}")
        return self.reader.read_image(int(row["shard_index"]), int(row["row_in_shard"]))

    def load_model_image(self, index: int) -> Image.Image:
        """Load one grayscale B-scan or a registered adjacent-slice RGB triplet."""
        if self.image_mode != "adjacent":
            return self.load_image(index).convert("L")
        slices = [
            self.load_image(int(position)).convert("L")
            for position in self.neighbor_indices[index]
        ]
        return Image.merge("RGB", tuple(slices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        target = torch.from_numpy(self.targets[index])
        clinical = torch.from_numpy(self.clinical[index])

        if self.return_image:
            image = self.load_model_image(index)
            image = self.transform(image) if self.transform is not None else image
        else:
            image = torch.zeros(0)

        return {
            "image": image,
            "clinical": clinical,
            "target": target,
            "row_uid": int(self.row_uids[index]),
            "patient_id": int(self.patient_ids[index]),
        }

    def label_prevalence(self) -> pd.Series:
        """Positive rate per label within this partition."""
        return pd.Series(self.targets.mean(axis=0), index=self.label_columns)


class OlivesDataModule:
    """Builds train/val/calibration/test datasets and loaders for one split.

    Owns the ordering that prevents leakage: the clinical preprocessor and class
    weights are fitted on the training partition *before* any other partition is
    constructed.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        assignment: Any,
        label_columns: list[str],
        transform_factory: Any,
        clinical_features: list[str] | None = None,
        use_missingness_indicators: bool = True,
        reader: ParquetImageReader | None = None,
        return_image: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        group_key: str = "patient_id",
        image_mode: str = "repeat",
        normalization_samples: int = 512,
        sampler: str = "shuffle",
        samples_per_epoch: int | None = None,
        rare_positive_sampling_power: float = 0.0,
    ) -> None:
        self.frame = frame
        self.assignment = assignment
        self.label_columns = label_columns
        self.transform_factory = transform_factory
        self.reader = reader
        self.return_image = return_image
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.group_key = group_key
        self.image_mode = image_mode
        self.normalization_samples = normalization_samples
        self.sampler = sampler
        self.samples_per_epoch = samples_per_epoch
        self.rare_positive_sampling_power = rare_positive_sampling_power

        self.preprocessor = ClinicalPreprocessor(
            features=clinical_features or ["bcva", "cst"],
            use_missingness_indicators=use_missingness_indicators,
        )
        self.datasets: dict[str, OlivesDataset] = {}

    def partition_frame(self, name: str) -> pd.DataFrame:
        """Rows belonging to one partition."""
        patients = set(self.assignment.partitions.get(name, []))
        return self.frame[self.frame[self.group_key].isin(patients)].reset_index(drop=True)

    def setup(self) -> OlivesDataModule:
        """Fit training-fold statistics, then build every partition's dataset."""
        train_frame = self.partition_frame("train")
        if train_frame.empty:
            raise ValueError("training partition is empty")
        self.preprocessor.fit(train_frame)
        if self.return_image and self.transform_factory.normalization == "train_fold":
            raw = OlivesDataset(
                frame=train_frame,
                label_columns=self.label_columns,
                reader=self.reader,
                return_image=True,
                image_mode="repeat",
            )
            count = min(self.normalization_samples, len(raw))
            positions = np.linspace(0, len(raw) - 1, count, dtype=int)
            self.transform_factory.fit_normalization(
                raw.load_image(int(position)).convert("L") for position in positions
            )

        for name in self.assignment.partitions:
            frame = self.partition_frame(name)
            if frame.empty:
                LOGGER.warning("partition '%s' has no rows; skipping", name)
                continue
            self.datasets[name] = OlivesDataset(
                frame=frame,
                label_columns=self.label_columns,
                clinical_preprocessor=self.preprocessor,
                transform=self.transform_factory.build(train=(name == "train")),
                reader=self.reader,
                return_image=self.return_image,
                image_mode=self.image_mode,
            )
            LOGGER.info("partition '%s': %d rows, %d patients", name, len(frame), frame[self.group_key].nunique())
        return self

    def dataloader(self, name: str, shuffle: bool | None = None, seed: int = 42) -> Any:
        """Build a DataLoader for one partition."""
        from torch.utils.data import DataLoader

        from olives_biomarkers.utils.reproducibility import SeedManager

        if name not in self.datasets:
            raise KeyError(f"partition '{name}' not built; call setup() first")
        do_shuffle = (name == "train") if shuffle is None else shuffle
        group_sampler = None
        if name == "train" and self.sampler != "shuffle" and shuffle is not False:
            dataset = self.datasets[name]
            column = self.group_key if self.sampler == "patient" else "visit_uid"
            if column not in dataset.frame.columns:
                raise KeyError(f"{self.sampler} sampling requires column {column!r}")
            group_sampler = GroupBalancedSampler(
                dataset.frame[column].to_numpy(),
                targets=dataset.targets,
                num_samples=self.samples_per_epoch,
                rare_positive_power=self.rare_positive_sampling_power,
                seed=seed,
            )
            do_shuffle = False
        return DataLoader(
            self.datasets[name],
            batch_size=self.batch_size,
            shuffle=do_shuffle,
            sampler=group_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=SeedManager(seed).worker_init_fn,
        )

    def pos_weight(self, cap: float = 20.0, unit: str = "scan") -> np.ndarray:
        """Class weights computed from the training partition only."""
        from olives_biomarkers.data.preprocessing import PosWeightCalculator

        if "train" not in self.datasets:
            raise KeyError("call setup() first")
        dataset = self.datasets["train"]
        targets = dataset.targets
        if unit != "scan":
            column = self.group_key if unit == "patient" else "visit_uid"
            if column not in dataset.frame.columns:
                raise KeyError(f"{unit} class weights require column {column!r}")
            target_frame = pd.DataFrame(targets, columns=self.label_columns)
            target_frame["_group"] = dataset.frame[column].to_numpy()
            targets = target_frame.groupby("_group", sort=False)[self.label_columns].mean().to_numpy()
        return PosWeightCalculator(cap).compute(targets)
