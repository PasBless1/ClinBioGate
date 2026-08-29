"""High-level orchestration used by both the scripts and the notebook.

``OlivesPipeline`` is the single entry point a notebook needs: it resolves the
environment, loads config and schema, builds or loads the manifest, runs the
audit, and produces splits. Each stage is idempotent and cached on disk, so a
notebook can be re-run cheaply.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from olives_biomarkers.config import ConfigLoader, ExperimentConfig
from olives_biomarkers.data.audit import AuditReport, DataAuditor
from olives_biomarkers.data.manifests import Manifest, ManifestBuilder
from olives_biomarkers.data.schema import LabelSchema
from olives_biomarkers.data.splits import (
    GroupedCrossValidator,
    PatientGroupedSplitter,
    SplitAssignment,
    SplitManifestWriter,
)
from olives_biomarkers.utils.environment import RuntimeEnvironment
from olives_biomarkers.utils.logging import LoggerFactory
from olives_biomarkers.utils.reproducibility import SeedManager

LOGGER = LoggerFactory.get("olives.pipeline")


class OlivesPipeline:
    """Wires together environment, config, schema, manifest, audit and splits.

    Example:
        >>> pipeline = OlivesPipeline.from_config("configs/data.yaml")
        >>> manifest = pipeline.get_manifest()
        >>> report = pipeline.run_audit(manifest)
    """

    def __init__(self, config: ExperimentConfig, env: RuntimeEnvironment | None = None) -> None:
        self.config = config
        self.env = env or RuntimeEnvironment.detect()
        self.env.ensure_importable()
        SeedManager(config.project.seed).apply()

        self.data_root = self.env.resolve_data_root(
            config.data.root,
            colab_drive_subpath=config.data.colab.drive_data_subpath,
            drive_mount=config.data.colab.drive_mount,
        )
        self.schema = LabelSchema.from_yaml(
            self.env.resolve(config.data.label_schema),
            target_set=config.data.target_set,
            config_name=config.data.config_name,
        )
        self.manifest_dir = self.env.resolve(config.data.manifest_dir)
        self.output_dir = self.env.resolve("outputs")

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls, config_path: str | Path, repo_root: str | Path | None = None
    ) -> OlivesPipeline:
        """Build a pipeline from a YAML config path."""
        env = RuntimeEnvironment.detect(repo_root)
        config = ConfigLoader(env.repo_root).load(config_path)
        return cls(config, env=env)

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    def manifest_path(self, split: str | None = "train") -> Path:
        """Canonical on-disk location of the manifest for one split."""
        suffix = split or "all"
        return self.manifest_dir / f"{self.config.data.config_name}_{suffix}.parquet"

    @property
    def split_dir(self) -> Path:
        """Directory holding split assignment manifests."""
        return self.manifest_dir / "splits"

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------
    def build_manifest(self, split: str | None = "train", save: bool = True) -> Manifest:
        """Scan the parquet shards and build the metadata-only manifest."""
        builder = ManifestBuilder(
            self.data_root,
            self.schema,
            compute_image_hashes=self.config.manifest.compute_image_hashes,
            hash_digest_size=self.config.manifest.hash_digest_size,
        )
        manifest = builder.build(split=split)
        if save:
            manifest.save(self.manifest_path(split))
        return manifest

    def get_manifest(self, split: str | None = "train", rebuild: bool = False) -> Manifest:
        """Load a cached manifest, building it first if absent or ``rebuild``."""
        path = self.manifest_path(split)
        if path.exists() and not rebuild:
            LOGGER.info("loading cached manifest: %s", path)
            return Manifest.load(path, schema=self.schema, split=split)
        return self.build_manifest(split=split)

    # ------------------------------------------------------------------
    # modelling frame
    # ------------------------------------------------------------------
    @property
    def image_cache_dir(self) -> Path:
        """Directory holding the exported PNG cache of the modelling subset."""
        return self.env.resolve("data/processed/images_labelled")

    def export_image_cache(
        self, manifest: Manifest | None = None, overwrite: bool = False
    ) -> Path:
        """Write the modelling subset to PNGs so training does not re-read parquet.

        This is also what makes Colab practical: the cache is roughly 2 GB against
        30 GB of parquet.
        """
        from olives_biomarkers.data.dataset import ImageCacheExporter

        manifest = manifest or self.get_manifest()
        frame = manifest.modelling_frame(
            policy=self.config.duplicates.policy, labelled_only=True
        )
        exporter = ImageCacheExporter(
            self.data_root,
            self.config.data.config_name,
            output_dir=self.image_cache_dir,
            split="train",
        )
        exported = exporter.export(frame, overwrite=overwrite)
        index_path = self.manifest_dir / "image_cache_index.parquet"
        exported[["row_uid", "patient_id", "cache_path"]].to_parquet(index_path, index=False)
        LOGGER.info(
            "image cache: %d files, %.2f GB, index at %s",
            len(exported),
            exporter.cache_size_gb(),
            index_path,
        )
        return self.image_cache_dir

    def modelling_frame(
        self, manifest: Manifest | None = None, attach_cache: bool = True
    ) -> pd.DataFrame:
        """Deduplicated, biomarker-labelled rows, with cached image paths attached.

        Args:
            attach_cache: Add a ``cache_path`` column pointing at the exported PNGs.
                Falls back silently to parquet reads when the cache is absent.
        """
        manifest = manifest or self.get_manifest()
        frame = manifest.modelling_frame(
            policy=self.config.duplicates.policy, labelled_only=True
        )
        if not attach_cache:
            return frame

        from olives_biomarkers.data.dataset import ImageCacheExporter

        cache_dir = self.image_cache_dir
        if not cache_dir.exists():
            LOGGER.warning(
                "image cache not found at %s; call export_image_cache() for much faster training",
                cache_dir,
            )
            return frame

        names = frame.apply(ImageCacheExporter.cache_filename, axis=1)
        frame = frame.copy()
        frame["cache_path"] = [str(cache_dir / name) for name in names]
        missing = sum(1 for p in frame["cache_path"] if not Path(p).exists())
        if missing:
            LOGGER.warning(
                "%d of %d cached images are missing; re-run export_image_cache()",
                missing,
                len(frame),
            )
        return frame

    def pretraining_frame(
        self,
        assignment: SplitAssignment,
        manifest: Manifest | None = None,
        include_partitions: tuple[str, ...] = ("train",),
    ) -> pd.DataFrame:
        """Unlabelled scans usable for self-supervised pretraining on one fold.

        OLIVES has ~70k unlabelled scans against 9.4k labelled ones, so
        pretraining on the unlabelled pool is attractive. The trap is that
        "unlabelled" does not mean "safe": those scans still belong to patients,
        and pretraining on a test patient's images — even without their labels —
        lets the encoder learn that patient's anatomy. The resulting number is no
        longer an inductive estimate.

        This restricts the pool to patients in ``include_partitions`` (training
        only by default) of the fold being evaluated, which is the guard that
        makes a fold-wise SSL claim defensible. It must be called **per fold**;
        a single pool shared across folds reintroduces the leak.

        Args:
            assignment: The fold whose training patients define the safe pool.
            include_partitions: Partitions to draw from. Adding ``"val"`` is
                defensible only if validation is not used for model selection.

        Returns:
            Deduplicated rows for the permitted patients, labelled or not.
        """
        manifest = manifest or self.get_manifest()
        allowed: set[int] = set()
        for name in include_partitions:
            if name not in assignment.partitions:
                raise KeyError(
                    f"partition {name!r} not in this split; have "
                    f"{sorted(assignment.partitions)}"
                )
            allowed.update(assignment.partitions[name])

        frame = manifest.deduplicated(self.config.duplicates.policy)
        pool = frame[frame[self.config.data.group_key].isin(allowed)].reset_index(drop=True)

        held_out = set().union(
            *(
                set(patients)
                for name, patients in assignment.partitions.items()
                if name not in include_partitions
            )
        )
        contamination = set(pool[self.config.data.group_key]) & held_out
        if contamination:
            raise ValueError(
                f"pretraining pool contains held-out patients {sorted(contamination)}; "
                "this would leak at the patient level"
            )

        LOGGER.info(
            "pretraining pool for '%s': %d scans from %d patients (%d labelled), "
            "%d patients excluded as held-out",
            assignment.name,
            len(pool),
            pool[self.config.data.group_key].nunique(),
            int(pool["has_biomarkers"].sum()) if "has_biomarkers" in pool else 0,
            len(held_out),
        )
        return pool

    def run_audit(
        self, manifest: Manifest | None = None, write: bool = True
    ) -> AuditReport:
        """Run the Phase 0 audit and optionally write the Markdown/JSON reports."""
        manifest = manifest or self.get_manifest()
        auditor = DataAuditor(manifest, self.schema, self.data_root)
        report = auditor.run()
        if write:
            reports_dir = self.output_dir / "reports"
            report.to_markdown(reports_dir / "data_audit.md")
            report.to_json(reports_dir / "data_audit.json")
            LOGGER.info("audit report written to %s", reports_dir)
        return report

    def make_holdout_split(
        self, manifest: Manifest | None = None, write: bool = True
    ) -> SplitAssignment:
        """Create the patient-grouped train/val/test/calibration holdout."""
        manifest = manifest or self.get_manifest()
        frame = manifest.modelling_frame(
            policy=self.config.duplicates.policy, labelled_only=True
        )
        splitter = PatientGroupedSplitter(
            train_fraction=self.config.split.train_fraction,
            val_fraction=self.config.split.val_fraction,
            test_fraction=self.config.split.test_fraction,
            calibration_fraction_of_train=self.config.split.calibration_fraction_of_train,
            stratify_on=self.config.split.stratify_on,
            group_key=self.config.data.group_key,
            seed=self.config.split.seed,
        )
        assignment = splitter.split(frame, name="holdout")
        if write:
            SplitManifestWriter(self.split_dir, self.config.data.group_key).write(
                assignment, frame, manifest.label_columns
            )
        return assignment

    def make_folds(
        self, manifest: Manifest | None = None, write: bool = True
    ) -> list[SplitAssignment]:
        """Create the patient-grouped k-fold assignments for final evaluation."""
        manifest = manifest or self.get_manifest()
        frame = manifest.modelling_frame(
            policy=self.config.duplicates.policy, labelled_only=True
        )
        cv = GroupedCrossValidator(
            n_folds=self.config.split.n_folds,
            calibration_fraction_of_train=self.config.split.calibration_fraction_of_train,
            stratify_on=self.config.split.stratify_on,
            group_key=self.config.data.group_key,
            seed=self.config.split.seed,
        )
        folds = cv.split(frame)
        if write:
            writer = SplitManifestWriter(self.split_dir, self.config.data.group_key)
            for fold in folds:
                writer.write(fold, frame, manifest.label_columns)
        return folds

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def describe(self) -> str:
        """Multi-line description of the resolved pipeline state."""
        lines = [
            "OLIVES pipeline",
            "=" * 60,
            self.env.summary(),
            "-" * 60,
            f"{'config':>16}: {self.config.source_path}",
            f"{'experiment':>16}: {self.config.experiment.name}",
            f"{'data_root':>16}: {self.data_root}",
            f"{'data_exists':>16}: {self.data_root.exists()}",
            f"{'hf_config':>16}: {self.config.data.config_name}",
            f"{'target_set':>16}: {self.config.data.target_set} ({self.schema.n_labels} labels)",
            f"{'manifest':>16}: {self.manifest_path()}",
            f"{'manifest_cached':>16}: {self.manifest_path().exists()}",
            "-" * 60,
            self.schema.describe(),
        ]
        return "\n".join(lines)

    def state(self) -> dict[str, Any]:
        """Machine-readable pipeline state, for run metadata."""
        return {
            "config_path": self.config.source_path,
            "data_root": str(self.data_root),
            "data_exists": self.data_root.exists(),
            "manifest_path": str(self.manifest_path()),
            "manifest_cached": self.manifest_path().exists(),
            "target_set": self.config.data.target_set,
            "n_labels": self.schema.n_labels,
            "device": self.env.device,
            "is_colab": self.env.is_colab,
        }
