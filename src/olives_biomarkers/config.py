"""Typed configuration objects loaded from YAML.

Every knob the pipeline exposes lives here, so no script or notebook needs a
hard-coded path or hyperparameter. ``ConfigLoader.load`` resolves the optional
``defaults:`` key so experiment configs inherit from ``configs/data.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from olives_biomarkers.utils.io import YamlIO, deep_merge


@dataclass
class ProjectConfig:
    """Top-level identity of the project."""

    name: str = "olives_multimodal_biomarkers"
    seed: int = 42


@dataclass
class ColabConfig:
    """Where the data lives when running on a Colab runtime."""

    drive_mount: str = "/content/drive"
    drive_data_subpath: str = "MyDrive/olives/data/raw/OLIVES"


@dataclass
class DataConfig:
    """Dataset location, schema and loading options."""

    source: str = "local_parquet"
    root: str = "data/raw/OLIVES"
    hf_repo_id: str = "gOLIVES/OLIVES_Dataset"
    config_name: str = "disease_classification"
    label_schema: str = "configs/label_schema.yaml"
    manifest_dir: str = "data/manifests"
    target_set: str = "sixteen"
    group_key: str = "patient_id"
    image_size: tuple[int, int] = (224, 224)
    num_workers: int = 4
    image_mode: str = "repeat"  # repeat | grayscale | adjacent
    crop_retina: bool = False
    crop_threshold: float = 0.04
    crop_padding: float = 0.08
    preserve_aspect_ratio: bool = False
    normalization: str = "imagenet"  # imagenet | olives | train_fold
    normalization_samples: int = 512
    horizontal_flip: bool = False
    # Within-eye clinical context. Absolute BCVA/CST are largely redundant with
    # the B-scan; the change from the eye's own baseline visit is not.
    longitudinal_clinical: bool = False
    # Control ladder separating clinical signal from visit identity. Any value
    # other than "none" marks the run as a control, not a model result.
    clinical_perturbation: str = "none"
    clinical_perturbation_seed: int = 42
    clinical_bin_cst: float = 25.0
    clinical_bin_bcva: float = 5.0
    colab: ColabConfig = field(default_factory=ColabConfig)

    PERTURBATIONS = (
        "none",
        "patient_mean",
        "within_patient_shuffle",
        "across_patient_shuffle",
        "quantise",
    )

    def __post_init__(self) -> None:
        if self.target_set not in {"six", "sixteen"}:
            raise ValueError(f"target_set must be 'six' or 'sixteen', got {self.target_set!r}")
        if self.config_name not in {"disease_classification", "biomarker_detection"}:
            raise ValueError(f"unknown config_name {self.config_name!r}")
        if self.clinical_perturbation not in self.PERTURBATIONS:
            raise ValueError(
                f"unknown clinical_perturbation {self.clinical_perturbation!r}; "
                f"choose from {self.PERTURBATIONS}"
            )
        self.image_size = tuple(self.image_size)  # type: ignore[assignment]

    @property
    def clinical_bins(self) -> dict[str, float]:
        """Bin widths used by the ``quantise`` control, in clinical units."""
        return {"cst": self.clinical_bin_cst, "bcva": self.clinical_bin_bcva}

    @property
    def is_control_arm(self) -> bool:
        """Whether this configuration perturbs the clinical inputs."""
        return self.clinical_perturbation != "none"


@dataclass
class ManifestConfig:
    """Controls the metadata-only manifest build."""

    compute_image_hashes: bool = True
    hash_digest_size: int = 12
    biomarker_labelled_only: bool = False


@dataclass
class SplitConfig:
    """Patient-grouped partitioning options."""

    strategy: str = "grouped_holdout"
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    calibration_fraction_of_train: float = 0.15
    n_folds: int = 5
    stratify_on: str | None = "disease_label"
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test fractions must sum to 1.0, got {total}")
        if self.strategy not in {"grouped_holdout", "grouped_kfold"}:
            raise ValueError(f"unknown split strategy {self.strategy!r}")


@dataclass
class DuplicateConfig:
    """How exact-duplicate images are handled."""

    policy: str = "keep_first"
    block_cross_partition: bool = True

    def __post_init__(self) -> None:
        if self.policy not in {"keep_first", "keep_all", "drop_all"}:
            raise ValueError(f"unknown duplicate policy {self.policy!r}")


@dataclass
class ModelConfig:
    """Architecture selection and dimensions."""

    name: str = "oct_only"
    image_encoder: str = "resnet18"
    pretrained: bool = True
    image_embedding_dim: int = 256
    clinical_features: list[str] = field(default_factory=lambda: ["bcva", "cst"])
    use_missingness_indicators: bool = True
    clinical_embedding_dim: int = 32
    clinical_hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.30
    in_channels: int = 3
    pretrained_checkpoint: str | None = None
    checkpoint_key: str = "model"
    film_max_scale: float = 0.25
    film_max_shift: float = 0.25
    clinical_residual_max_scale: float = 1.0
    clinical_residual_per_label: bool = True
    gate_residual: bool = True
    gate_bias_init: float | None = None
    gate_scale_alpha: float = 1.0


@dataclass
class TrainingConfig:
    """Optimisation schedule and early stopping."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    backbone_learning_rate: float | None = None
    head_learning_rate: float | None = None
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    monitor: str = "val_macro_auprc"
    monitor_mode: str = "max"
    amp: bool = True
    pos_weight: str | None = "from_train_fold"
    pos_weight_cap: float = 20.0
    loss: str = "bce"
    asl_gamma_negative: float = 4.0
    asl_gamma_positive: float = 1.0
    asl_clip: float = 0.05
    exclusive_label_pairs: list[list[str]] = field(default_factory=list)
    exclusivity_penalty: float = 0.0
    model_regularization_weight: float = 0.0
    pos_weight_unit: str = "scan"  # scan | patient | visit
    sampler: str = "shuffle"  # shuffle | patient | visit
    samples_per_epoch: int | None = None
    rare_positive_sampling_power: float = 0.0
    freeze_backbone_epochs: int = 0
    gradual_unfreeze_epochs: int = 0
    scheduler: str = "none"  # none | cosine
    warmup_epochs: int = 0
    min_learning_rate_ratio: float = 0.01
    grad_clip_norm: float | None = None
    resume_from_checkpoint: bool = True
    reuse_completed_run: bool = True


@dataclass
class UncertaintyConfig:
    """Monte Carlo dropout settings."""

    mc_dropout_passes: int = 30


@dataclass
class EvaluationConfig:
    """Metric computation and threshold policy."""

    bootstrap_iterations: int = 1000
    threshold_strategy: str = "per_label_validation_f1"
    coverage_levels: list[float] = field(default_factory=lambda: [1.0, 0.9, 0.8, 0.7])
    tta_horizontal_flip: bool = False
    ensemble_space: str = "logit"  # logit | probability


@dataclass
class ExperimentMeta:
    """Human-facing identity of one experiment."""

    name: str = "unnamed"
    description: str = ""


@dataclass
class ExperimentConfig:
    """The complete, resolved configuration for one run."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    manifest: ManifestConfig = field(default_factory=ManifestConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    duplicates: DuplicateConfig = field(default_factory=DuplicateConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view for logging and run metadata."""
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Persist the resolved config beside a run's artefacts."""
        return YamlIO.write(self.to_dict(), path)


class ConfigLoader:
    """Loads YAML into :class:`ExperimentConfig`, resolving ``defaults:`` inheritance."""

    SECTION_TYPES: dict[str, type] = {
        "project": ProjectConfig,
        "data": DataConfig,
        "manifest": ManifestConfig,
        "split": SplitConfig,
        "duplicates": DuplicateConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "uncertainty": UncertaintyConfig,
        "evaluation": EvaluationConfig,
        "experiment": ExperimentMeta,
    }

    def __init__(self, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.repo_root / candidate)

    def _read_with_defaults(self, path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
        """Read one YAML file, recursively merging any ``defaults:`` parent first."""
        resolved = self._resolve(path).resolve()
        seen = _seen or set()
        if resolved in seen:
            raise ValueError(f"circular defaults chain at {resolved}")
        seen.add(resolved)
        if not resolved.exists():
            raise FileNotFoundError(f"config not found: {resolved}")

        raw = YamlIO.read(resolved)
        parent_path = raw.pop("defaults", None)
        if parent_path is None:
            return raw
        parent = self._read_with_defaults(parent_path, seen)
        return deep_merge(parent, raw)

    @staticmethod
    def _build_section(section_type: type, payload: Any) -> Any:
        """Instantiate a config dataclass, rejecting unknown keys loudly."""
        if not isinstance(payload, dict):
            raise TypeError(f"expected a mapping for {section_type.__name__}, got {type(payload)}")
        allowed = {f.name for f in section_type.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"unknown keys for {section_type.__name__}: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}"
            )
        return section_type(**payload)

    #: Top-level keys that are metadata rather than config sections. A config
    #: saved by ``ExperimentConfig.save`` round-trips through here, so anything
    #: ``to_dict`` emits outside SECTION_TYPES must be listed.
    NON_SECTION_KEYS = frozenset({"source_path"})

    def load(self, path: str | Path) -> ExperimentConfig:
        """Load and validate a configuration file into an :class:`ExperimentConfig`.

        Accepts both hand-written configs and ones previously written by
        :meth:`ExperimentConfig.save`, so a finished run can be re-loaded for
        evaluation without editing its artefacts.
        """
        raw = self._read_with_defaults(path)
        raw = {k: v for k, v in raw.items() if k not in self.NON_SECTION_KEYS}

        # Nested dataclass inside DataConfig needs building before DataConfig itself.
        data_payload = dict(raw.get("data", {}))
        if "colab" in data_payload:
            data_payload["colab"] = self._build_section(ColabConfig, data_payload["colab"])
        raw = dict(raw)
        if data_payload:
            raw["data"] = data_payload

        sections: dict[str, Any] = {}
        for key, section_type in self.SECTION_TYPES.items():
            if key in raw:
                sections[key] = self._build_section(section_type, raw[key])

        unknown_sections = set(raw) - set(self.SECTION_TYPES)
        if unknown_sections:
            raise ValueError(f"unknown config sections: {sorted(unknown_sections)}")

        config = ExperimentConfig(**sections)
        config.source_path = str(self._resolve(path))
        return config
