"""Clinical preprocessing and OCT image transforms.

Two hard rules are enforced structurally rather than by convention:

1. Imputers, scalers and class weights are **fitted on the training fold only**;
   :class:`ClinicalPreprocessor` refuses to transform before ``fit`` and refuses
   to re-fit silently.
2. Augmentations never include a vertical flip, and horizontal flips are opt-in,
   because OCT B-scans encode retinal layer order vertically and laterality
   horizontally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from olives_biomarkers.utils.io import JsonIO
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.preprocessing")


class NotFittedError(RuntimeError):
    """Raised when a preprocessor is used before being fitted."""


@dataclass
class ClinicalPreprocessorState:
    """Fitted parameters, persisted so evaluation reuses the training statistics."""

    features: list[str]
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    use_missingness_indicators: bool = True
    n_fit_rows: int = 0
    fit_patients: list[int] = field(default_factory=list)


class ClinicalPreprocessor:
    """Median-imputes and standardizes BCVA/CST, keeping missingness indicators.

    The missingness indicator is a feature in its own right: in OLIVES the
    missing values are concentrated in a single patient, so "this value is
    absent" carries information the imputed median destroys.

    Example:
        >>> pre = ClinicalPreprocessor(["bcva", "cst"])
        >>> pre.fit(train_frame)
        >>> features = pre.transform(val_frame)
    """

    def __init__(
        self,
        features: list[str] | None = None,
        use_missingness_indicators: bool = True,
    ) -> None:
        self.features = list(features) if features else ["bcva", "cst"]
        self.use_missingness_indicators = use_missingness_indicators
        self.state: ClinicalPreprocessorState | None = None

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """Whether ``fit`` has been called."""
        return self.state is not None

    @property
    def output_dim(self) -> int:
        """Width of the transformed feature vector."""
        return len(self.features) * (2 if self.use_missingness_indicators else 1)

    @property
    def feature_names(self) -> list[str]:
        """Names of the output columns, in order."""
        names = list(self.features)
        if self.use_missingness_indicators:
            names += [f"{f}_missing" for f in self.features]
        return names

    # ------------------------------------------------------------------
    def fit(self, frame: pd.DataFrame, allow_refit: bool = False) -> ClinicalPreprocessor:
        """Fit imputation and scaling statistics on the training fold only.

        Args:
            frame: Training-partition rows only. Passing validation or test rows
                here is the leak this class exists to prevent.
            allow_refit: Guard against accidental re-fitting on another partition.
        """
        if self.is_fitted and not allow_refit:
            raise RuntimeError(
                "ClinicalPreprocessor is already fitted. Re-fitting on a different partition "
                "leaks statistics across the split; pass allow_refit=True only if you mean it."
            )
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise KeyError(f"clinical features missing from frame: {missing}")

        state = ClinicalPreprocessorState(
            features=list(self.features),
            use_missingness_indicators=self.use_missingness_indicators,
            n_fit_rows=int(len(frame)),
            fit_patients=sorted(frame["patient_id"].unique().tolist())
            if "patient_id" in frame.columns
            else [],
        )
        for feature in self.features:
            values = pd.to_numeric(frame[feature], errors="coerce")
            median = float(values.median())
            imputed = values.fillna(median)
            std = float(imputed.std(ddof=0))
            state.medians[feature] = median
            state.means[feature] = float(imputed.mean())
            state.stds[feature] = std if std > 1e-8 else 1.0
        self.state = state
        LOGGER.info(
            "ClinicalPreprocessor fitted on %d rows / %d patients: %s",
            state.n_fit_rows,
            len(state.fit_patients),
            {f: round(state.medians[f], 2) for f in self.features},
        )
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Impute, standardize and append missingness indicators.

        Returns:
            Float32 array of shape ``(n_rows, output_dim)``.
        """
        if self.state is None:
            raise NotFittedError("call fit() on the training fold before transform()")

        columns: list[np.ndarray] = []
        indicators: list[np.ndarray] = []
        for feature in self.features:
            values = pd.to_numeric(frame[feature], errors="coerce")
            is_missing = values.isna().to_numpy(dtype=np.float32)
            filled = values.fillna(self.state.medians[feature]).to_numpy(dtype=np.float32)
            scaled = (filled - self.state.means[feature]) / self.state.stds[feature]
            columns.append(scaled.astype(np.float32))
            indicators.append(is_missing)

        if self.use_missingness_indicators:
            columns.extend(indicators)
        return np.stack(columns, axis=1).astype(np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Convenience wrapper; only ever call this on the training fold."""
        return self.fit(frame).transform(frame)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist fitted statistics next to the run's checkpoints."""
        if self.state is None:
            raise NotFittedError("nothing to save; preprocessor is not fitted")
        return JsonIO.write(vars(self.state), path)

    @classmethod
    def load(cls, path: str | Path) -> ClinicalPreprocessor:
        """Restore a fitted preprocessor."""
        payload = JsonIO.read(path)
        preprocessor = cls(
            features=payload["features"],
            use_missingness_indicators=payload["use_missingness_indicators"],
        )
        preprocessor.state = ClinicalPreprocessorState(**payload)
        return preprocessor


class PosWeightCalculator:
    """Computes per-label ``pos_weight`` for ``BCEWithLogitsLoss`` from a training fold."""

    def __init__(self, cap: float = 20.0) -> None:
        self.cap = cap

    def compute(self, labels: np.ndarray) -> np.ndarray:
        """Return ``negatives / positives`` per label, capped and NaN-safe.

        Args:
            labels: Binary array of shape ``(n_samples, n_labels)`` from the
                training fold only.
        """
        labels = np.asarray(labels, dtype=np.float32)
        positives = labels.sum(axis=0)
        negatives = labels.shape[0] - positives
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(positives > 0, negatives / np.maximum(positives, 1.0), 1.0)
        weights = np.clip(weights, 1.0, self.cap)
        if np.any(positives == 0):
            empty = np.where(positives == 0)[0].tolist()
            LOGGER.warning(
                "labels %s have zero positives in this fold; pos_weight set to 1.0 and their "
                "metrics will be undefined",
                empty,
            )
        return weights.astype(np.float32)


class ImageTransformFactory:
    """Builds torchvision transforms appropriate to OCT B-scans.

    Vertical flips are never produced: retinal layers run top-to-bottom and a
    flip would invert anatomy. Horizontal flips are off by default because they
    mirror laterality; enable only after confirming it is acceptable.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    # Grayscale statistics reported for OLIVES B-scans in the dataset paper.
    OLIVES_MEAN = (0.482,)
    OLIVES_STD = (0.037,)

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        to_three_channel: bool = True,
        use_imagenet_norm: bool = True,
        horizontal_flip: bool = False,
    ) -> None:
        self.image_size = tuple(image_size)
        self.to_three_channel = to_three_channel
        self.use_imagenet_norm = use_imagenet_norm
        self.horizontal_flip = horizontal_flip

    def _normalization(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if self.to_three_channel and self.use_imagenet_norm:
            return self.IMAGENET_MEAN, self.IMAGENET_STD
        if self.to_three_channel:
            return self.OLIVES_MEAN * 3, self.OLIVES_STD * 3
        return self.OLIVES_MEAN, self.OLIVES_STD

    def build(self, train: bool = False) -> Any:
        """Return a torchvision transform pipeline.

        Args:
            train: Include the conservative augmentation block.
        """
        from torchvision import transforms

        channels = 3 if self.to_three_channel else 1
        steps: list[Any] = [
            transforms.Grayscale(num_output_channels=channels),
            transforms.Resize(self.image_size),
        ]
        if train:
            augmentations = [
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
            ]
            if self.horizontal_flip:
                augmentations.insert(0, transforms.RandomHorizontalFlip(p=0.5))
            steps.extend(augmentations)

        mean, std = self._normalization()
        steps.append(transforms.ToTensor())
        steps.append(transforms.Normalize(mean=mean, std=std))
        if train:
            steps.append(GaussianNoise(std=0.01))
        return transforms.Compose(steps)


class GaussianNoise:
    """Adds mild Gaussian noise to a normalized tensor (train-time only)."""

    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, tensor: Any) -> Any:
        import torch

        if self.std <= 0:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std

    def __repr__(self) -> str:
        return f"{type(self).__name__}(std={self.std})"
