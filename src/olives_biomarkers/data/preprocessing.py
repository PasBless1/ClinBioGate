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


class RetinalTissueCrop:
    """Crop empty OCT borders using image intensity only.

    The operation is deterministic and label-free. Rows and columns must contain
    enough foreground pixels, preventing isolated sensor noise from expanding
    the crop back to the full canvas.
    """

    def __init__(
        self,
        threshold: float = 0.04,
        padding: float = 0.08,
        min_occupancy: float = 0.02,
    ) -> None:
        self.threshold = threshold
        self.padding = padding
        self.min_occupancy = min_occupancy

    def __call__(self, image: Any) -> Any:
        from PIL import Image

        array = np.asarray(image)
        gray = array.mean(axis=2) if array.ndim == 3 else array
        foreground = gray > (255.0 * self.threshold)
        rows = np.flatnonzero(foreground.mean(axis=1) >= self.min_occupancy)
        cols = np.flatnonzero(foreground.mean(axis=0) >= self.min_occupancy)
        if len(rows) < 2 or len(cols) < 2:
            return image

        top, bottom = int(rows[0]), int(rows[-1] + 1)
        left, right = int(cols[0]), int(cols[-1] + 1)
        pad_y = int(round((bottom - top) * self.padding))
        pad_x = int(round((right - left) * self.padding))
        top, bottom = max(0, top - pad_y), min(gray.shape[0], bottom + pad_y)
        left, right = max(0, left - pad_x), min(gray.shape[1], right + pad_x)
        return Image.fromarray(array[top:bottom, left:right])


class SquarePad:
    """Pad an image to a square without stretching retinal anatomy."""

    def __init__(self, fill: int = 0) -> None:
        self.fill = fill

    def __call__(self, image: Any) -> Any:
        from PIL import ImageOps

        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return ImageOps.expand(
            image,
            border=(left, top, side - width - left, side - height - top),
            fill=self.fill,
        )


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
        image_mode: str | None = None,
        crop_retina: bool = False,
        crop_threshold: float = 0.04,
        crop_padding: float = 0.08,
        preserve_aspect_ratio: bool = False,
        normalization: str | None = None,
    ) -> None:
        self.image_size = tuple(image_size)
        self.image_mode = image_mode or ("repeat" if to_three_channel else "grayscale")
        if self.image_mode not in {"repeat", "grayscale", "adjacent"}:
            raise ValueError(f"unknown image mode {self.image_mode!r}")
        self.to_three_channel = self.image_mode != "grayscale"
        self.normalization = normalization or ("imagenet" if use_imagenet_norm else "olives")
        if self.normalization not in {"imagenet", "olives", "train_fold"}:
            raise ValueError(f"unknown normalization {self.normalization!r}")
        self.use_imagenet_norm = self.normalization == "imagenet"
        self.horizontal_flip = horizontal_flip
        self.crop_retina = crop_retina
        self.crop_threshold = crop_threshold
        self.crop_padding = crop_padding
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.fitted_mean: tuple[float, ...] | None = None
        self.fitted_std: tuple[float, ...] | None = None

    def _normalization(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if self.normalization == "train_fold":
            if self.fitted_mean is None or self.fitted_std is None:
                raise NotFittedError("fit image normalization on training patients first")
            return self.fitted_mean, self.fitted_std
        if self.to_three_channel and self.use_imagenet_norm:
            return self.IMAGENET_MEAN, self.IMAGENET_STD
        if self.to_three_channel:
            return self.OLIVES_MEAN * 3, self.OLIVES_STD * 3
        return self.OLIVES_MEAN, self.OLIVES_STD

    def _geometry_steps(self) -> list[Any]:
        """Deterministic, label-free geometry shared by fitting and inference."""
        from torchvision import transforms

        steps: list[Any] = []
        if self.image_mode != "adjacent":
            channels = 3 if self.to_three_channel else 1
            steps.append(transforms.Grayscale(num_output_channels=channels))
        if self.crop_retina:
            steps.append(RetinalTissueCrop(self.crop_threshold, self.crop_padding))
        if self.preserve_aspect_ratio:
            steps.append(SquarePad())
        steps.append(transforms.Resize(self.image_size))
        return steps

    def fit_normalization(self, images: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Estimate pixel statistics from sampled training-patient images only."""
        from torchvision import transforms

        geometry: list[Any] = [
            transforms.Grayscale(num_output_channels=1),
        ]
        if self.crop_retina:
            geometry.append(RetinalTissueCrop(self.crop_threshold, self.crop_padding))
        if self.preserve_aspect_ratio:
            geometry.append(SquarePad())
        geometry.extend([transforms.Resize(self.image_size), transforms.ToTensor()])
        prepare = transforms.Compose(geometry)

        total = 0.0
        squared = 0.0
        count = 0
        for image in images:
            tensor = prepare(image).double()
            total += float(tensor.sum())
            squared += float((tensor * tensor).sum())
            count += tensor.numel()
        if count == 0:
            raise ValueError("cannot fit image normalization from an empty training sample")
        mean_value = total / count
        variance = max(squared / count - mean_value * mean_value, 1e-8)
        std_value = float(np.sqrt(variance))
        channels = 3 if self.to_three_channel else 1
        self.fitted_mean = (float(mean_value),) * channels
        self.fitted_std = (std_value,) * channels
        LOGGER.info(
            "image normalization fitted on training patients: mean=%.4f std=%.4f",
            mean_value,
            std_value,
        )
        return self.fitted_mean, self.fitted_std

    def build(self, train: bool = False) -> Any:
        """Return a torchvision transform pipeline.

        Args:
            train: Include the conservative augmentation block.
        """
        from torchvision import transforms

        steps = self._geometry_steps()
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
