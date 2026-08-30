"""Training callbacks: early stopping, checkpointing, metric history."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from olives_biomarkers.utils.io import JsonIO
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.callbacks")


class EarlyStopping:
    """Stops training when the monitored metric stops improving.

    Args:
        patience: Epochs to wait after the last improvement.
        mode: ``"max"`` for metrics like AUPRC, ``"min"`` for losses.
        min_delta: Minimum change that counts as an improvement.
    """

    def __init__(self, patience: int = 8, mode: str = "max", min_delta: float = 0.0) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.best_epoch: int = -1
        self.counter: int = 0
        self.should_stop: bool = False

    def is_improvement(self, value: float) -> bool:
        """Whether ``value`` improves on the best seen so far."""
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Record an epoch's metric. Returns True when it was an improvement."""
        if self.is_improvement(value):
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            LOGGER.info(
                "early stopping at epoch %d; best %.4f at epoch %d", epoch, self.best, self.best_epoch
            )
        return False

    def state_dict(self) -> dict[str, Any]:
        """Serializable state needed to continue patience counting after a restart."""
        return {
            "best": self.best,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`."""
        self.best = state.get("best")
        self.best_epoch = int(state.get("best_epoch", -1))
        self.counter = int(state.get("counter", 0))
        self.should_stop = bool(state.get("should_stop", False))


class CheckpointManager:
    """Saves best and last checkpoints, plus everything needed to resume."""

    def __init__(self, directory: str | Path, run_id: str = "run") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    @property
    def best_path(self) -> Path:
        """Path of the best-metric checkpoint."""
        return self.directory / f"{self.run_id}_best.pt"

    @property
    def last_path(self) -> Path:
        """Path of the most recent checkpoint."""
        return self.directory / f"{self.run_id}_last.pt"

    def save(
        self,
        model: Any,
        optimizer: Any = None,
        epoch: int = 0,
        metrics: dict[str, float] | None = None,
        is_best: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically write ``last`` and, on improvement, ``best``.

        The temporary file lives beside the destination, so a Colab disconnect
        cannot leave a half-written checkpoint masquerading as a valid one on
        Google Drive.
        """
        import torch

        payload = {
            "checkpoint_version": 2,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "run_id": self.run_id,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            payload.update(extra)

        self._atomic_torch_save(payload, self.last_path)
        if is_best:
            self._atomic_copy(self.last_path, self.best_path)
            LOGGER.info("new best checkpoint at epoch %d -> %s", epoch, self.best_path.name)
        return self.last_path

    @staticmethod
    def _temporary_path(target: Path) -> Path:
        """A same-directory temporary path suitable for an atomic replacement."""
        return target.with_name(f".{target.name}.tmp")

    def _atomic_torch_save(self, payload: dict[str, Any], target: Path) -> None:
        """Serialize once and replace the destination only after success."""
        import torch

        temporary = self._temporary_path(target)
        try:
            torch.save(payload, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def _atomic_copy(self, source: Path, target: Path) -> None:
        """Copy a completed checkpoint without serializing the model twice."""
        temporary = self._temporary_path(target)
        try:
            shutil.copy2(source, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, model: Any, path: str | Path | None = None, map_location: str = "cpu") -> dict[str, Any]:
        """Restore weights into ``model`` and return the checkpoint payload."""
        import torch

        target = Path(path) if path else self.best_path
        if not target.exists():
            raise FileNotFoundError(f"checkpoint not found: {target}")
        payload = torch.load(target, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        LOGGER.info("loaded checkpoint %s (epoch %s)", target.name, payload.get("epoch"))
        return payload

    def export_model(
        self,
        model: Any,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save an inference-ready best-model bundle separate from resume state."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "run_id": self.run_id,
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }
        self._atomic_torch_save(payload, target)
        LOGGER.info("exported model bundle -> %s", target)
        return target


@dataclass
class MetricHistory:
    """Per-epoch metric log, written as CSV and JSON for the final report."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def append(self, epoch: int, **metrics: Any) -> None:
        """Record one epoch."""
        self.records.append({"epoch": epoch, **metrics})

    def to_frame(self) -> pd.DataFrame:
        """History as a DataFrame."""
        return pd.DataFrame(self.records)

    def best(self, metric: str, mode: str = "max") -> dict[str, Any] | None:
        """The record with the best value of ``metric``."""
        candidates = [r for r in self.records if metric in r and r[metric] is not None]
        if not candidates:
            return None
        return (max if mode == "max" else min)(candidates, key=lambda r: r[metric])

    def save(self, directory: str | Path, run_id: str = "run") -> dict[str, Path]:
        """Write ``<run_id>_history.csv`` and ``.json``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / f"{run_id}_history.csv"
        self.to_frame().to_csv(csv_path, index=False)
        json_path = JsonIO.write(self.records, out / f"{run_id}_history.json")
        return {"csv": csv_path, "json": json_path}
