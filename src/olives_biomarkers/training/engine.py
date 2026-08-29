"""Training loop.

One :class:`Trainer` drives every model in the comparison. Models declare which
inputs they consume via ``uses_image`` / ``uses_clinical``, so the same loop
trains the clinical-only baseline and the gated fusion model without branching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from olives_biomarkers.config import TrainingConfig
from olives_biomarkers.evaluation.metrics import MultiLabelMetrics
from olives_biomarkers.training.callbacks import CheckpointManager, EarlyStopping, MetricHistory
from olives_biomarkers.training.schedule import (
    FineTuneSchedule,
    ParameterGroupBuilder,
    WarmupCosineSchedule,
)
from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.trainer")


@dataclass
class EpochResult:
    """Outputs of one pass over a partition."""

    loss: float
    logits: np.ndarray
    targets: np.ndarray
    row_uids: np.ndarray
    patient_ids: np.ndarray
    duration_s: float = 0.0


class Trainer:
    """Trains and evaluates one model on one split.

    Args:
        model: Any :class:`BaseBiomarkerModel`.
        criterion: Loss consuming raw logits.
        config: Training hyperparameters.
        device: Torch device string.
        checkpoint_dir: Where best/last checkpoints are written.
        run_id: Identifier used in checkpoint and history filenames.
        label_names: Label names, for per-label metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config: TrainingConfig,
        device: str = "cpu",
        checkpoint_dir: str | Path = "outputs/checkpoints",
        run_id: str = "run",
        label_names: list[str] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion.to(device) if hasattr(criterion, "to") else criterion
        self.config = config
        self.device = device
        self.run_id = run_id
        self.label_names = label_names

        # Discriminative learning rates: a randomly initialised head needs a much
        # larger step than a pretrained backbone that is already near a solution.
        # Both fall back to `learning_rate`, so an unset config behaves as before.
        backbone_lr = config.backbone_learning_rate or config.learning_rate
        head_lr = config.head_learning_rate or config.learning_rate
        self.parameter_groups = ParameterGroupBuilder().build(
            self.model, backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=config.weight_decay
        )
        self.optimizer = torch.optim.AdamW(self.parameter_groups)

        self.fine_tune_schedule = FineTuneSchedule(
            freeze_epochs=config.freeze_backbone_epochs,
            gradual_epochs=config.gradual_unfreeze_epochs,
        )
        self.scheduler = (
            WarmupCosineSchedule(
                self.optimizer,
                total_epochs=config.epochs,
                warmup_epochs=config.warmup_epochs,
                min_ratio=config.min_learning_rate_ratio,
            )
            if config.scheduler == "cosine"
            else None
        )
        self._trainability_mode: str | None = None

        self.scaler = torch.amp.GradScaler(
            device="cuda", enabled=bool(config.amp and device == "cuda")
        )
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience, mode=config.monitor_mode
        )
        self.checkpoints = CheckpointManager(checkpoint_dir, run_id=run_id)
        self.history = MetricHistory()
        self.metrics = MultiLabelMetrics(label_names=label_names)

    # ------------------------------------------------------------------
    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Feed only the modalities the model declares it uses."""
        image = batch["image"].to(self.device, non_blocking=True) if self.model.uses_image else None
        clinical = (
            batch["clinical"].to(self.device, non_blocking=True) if self.model.uses_clinical else None
        )
        return self.model(image=image, clinical=clinical)

    def apply_trainability(self, mode: str) -> bool:
        """Set which encoder stages are trainable. Returns True when it changed.

        The optimiser is deliberately *not* rebuilt. Every parameter is already
        in a group; frozen ones simply produce no gradient and AdamW skips them,
        so unfreezing preserves the momentum accumulated so far.
        """
        if mode == self._trainability_mode:
            return False
        encoder = self.model.image_encoder_module() if hasattr(self.model, "image_encoder_module") else None
        if encoder is None:
            self._trainability_mode = mode
            return False
        encoder.set_backbone_trainable(mode)
        self._trainability_mode = mode
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        LOGGER.info(
            "encoder trainability -> %s (%s trainable parameters)", mode, f"{trainable:,}"
        )
        return True

    def _run_epoch(self, loader: Any, train: bool) -> EpochResult:
        """One pass over a loader, in train or eval mode."""
        self.model.train(train)
        if train:
            # Frozen stages must stay in eval mode so their BatchNorm running
            # statistics do not drift while their weights are held fixed.
            encoder = (
                self.model.image_encoder_module()
                if hasattr(self.model, "image_encoder_module")
                else None
            )
            if encoder is not None:
                encoder.enforce_frozen_eval()
        total_loss, n_batches = 0.0, 0
        logits_out: list[np.ndarray] = []
        targets_out: list[np.ndarray] = []
        uids_out: list[np.ndarray] = []
        patients_out: list[np.ndarray] = []
        start = time.time()

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                targets = batch["target"].to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type="cuda" if self.device == "cuda" else "cpu",
                    enabled=bool(self.config.amp and self.device == "cuda"),
                ):
                    logits = self._forward(batch)
                    loss = self.criterion(logits, targets)

                if train:
                    self.scaler.scale(loss).backward()
                    if self.config.grad_clip_norm:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += float(loss.detach())
                n_batches += 1
                logits_out.append(logits.detach().float().cpu().numpy())
                targets_out.append(targets.detach().cpu().numpy())
                uids_out.append(batch["row_uid"].numpy())
                patients_out.append(batch["patient_id"].numpy())

        return EpochResult(
            loss=total_loss / max(n_batches, 1),
            logits=np.concatenate(logits_out) if logits_out else np.empty((0, 0)),
            targets=np.concatenate(targets_out) if targets_out else np.empty((0, 0)),
            row_uids=np.concatenate(uids_out) if uids_out else np.empty(0),
            patient_ids=np.concatenate(patients_out) if patients_out else np.empty(0),
            duration_s=time.time() - start,
        )

    # ------------------------------------------------------------------
    def train_epoch(self, loader: Any) -> EpochResult:
        """Run one training epoch."""
        return self._run_epoch(loader, train=True)

    @torch.no_grad()
    def evaluate(self, loader: Any) -> EpochResult:
        """Run one deterministic evaluation pass."""
        return self._run_epoch(loader, train=False)

    def fit(self, train_loader: Any, val_loader: Any, epochs: int | None = None) -> MetricHistory:
        """Train with early stopping on the monitored validation metric."""
        total_epochs = epochs or self.config.epochs
        monitor = self.config.monitor
        LOGGER.info(
            "training %s for up to %d epochs on %s (monitor=%s, mode=%s)",
            type(self.model).__name__,
            total_epochs,
            self.device,
            monitor,
            self.config.monitor_mode,
        )

        if self.fine_tune_schedule.enabled:
            LOGGER.info("fine-tune schedule: %s", self.fine_tune_schedule.describe())
        if self.scheduler is not None:
            LOGGER.info(
                "cosine schedule with %d warmup epoch(s), floor %.3f of base LR",
                self.config.warmup_epochs,
                self.config.min_learning_rate_ratio,
            )

        for epoch in range(1, total_epochs + 1):
            phase = self.fine_tune_schedule.mode_for_epoch(epoch)
            self.apply_trainability(phase)
            learning_rates = (
                self.scheduler.current_learning_rates()
                if self.scheduler is not None and epoch > 1
                else {}
            )
            if self.scheduler is not None:
                self.scheduler.step(epoch)
                learning_rates = self.scheduler.current_learning_rates()

            train_result = self.train_epoch(train_loader)
            val_result = self.evaluate(val_loader)

            val_metrics = self.metrics.compute(
                targets=val_result.targets,
                probabilities=self._to_probabilities(val_result.logits),
            )
            record = {
                "train_loss": train_result.loss,
                "val_loss": val_result.loss,
                "train_seconds": round(train_result.duration_s, 1),
                "phase": phase,
                **{f"lr_{name}": value for name, value in learning_rates.items()},
                **{f"val_{k}": v for k, v in val_metrics.items() if np.isscalar(v)},
            }
            self.history.append(epoch, **record)

            monitored = record.get(monitor)
            if monitored is None:
                raise KeyError(
                    f"monitor {monitor!r} not produced; available: {sorted(record)}"
                )

            is_best = self.early_stopping.step(float(monitored), epoch)
            self.checkpoints.save(
                self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=record,
                is_best=is_best,
            )
            LOGGER.info(
                "epoch %3d [%s] | train_loss %.4f | val_loss %.4f | %s %.4f%s",
                epoch,
                phase,
                train_result.loss,
                val_result.loss,
                monitor,
                monitored,
                "  *" if is_best else "",
            )
            if self.early_stopping.should_stop:
                break

        return self.history

    @staticmethod
    def _to_probabilities(logits: np.ndarray) -> np.ndarray:
        """Sigmoid, applied only after the loss has consumed raw logits."""
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

    def predict(self, loader: Any) -> dict[str, np.ndarray]:
        """Deterministic prediction pass returning logits and probabilities."""
        result = self.evaluate(loader)
        return {
            "logits": result.logits,
            "probabilities": self._to_probabilities(result.logits),
            "targets": result.targets,
            "row_uid": result.row_uids,
            "patient_id": result.patient_ids,
        }

    def load_best(self) -> dict[str, Any]:
        """Restore the best checkpoint into the model."""
        return self.checkpoints.load(self.model, map_location=self.device)
