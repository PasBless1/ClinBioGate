"""Optimiser parameter groups, LR schedule, and progressive unfreezing.

Three things a single fixed learning rate over the whole network cannot express,
all of which matter when fine-tuning a pretrained encoder on 52 patients:

* **Discriminative learning rates.** A randomly initialised head needs a much
  larger step than a pretrained backbone, which is already near a good solution
  and will be destroyed by the head's learning rate.
* **Warmup then cosine decay.** The first optimiser steps see a randomly
  initialised head producing large, badly directed gradients; warming up stops
  those from wrecking the encoder.
* **Progressive unfreezing.** Training the head against a frozen encoder first
  gives it something sensible to ask for before any encoder weight moves.

The schedule is stated declaratively by :class:`FineTuneSchedule` so the training
loop only has to ask "what should be trainable at epoch N", and the phase
transitions land in the run history where they can be read against the loss
curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator

from torch import nn

from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.schedule")


class ParameterGroupBuilder:
    """Splits a model into backbone and head parameter groups.

    The backbone group is everything inside the image encoder's pretrained
    trunk; the head group is everything else — projection, clinical encoder,
    fusion parameters and the classifier.

    Normalisation and bias parameters are additionally split out with zero weight
    decay: decaying a BatchNorm scale or a bias shrinks it toward zero for no
    regularisation benefit, which matters more here because the weight decay
    being recommended (1e-3) is an order of magnitude above the previous default.
    """

    NO_DECAY_SUFFIXES = (".bias",)
    NO_DECAY_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)

    def __init__(self, separate_no_decay: bool = True) -> None:
        self.separate_no_decay = separate_no_decay

    @staticmethod
    def _backbone_parameter_names(model: nn.Module) -> set[str]:
        """Fully-qualified names of parameters inside the pretrained trunk."""
        encoder = getattr(model, "image_encoder_module", None)
        trunk = encoder() if callable(encoder) else None
        if trunk is None or not hasattr(trunk, "backbone"):
            return set()
        # Locate the encoder's prefix in the parent model so names line up.
        prefixes = [
            name for name, module in model.named_modules() if module is trunk.backbone
        ]
        if not prefixes:
            return set()
        prefix = prefixes[0]
        return {
            name
            for name, _ in model.named_parameters()
            if name == prefix or name.startswith(prefix + ".")
        }

    def _no_decay_parameter_names(self, model: nn.Module) -> set[str]:
        names: set[str] = set()
        for module_name, module in model.named_modules():
            if isinstance(module, self.NO_DECAY_TYPES):
                for parameter_name, _ in module.named_parameters(recurse=False):
                    names.add(f"{module_name}.{parameter_name}" if module_name else parameter_name)
        for name, _ in model.named_parameters():
            if name.endswith(self.NO_DECAY_SUFFIXES):
                names.add(name)
        return names

    def build(
        self,
        model: nn.Module,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        """Return AdamW parameter groups tagged with a readable ``name``.

        Every parameter is included regardless of ``requires_grad``. Parameters
        that are currently frozen simply receive no gradient, and AdamW skips
        them — which is what lets progressive unfreezing work without rebuilding
        the optimiser and discarding its momentum state.
        """
        backbone_names = self._backbone_parameter_names(model)
        no_decay_names = self._no_decay_parameter_names(model) if self.separate_no_decay else set()

        buckets: dict[str, dict[str, Any]] = {
            "backbone": {"params": [], "lr": backbone_lr, "weight_decay": weight_decay},
            "backbone_no_decay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0},
            "head": {"params": [], "lr": head_lr, "weight_decay": weight_decay},
            "head_no_decay": {"params": [], "lr": head_lr, "weight_decay": 0.0},
        }
        for name, parameter in model.named_parameters():
            scope = "backbone" if name in backbone_names else "head"
            if name in no_decay_names:
                scope = f"{scope}_no_decay"
            buckets[scope]["params"].append(parameter)

        groups = [{"name": key, **value} for key, value in buckets.items() if value["params"]]
        LOGGER.info(
            "parameter groups: %s",
            {g["name"]: (len(g["params"]), g["lr"], g["weight_decay"]) for g in groups},
        )
        return groups


@dataclass
class FineTuneSchedule:
    """Declares which parts of the encoder are trainable at each epoch.

    Timeline for ``freeze_epochs=4, gradual_epochs=6``:

    ===========  ==========  ===================================================
    Epochs       Mode        What is training
    ===========  ==========  ===================================================
    1 - 4        ``frozen``  head only; encoder frozen and held in eval mode
    5 - 10       ``last``    head plus the encoder's final stage
    11+          ``all``     the whole network
    ===========  ==========  ===================================================

    With both values at 0 the schedule is a no-op and the whole model trains from
    the first epoch, which keeps the previous behaviour as the default.
    """

    freeze_epochs: int = 0
    gradual_epochs: int = 0

    @property
    def enabled(self) -> bool:
        """Whether the schedule does anything at all."""
        return self.freeze_epochs > 0 or self.gradual_epochs > 0

    def mode_for_epoch(self, epoch: int) -> str:
        """Trainability mode for a 1-indexed epoch."""
        if not self.enabled:
            return "all"
        if epoch <= self.freeze_epochs:
            return "frozen"
        if epoch <= self.freeze_epochs + self.gradual_epochs:
            return "last"
        return "all"

    def describe(self) -> str:
        """One-line summary for the training log."""
        if not self.enabled:
            return "no freezing; whole model trains from epoch 1"
        first = f"epochs 1-{self.freeze_epochs}: head only"
        if self.gradual_epochs:
            middle = (
                f"; {self.freeze_epochs + 1}-{self.freeze_epochs + self.gradual_epochs}: "
                "+ final encoder stage"
            )
        else:
            middle = ""
        return f"{first}{middle}; then the full network"


class WarmupCosineSchedule:
    """Linear warmup into cosine decay, stepped once per epoch.

    Each parameter group keeps its own base learning rate, so the discriminative
    backbone/head ratio is preserved throughout: both are scaled by the same
    factor rather than collapsed onto a single curve.

    Args:
        optimizer: Optimiser whose groups carry the base learning rates.
        total_epochs: Length of the full schedule.
        warmup_epochs: Epochs spent ramping linearly from ``min_ratio`` to 1.0.
        min_ratio: Floor of the cosine decay, as a fraction of the base rate.
    """

    def __init__(
        self,
        optimizer: Any,
        total_epochs: int,
        warmup_epochs: int = 0,
        min_ratio: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_ratio = float(min_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_factor = 1.0

    def factor_for_epoch(self, epoch: int) -> float:
        """Multiplier applied to every group's base learning rate.

        Epochs are 1-indexed. The first post-warmup epoch sits at the full rate
        and the final epoch at ``min_ratio``, so no epoch of the budget is spent
        below the intended peak or above the intended floor.
        """
        if epoch <= self.warmup_epochs and self.warmup_epochs > 0:
            # Ramp from min_ratio up to 1.0, reaching 1.0 on the last warmup epoch.
            progress = epoch / max(self.warmup_epochs, 1)
            return self.min_ratio + (1.0 - self.min_ratio) * progress
        # Decay spans the epochs after warmup; progress 0 on the first of them.
        decay_span = max(1, self.total_epochs - self.warmup_epochs - 1)
        progress = min(1.0, max(0.0, (epoch - self.warmup_epochs - 1) / decay_span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_ratio + (1.0 - self.min_ratio) * cosine

    def step(self, epoch: int) -> float:
        """Set the learning rate for a 1-indexed epoch and return the factor."""
        factor = self.factor_for_epoch(epoch)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor
        self.last_factor = factor
        return factor

    def current_learning_rates(self) -> dict[str, float]:
        """Current learning rate per named group, for the run history."""
        return {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(self.optimizer.param_groups)
        }

    def __iter__(self) -> Iterator[float]:
        """Iterate the factor for every epoch, for plotting the planned curve."""
        for epoch in range(1, self.total_epochs + 1):
            yield self.factor_for_epoch(epoch)
