"""Multilabel losses.

All losses consume **raw logits**. Nothing here applies a sigmoid before the
loss; doing so would silently break the numerically stable formulations.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MaskedBCEWithLogitsLoss(nn.Module):
    """Weighted BCE that ignores missing labels.

    OLIVES annotates biomarkers only on the first and last visit, and a handful
    of rows carry an incomplete vector. Passing a mask keeps those entries out of
    the gradient rather than treating an unknown as a negative.

    Args:
        pos_weight: Per-label positive class weights from the training fold.
        reduction: ``"mean"`` over unmasked entries, or ``"none"``.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | np.ndarray | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction
        if pos_weight is not None:
            weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss over unmasked entries.

        Args:
            logits: Raw model outputs, ``(batch, n_labels)``.
            targets: Binary targets, same shape.
            mask: Optional boolean mask; True marks an observed label.
        """
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        if mask is not None:
            losses = losses * mask.float()
            if self.reduction == "mean":
                denominator = mask.float().sum().clamp(min=1.0)
                return losses.sum() / denominator
        if self.reduction == "mean":
            return losses.mean()
        return losses


class FocalLoss(nn.Module):
    """Focal loss for multilabel targets, offered strictly as an ablation.

    The brief's order is deliberate: get weighted BCE working first, then test
    whether focusing helps. Reaching for focal loss early tends to hide a
    thresholding problem rather than fix an imbalance problem.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute focal loss over unmasked entries."""
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        losses = alpha_t * (1 - p_t).pow(self.gamma) * bce

        if mask is not None:
            losses = losses * mask.float()
            if self.reduction == "mean":
                return losses.sum() / mask.float().sum().clamp(min=1.0)
        if self.reduction == "mean":
            return losses.mean()
        return losses


class AsymmetricLoss(nn.Module):
    """Asymmetric multilabel loss with easy-negative probability clipping."""

    def __init__(
        self,
        gamma_negative: float = 4.0,
        gamma_positive: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positive = torch.sigmoid(logits)
        negative = 1.0 - positive
        if self.clip > 0:
            negative = (negative + self.clip).clamp(max=1.0)

        log_likelihood = targets * torch.log(positive.clamp(min=self.eps))
        log_likelihood += (1.0 - targets) * torch.log(negative.clamp(min=self.eps))
        gamma = self.gamma_positive * targets + self.gamma_negative * (1.0 - targets)
        base = 1.0 - positive * targets - negative * (1.0 - targets)
        losses = -log_likelihood * base.clamp(min=0.0).pow(gamma)
        if mask is not None:
            losses = losses * mask.float()
            if self.reduction == "mean":
                return losses.sum() / mask.float().sum().clamp(min=1.0)
        return losses.mean() if self.reduction == "mean" else losses


class ExclusivityRegularizedLoss(nn.Module):
    """Add a probability-overlap penalty for verified exclusive label pairs."""

    def __init__(
        self,
        base_loss: nn.Module,
        exclusive_pairs: list[tuple[int, int]],
        coefficient: float,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.exclusive_pairs = exclusive_pairs
        self.coefficient = coefficient

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.base_loss(logits, targets, mask)
        if not self.exclusive_pairs or self.coefficient <= 0:
            return loss
        probabilities = torch.sigmoid(logits)
        overlap = torch.stack(
            [probabilities[:, left] * probabilities[:, right]
             for left, right in self.exclusive_pairs],
            dim=1,
        )
        return loss + self.coefficient * overlap.mean()


class LossFactory:
    """Builds the configured loss with training-fold class weights."""

    REGISTRY = {
        "bce": MaskedBCEWithLogitsLoss,
        "focal": FocalLoss,
        "asl": AsymmetricLoss,
    }

    def build(
        self,
        name: str = "bce",
        pos_weight: np.ndarray | None = None,
        **kwargs: object,
    ) -> nn.Module:
        """Instantiate a loss by name.

        Args:
            name: ``"bce"`` or ``"focal"``.
            pos_weight: Per-label weights; ignored by focal loss.
        """
        if name not in self.REGISTRY:
            raise ValueError(f"unknown loss {name!r}; available: {sorted(self.REGISTRY)}")
        if name == "bce":
            return MaskedBCEWithLogitsLoss(pos_weight=pos_weight, **kwargs)  # type: ignore[arg-type]
        return self.REGISTRY[name](**kwargs)  # type: ignore[call-arg]
