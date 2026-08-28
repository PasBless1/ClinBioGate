"""Grad-CAM explanations for OCT encoders.

One heatmap per biomarker, not one generic saliency map: the question is whether
the evidence for *intraretinal fluid* sits on plausible retinal structure, and a
class-agnostic map cannot answer that.

Explanations are supporting evidence about where a model looked. They are not
evidence that the model reasoned correctly, and they are not causal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from olives_biomarkers.utils.logging import LoggerFactory

LOGGER = LoggerFactory.get("olives.explainability")


@dataclass
class CamResult:
    """One Grad-CAM heatmap and the context needed to read it."""

    heatmap: np.ndarray
    label_index: int
    label_name: str
    probability: float
    target: float | None = None

    @property
    def outcome(self) -> str:
        """TP / TN / FP / FN classification at a 0.5 threshold."""
        if self.target is None:
            return "unknown"
        predicted = self.probability >= 0.5
        actual = self.target >= 0.5
        if predicted and actual:
            return "true_positive"
        if not predicted and not actual:
            return "true_negative"
        return "false_positive" if predicted else "false_negative"


class GradCAM:
    """Gradient-weighted class activation mapping on the last conv block.

    Args:
        model: A model exposing ``feature_layer``.
        target_layer: Override the layer to hook.
        device: Torch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        layer = target_layer if target_layer is not None else getattr(model, "feature_layer", None)
        if layer is None:
            raise ValueError(
                "model exposes no `feature_layer`; pass target_layer explicitly "
                "(clinical-only models have no image features to explain)"
            )
        self.target_layer = layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[Any] = []

    # ------------------------------------------------------------------
    def _forward_hook(self, _module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _backward_hook(self, _module: nn.Module, _grad_in: Any, grad_out: tuple[torch.Tensor, ...]) -> None:
        self._gradients = grad_out[0].detach()

    def __enter__(self) -> GradCAM:
        self._handles = [
            self.target_layer.register_forward_hook(self._forward_hook),
            self.target_layer.register_full_backward_hook(self._backward_hook),
        ]
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove_hooks()

    def remove_hooks(self) -> None:
        """Detach the forward/backward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------------
    def generate(
        self,
        image: torch.Tensor,
        label_index: int,
        clinical: torch.Tensor | None = None,
        label_name: str | None = None,
        target: float | None = None,
    ) -> CamResult:
        """Produce the heatmap for one image and one biomarker.

        Args:
            image: ``(1, C, H, W)`` tensor.
            label_index: Which biomarker to explain.
            clinical: Required by fusion models.
            target: Ground-truth value, used to label the outcome.
        """
        if not self._handles:
            raise RuntimeError("use GradCAM as a context manager, or call __enter__ first")

        self.model.eval()
        image = image.to(self.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if clinical is not None:
            clinical = clinical.to(self.device)
            if clinical.dim() == 1:
                clinical = clinical.unsqueeze(0)

        self.model.zero_grad(set_to_none=True)
        kwargs: dict[str, Any] = {"image": image}
        if getattr(self.model, "uses_clinical", False):
            kwargs["clinical"] = clinical
        logits = self.model(**kwargs)

        score = logits[0, label_index]
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("hooks captured nothing; check the target layer")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        span = cam.max() - cam.min()
        cam = (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)

        return CamResult(
            heatmap=cam,
            label_index=label_index,
            label_name=label_name or f"label_{label_index}",
            probability=float(torch.sigmoid(logits[0, label_index]).detach()),
            target=target,
        )

    def generate_all_labels(
        self,
        image: torch.Tensor,
        label_names: list[str],
        clinical: torch.Tensor | None = None,
        targets: np.ndarray | None = None,
    ) -> list[CamResult]:
        """One heatmap per biomarker for a single scan."""
        return [
            self.generate(
                image,
                index,
                clinical=clinical,
                label_name=name,
                target=float(targets[index]) if targets is not None else None,
            )
            for index, name in enumerate(label_names)
        ]


class AttentionSanityChecker:
    """Flags heatmaps that concentrate on borders or empty background.

    A map whose mass sits in the image margin is attending to padding, vignetting
    or burnt-in acquisition text rather than retina, and any clinical reading of
    it would be wrong.
    """

    def __init__(self, border_fraction: float = 0.10, intensity_threshold: float = 0.05) -> None:
        self.border_fraction = border_fraction
        self.intensity_threshold = intensity_threshold

    def border_mass(self, heatmap: np.ndarray) -> float:
        """Fraction of total activation lying in the image border band."""
        height, width = heatmap.shape
        band_h = max(1, int(self.border_fraction * height))
        band_w = max(1, int(self.border_fraction * width))
        mask = np.zeros_like(heatmap, dtype=bool)
        mask[:band_h, :] = mask[-band_h:, :] = True
        mask[:, :band_w] = mask[:, -band_w:] = True
        total = heatmap.sum()
        return float(heatmap[mask].sum() / total) if total > 1e-8 else 0.0

    def background_mass(self, heatmap: np.ndarray, image: np.ndarray) -> float:
        """Fraction of activation over near-black (non-tissue) pixels."""
        image = np.asarray(image)
        if image.ndim == 3:
            channel_axis = 0 if image.shape[0] in {1, 3, 4} else -1
            image = image.mean(axis=channel_axis)
        if image.ndim != 2:
            raise ValueError(f'image must be 2D or 3D, got shape {image.shape}')
        if image.shape != heatmap.shape:
            resized = torch.as_tensor(image, dtype=torch.float32)[None, None]
            image = torch.nn.functional.interpolate(
                resized, size=heatmap.shape, mode='bilinear', align_corners=False
            )[0, 0].numpy()
        normalized = (image - image.min()) / max(image.max() - image.min(), 1e-8)
        background = normalized < self.intensity_threshold
        total = heatmap.sum()
        return float(heatmap[background].sum() / total) if total > 1e-8 else 0.0

    def check(self, result: CamResult, image: np.ndarray | None = None) -> dict[str, Any]:
        """Summarise where a heatmap's mass sits and whether that is suspicious."""
        border = self.border_mass(result.heatmap)
        report: dict[str, Any] = {
            "label": result.label_name,
            "outcome": result.outcome,
            "probability": result.probability,
            "border_mass": border,
            "suspicious_border": border > 0.35,
        }
        if image is not None:
            background = self.background_mass(result.heatmap, image)
            report["background_mass"] = background
            report["suspicious_background"] = background > 0.35
        return report
