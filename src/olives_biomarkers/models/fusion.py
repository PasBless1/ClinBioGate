"""Model D: the proposed clinically gated fusion model."""

from __future__ import annotations

import torch
from torch import nn

from olives_biomarkers.models.baselines import BaseBiomarkerModel
from olives_biomarkers.models.encoders import ClinicalEncoder, ImageEncoder
from olives_biomarkers.models.heads import MultiLabelHead


class ClinicalGate(nn.Module):
    """Produces a per-channel multiplicative gate from the clinical embedding.

    The failure mode this guards against is a gate that collapses toward zero
    early in training and erases the OCT signal before the image encoder has
    learned anything. Both modes therefore start as an **exact identity**: the
    gate's output layer has zeroed weights, so at initialisation every sample
    receives the same gate value determined by ``bias_init``.

    * ``residual=True`` (default): the applied scale is
      ``1 + alpha * (2 * gate - 1)``, which equals 1 when ``gate == 0.5``. The
      matching ``bias_init`` is therefore **0.0**. The scale is centred on 1 and
      confined to ``[1 - alpha, 1 + alpha]``, so modulation is symmetric —
      clinical context can damp or amplify an image feature by the same factor.
    * ``residual=False``: the scale is the raw gate, so identity needs
      ``sigmoid(bias) -> 1`` and ``bias_init`` should be large and positive
      (2.0 gives 0.88, 4.0 gives 0.98).

    :meth:`default_bias_init` returns the identity-preserving bias for a mode.

    Args:
        clinical_dim: Width of the clinical embedding.
        image_dim: Width of the image embedding being modulated.
        bias_init: Initial bias of the gate's output layer.
        residual: Use the residual formulation rather than a raw sigmoid.
        alpha: Modulation strength in residual mode.
    """

    def __init__(
        self,
        clinical_dim: int,
        image_dim: int,
        bias_init: float | None = None,
        residual: bool = True,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.alpha = alpha
        self.bias_init = (
            self.default_bias_init(residual) if bias_init is None else float(bias_init)
        )
        self.projection = nn.Sequential(
            nn.Linear(clinical_dim, image_dim),
            nn.ReLU(inplace=True),
            nn.Linear(image_dim, image_dim),
        )
        final = self.projection[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, self.bias_init)

    @staticmethod
    def default_bias_init(residual: bool) -> float:
        """Bias that makes the gate an identity at initialisation."""
        return 0.0 if residual else 4.0

    def forward(self, clinical_embedding: torch.Tensor) -> torch.Tensor:
        """Return gate values in ``(0, 1)`` of shape ``(batch, image_dim)``."""
        return torch.sigmoid(self.projection(clinical_embedding))

    def scale(self, gate: torch.Tensor) -> torch.Tensor:
        """Multiplicative factor actually applied to the image embedding."""
        if not self.residual:
            return gate
        return 1.0 + self.alpha * (2.0 * gate - 1.0)

    def modulate(self, image_embedding: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Apply the gate to the image embedding."""
        return image_embedding * self.scale(gate)


class GatedFusionModel(BaseBiomarkerModel):
    """Model D: clinical measurements modulate OCT features before classification.

    The hypothesis is that BCVA and CST tell the model *how to read* the scan
    rather than what the answer is: a high CST should raise the weight on
    fluid-related image features. Concatenation (Model C) cannot express that
    interaction; a multiplicative gate can.

    Both the gated and ungated image embeddings reach the head, so the model can
    always fall back on unmodulated OCT features.
    """

    uses_image = True
    uses_clinical = True

    def __init__(
        self,
        clinical_dim: int,
        n_labels: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        image_embedding_dim: int = 256,
        clinical_embedding_dim: int = 32,
        clinical_hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        in_channels: int = 3,
        gate_residual: bool = True,
        gate_bias_init: float | None = None,
        gate_scale_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained,
            embedding_dim=image_embedding_dim,
            dropout=dropout,
            in_channels=in_channels,
        )
        self.clinical_encoder = ClinicalEncoder(
            input_dim=clinical_dim,
            hidden_dims=clinical_hidden_dims,
            embedding_dim=clinical_embedding_dim,
            dropout=dropout,
        )
        self.gate = ClinicalGate(
            clinical_dim=clinical_embedding_dim,
            image_dim=image_embedding_dim,
            bias_init=gate_bias_init,
            residual=gate_residual,
            alpha=gate_scale_alpha,
        )
        fused_dim = image_embedding_dim + clinical_embedding_dim
        self.head = MultiLabelHead(fused_dim, n_labels, hidden_dim=fused_dim // 2, dropout=dropout)

    def _fuse(self, image: torch.Tensor, clinical: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run both encoders, gate the image embedding and concatenate."""
        image_embedding = self.image_encoder(image)
        clinical_embedding = self.clinical_encoder(clinical)
        gate = self.gate(clinical_embedding)
        gated = self.gate.modulate(image_embedding, gate)
        fused = torch.cat([gated, clinical_embedding], dim=1)
        return {
            "image": image_embedding,
            "clinical": clinical_embedding,
            "gate": gate,
            "gated_image": gated,
            "fused": fused,
        }

    def forward(self, image: torch.Tensor | None = None, clinical: torch.Tensor | None = None) -> torch.Tensor:
        if image is None or clinical is None:
            raise ValueError("GatedFusionModel requires both image and clinical inputs")
        return self.head(self._fuse(image, clinical)["fused"])

    def embeddings(
        self, image: torch.Tensor | None = None, clinical: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Image, clinical, gate, gated-image and fused representations."""
        return self._fuse(image, clinical)

    @property
    def feature_layer(self) -> nn.Module:
        """Grad-CAM attachment point."""
        return self.image_encoder.feature_layer

    @torch.no_grad()
    def gate_statistics(self, clinical: torch.Tensor) -> dict[str, float]:
        """Summary statistics of the gate and the scale it actually applies.

        ``scale_mean`` is the diagnostic that matters: 1.0 means the image
        embedding passes through untouched, and values near 0 mean clinical
        features are suppressing the OCT signal.
        """
        self.eval()
        gate = self.gate(self.clinical_encoder(clinical))
        scale = self.gate.scale(gate)
        return {
            "gate_mean": float(gate.mean()),
            "gate_std": float(gate.std()),
            "gate_min": float(gate.min()),
            "gate_max": float(gate.max()),
            "scale_mean": float(scale.mean()),
            "scale_std": float(scale.std()),
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
            "fraction_scale_below_0.1": float((scale.abs() < 0.1).float().mean()),
        }
