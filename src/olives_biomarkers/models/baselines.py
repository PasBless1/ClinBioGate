"""Baseline models A, B and C.

Every model returns raw logits and exposes ``embeddings()`` so the same training,
uncertainty and analysis code works across the comparison.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from olives_biomarkers.models.encoders import ClinicalEncoder, ImageEncoder
from olives_biomarkers.models.heads import MultiLabelHead


class BaseBiomarkerModel(nn.Module):
    """Shared interface for every biomarker model in the study."""

    #: Which inputs ``forward`` actually consumes.
    uses_image: bool = True
    uses_clinical: bool = True

    def forward(self, image: torch.Tensor | None = None, clinical: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def embeddings(
        self, image: torch.Tensor | None = None, clinical: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Intermediate representations, for ablation and fusion analysis."""
        raise NotImplementedError

    def image_encoder_module(self) -> ImageEncoder | None:
        """The pretrained image encoder, or None for clinical-only models.

        Lets the trainer apply discriminative learning rates and progressive
        unfreezing without knowing which attribute each model stores it under.
        """
        for attribute in ("encoder", "image_encoder"):
            module = getattr(self, attribute, None)
            if isinstance(module, ImageEncoder):
                return module
        return None

    def n_parameters(self, trainable_only: bool = True) -> int:
        """Parameter count, reported alongside every result."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def describe(self) -> dict[str, Any]:
        """Compact description for run metadata."""
        return {
            "class": type(self).__name__,
            "uses_image": self.uses_image,
            "uses_clinical": self.uses_clinical,
            "n_trainable_parameters": self.n_parameters(True),
            "n_total_parameters": self.n_parameters(False),
        }


class ClinicalOnlyModel(BaseBiomarkerModel):
    """Model A: how much biomarker signal lives in BCVA and CST alone.

    A deliberately weak baseline. If it approaches the OCT model, the imaging
    contribution is smaller than it looks.
    """

    uses_image = False
    uses_clinical = True

    def __init__(
        self,
        clinical_dim: int,
        n_labels: int,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = ClinicalEncoder(
            input_dim=clinical_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.head = MultiLabelHead(embedding_dim, n_labels, dropout=dropout)

    def forward(self, image: torch.Tensor | None = None, clinical: torch.Tensor | None = None) -> torch.Tensor:
        if clinical is None:
            raise ValueError("ClinicalOnlyModel requires clinical features")
        return self.head(self.encoder(clinical))

    def embeddings(
        self, image: torch.Tensor | None = None, clinical: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        embedding = self.encoder(clinical)
        return {"clinical": embedding, "fused": embedding}


class OCTOnlyModel(BaseBiomarkerModel):
    """Model B: the imaging baseline every fusion claim is measured against."""

    uses_image = True
    uses_clinical = False

    def __init__(
        self,
        n_labels: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        in_channels: int = 3,
        pretrained_checkpoint: str | None = None,
        checkpoint_key: str = "model",
    ) -> None:
        super().__init__()
        self.encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            dropout=dropout,
            in_channels=in_channels,
            pretrained_checkpoint=pretrained_checkpoint,
            checkpoint_key=checkpoint_key,
        )
        self.head = MultiLabelHead(embedding_dim, n_labels, dropout=dropout)

    def forward(self, image: torch.Tensor | None = None, clinical: torch.Tensor | None = None) -> torch.Tensor:
        if image is None:
            raise ValueError("OCTOnlyModel requires images")
        return self.head(self.encoder(image))

    def embeddings(
        self, image: torch.Tensor | None = None, clinical: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        embedding = self.encoder(image)
        return {"image": embedding, "fused": embedding}

    @property
    def feature_layer(self) -> nn.Module:
        """Grad-CAM attachment point."""
        return self.encoder.feature_layer


class ConcatFusionModel(BaseBiomarkerModel):
    """Model C: plain feature-level concatenation of the two modalities.

    The control for Model D. If gating does not beat this, the gate is not
    earning its extra parameters.
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
        pretrained_checkpoint: str | None = None,
        checkpoint_key: str = "model",
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained,
            embedding_dim=image_embedding_dim,
            dropout=dropout,
            in_channels=in_channels,
            pretrained_checkpoint=pretrained_checkpoint,
            checkpoint_key=checkpoint_key,
        )
        self.clinical_encoder = ClinicalEncoder(
            input_dim=clinical_dim,
            hidden_dims=clinical_hidden_dims,
            embedding_dim=clinical_embedding_dim,
            dropout=dropout,
        )
        fused_dim = image_embedding_dim + clinical_embedding_dim
        self.head = MultiLabelHead(fused_dim, n_labels, hidden_dim=fused_dim // 2, dropout=dropout)

    def forward(self, image: torch.Tensor | None = None, clinical: torch.Tensor | None = None) -> torch.Tensor:
        if image is None or clinical is None:
            raise ValueError("ConcatFusionModel requires both image and clinical inputs")
        fused = torch.cat([self.image_encoder(image), self.clinical_encoder(clinical)], dim=1)
        return self.head(fused)

    def embeddings(
        self, image: torch.Tensor | None = None, clinical: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        image_embedding = self.image_encoder(image)
        clinical_embedding = self.clinical_encoder(clinical)
        return {
            "image": image_embedding,
            "clinical": clinical_embedding,
            "fused": torch.cat([image_embedding, clinical_embedding], dim=1),
        }

    @property
    def feature_layer(self) -> nn.Module:
        """Grad-CAM attachment point."""
        return self.image_encoder.feature_layer
