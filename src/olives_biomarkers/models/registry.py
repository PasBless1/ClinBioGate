"""Model construction from configuration."""

from __future__ import annotations

from typing import Any

from olives_biomarkers.config import ModelConfig
from olives_biomarkers.models.baselines import (
    BaseBiomarkerModel,
    ClinicalOnlyModel,
    ConcatFusionModel,
    OCTOnlyModel,
)
from olives_biomarkers.models.fusion import GatedFusionModel


class ModelFactory:
    """Builds any model in the comparison from a :class:`ModelConfig`.

    Example:
        >>> model = ModelFactory().build(config.model, n_labels=16, clinical_dim=4)
    """

    REGISTRY: dict[str, type[BaseBiomarkerModel]] = {
        "clinical_only": ClinicalOnlyModel,
        "oct_only": OCTOnlyModel,
        "concat_fusion": ConcatFusionModel,
        "gated_fusion": GatedFusionModel,
    }

    def build(
        self,
        config: ModelConfig,
        n_labels: int,
        clinical_dim: int = 4,
        in_channels: int = 3,
    ) -> BaseBiomarkerModel:
        """Instantiate the model named by ``config.name``.

        Args:
            config: Model section of the experiment config.
            n_labels: Size of the target set (6 or 16).
            clinical_dim: Width of the clinical feature vector after preprocessing.
            in_channels: 3 to reuse ImageNet stems, 1 for raw grayscale.
        """
        if config.name not in self.REGISTRY:
            raise ValueError(
                f"unknown model {config.name!r}; available: {sorted(self.REGISTRY)}"
            )

        common: dict[str, Any] = {"n_labels": n_labels, "dropout": config.dropout}

        if config.name == "clinical_only":
            return ClinicalOnlyModel(
                clinical_dim=clinical_dim,
                hidden_dims=config.clinical_hidden_dims,
                embedding_dim=config.clinical_embedding_dim,
                **common,
            )
        if config.name == "oct_only":
            return OCTOnlyModel(
                backbone=config.image_encoder,
                pretrained=config.pretrained,
                embedding_dim=config.image_embedding_dim,
                in_channels=in_channels,
                **common,
            )

        fusion_kwargs: dict[str, Any] = {
            "clinical_dim": clinical_dim,
            "backbone": config.image_encoder,
            "pretrained": config.pretrained,
            "image_embedding_dim": config.image_embedding_dim,
            "clinical_embedding_dim": config.clinical_embedding_dim,
            "clinical_hidden_dims": config.clinical_hidden_dims,
            "in_channels": in_channels,
            **common,
        }
        if config.name == "concat_fusion":
            return ConcatFusionModel(**fusion_kwargs)
        return GatedFusionModel(
            gate_residual=config.gate_residual,
            gate_bias_init=config.gate_bias_init,
            gate_scale_alpha=config.gate_scale_alpha,
            **fusion_kwargs,
        )

    @classmethod
    def available(cls) -> list[str]:
        """Names accepted by :meth:`build`."""
        return sorted(cls.REGISTRY)
