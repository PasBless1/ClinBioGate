"""Model construction from configuration.

Every architecture in the comparison is built from the same :class:`ModelConfig`,
so a config file is a complete description of the model and nothing is set by an
argument the config cannot express.
"""

from __future__ import annotations

from typing import Any

from olives_biomarkers.config import ModelConfig
from olives_biomarkers.models.baselines import (
    BaseBiomarkerModel,
    ClinicalOnlyModel,
    ConcatFusionModel,
    OCTOnlyModel,
)
from olives_biomarkers.models.fusion import (
    BoundedFiLMFusionModel,
    GatedFusionModel,
    ResidualLogitFusionModel,
)


class ModelFactory:
    """Builds any model in the comparison from a :class:`ModelConfig`.

    Fusion variants, in increasing order of how much influence clinical features
    are given over the OCT representation:

    ============================  ====================================================
    Name                          Clinical influence
    ============================  ====================================================
    ``concat_fusion``             Additive, in feature space, via a shared head
    ``residual_logit_fusion``     Additive, in *logit* space, one bounded coefficient
                                  per biomarker; starts as the OCT baseline exactly
    ``bounded_film_fusion``       Multiplicative scale + shift, bounded by ``tanh``
    ``gated_fusion``              Multiplicative gate over every channel (unbounded
                                  toward 2x); the A100 run showed this behaves as
                                  near-global amplification
    ============================  ====================================================

    Example:
        >>> model = ModelFactory().build(config.model, n_labels=16, clinical_dim=4)
    """

    REGISTRY: dict[str, type[BaseBiomarkerModel]] = {
        "clinical_only": ClinicalOnlyModel,
        "oct_only": OCTOnlyModel,
        "concat_fusion": ConcatFusionModel,
        "gated_fusion": GatedFusionModel,
        "bounded_film_fusion": BoundedFiLMFusionModel,
        "residual_logit_fusion": ResidualLogitFusionModel,
    }

    #: Models that consume images and therefore accept encoder options.
    IMAGE_MODELS = frozenset(REGISTRY) - {"clinical_only"}

    def build(
        self,
        config: ModelConfig,
        n_labels: int,
        clinical_dim: int = 4,
        in_channels: int | None = None,
    ) -> BaseBiomarkerModel:
        """Instantiate the model named by ``config.name``.

        Args:
            config: Model section of the experiment config.
            n_labels: Size of the target set (6 or 16).
            clinical_dim: Width of the clinical feature vector after preprocessing.
            in_channels: Overrides ``config.in_channels``; 3 for repeated grayscale
                or adjacent-slice input, 1 for a single raw channel.
        """
        if config.name not in self.REGISTRY:
            raise ValueError(
                f"unknown model {config.name!r}; available: {sorted(self.REGISTRY)}"
            )
        channels = config.in_channels if in_channels is None else in_channels

        common: dict[str, Any] = {"n_labels": n_labels, "dropout": config.dropout}

        if config.name == "clinical_only":
            return ClinicalOnlyModel(
                clinical_dim=clinical_dim,
                hidden_dims=config.clinical_hidden_dims,
                embedding_dim=config.clinical_embedding_dim,
                **common,
            )

        # Encoder options shared by every image model, including the domain
        # checkpoint used for RETFound weights.
        encoder_kwargs: dict[str, Any] = {
            "backbone": config.image_encoder,
            "pretrained": config.pretrained,
            "in_channels": channels,
            "pretrained_checkpoint": config.pretrained_checkpoint,
            "checkpoint_key": config.checkpoint_key,
        }

        if config.name == "oct_only":
            return OCTOnlyModel(
                embedding_dim=config.image_embedding_dim, **encoder_kwargs, **common
            )

        fusion_kwargs: dict[str, Any] = {
            "clinical_dim": clinical_dim,
            "image_embedding_dim": config.image_embedding_dim,
            "clinical_embedding_dim": config.clinical_embedding_dim,
            "clinical_hidden_dims": config.clinical_hidden_dims,
            **encoder_kwargs,
            **common,
        }
        if config.name == "concat_fusion":
            return ConcatFusionModel(**fusion_kwargs)
        if config.name == "bounded_film_fusion":
            return BoundedFiLMFusionModel(
                max_scale=config.film_max_scale,
                max_shift=config.film_max_shift,
                **fusion_kwargs,
            )
        if config.name == "residual_logit_fusion":
            return ResidualLogitFusionModel(
                max_scale=config.clinical_residual_max_scale,
                per_label=config.clinical_residual_per_label,
                **fusion_kwargs,
            )
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
