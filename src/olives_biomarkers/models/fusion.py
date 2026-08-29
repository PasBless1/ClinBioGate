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
        self._last_gate_scale = self.gate.scale(gate)
        fused = torch.cat([gated, clinical_embedding], dim=1)
        return {
            "image": image_embedding,
            "clinical": clinical_embedding,
            "gate": gate,
            "gated_image": gated,
            "fused": fused,
        }

    def regularization_loss(self) -> torch.Tensor:
        """Identity penalty used to keep the legacy multiplicative gate bounded."""
        scale = getattr(self, "_last_gate_scale", None)
        if scale is None:
            return next(self.parameters()).new_zeros(())
        return (scale - 1.0).pow(2).mean()

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


class BoundedFiLMFusionModel(BaseBiomarkerModel):
    """Clinical modulation with bounded scale and shift around OCT identity."""

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
        max_scale: float = 0.25,
        max_shift: float = 0.25,
        pretrained_checkpoint: str | None = None,
        checkpoint_key: str = "model",
    ) -> None:
        super().__init__()
        self.max_scale = max_scale
        self.max_shift = max_shift
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
        self.film = nn.Linear(clinical_embedding_dim, image_embedding_dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.head = MultiLabelHead(
            image_embedding_dim,
            n_labels,
            hidden_dim=image_embedding_dim // 2,
            dropout=dropout,
        )

    def _fuse(self, image: torch.Tensor, clinical: torch.Tensor) -> dict[str, torch.Tensor]:
        image_embedding = self.image_encoder(image)
        clinical_embedding = self.clinical_encoder(clinical)
        raw_scale, raw_shift = self.film(clinical_embedding).chunk(2, dim=1)
        scale = 1.0 + self.max_scale * torch.tanh(raw_scale)
        shift = self.max_shift * torch.tanh(raw_shift)
        fused = image_embedding * scale + shift
        return {
            "image": image_embedding,
            "clinical": clinical_embedding,
            "scale": scale,
            "shift": shift,
            "fused": fused,
        }

    def forward(
        self,
        image: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image is None or clinical is None:
            raise ValueError("BoundedFiLMFusionModel requires image and clinical inputs")
        return self.head(self._fuse(image, clinical)["fused"])

    def embeddings(
        self,
        image: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if image is None or clinical is None:
            raise ValueError("BoundedFiLMFusionModel requires image and clinical inputs")
        return self._fuse(image, clinical)

    @property
    def feature_layer(self) -> nn.Module:
        return self.image_encoder.feature_layer


class ResidualLogitFusionModel(BaseBiomarkerModel):
    """Clinical evidence added in *logit* space, not feature space.

    ``logits = oct_logits + beta * clinical_logits``

    The gated and FiLM models both modulate the OCT *representation*, which gives
    clinical features a large, diffuse influence over every downstream channel.
    The A100 run showed what that costs: the learned gate amplified 97.78% of
    channels with a mean scale of 1.897 out of 2, behaving as near-global gain
    rather than selective modulation, and it did not beat the OCT baseline.

    This design constrains clinical input to the narrowest channel that can still
    carry the hypothesis: a per-biomarker additive correction to the OCT decision.

    Two properties make it a strictly safer bet than the gate:

    * **It starts as the OCT baseline exactly.** ``beta`` is initialised to zero,
      so at step one the model's output is identical to Model B's. Any departure
      has to be earned against validation.
    * **The correction is bounded.** The effective coefficient is
      ``max_scale * tanh(beta)``, so a single biomarker's clinical term cannot run
      away and swamp the image evidence.

    ``per_label=True`` gives each biomarker its own coefficient, which is what
    makes the result readable: CST should be allowed to inform DRT/ME, IRF and
    SRF while leaving the vitreous-face labels near zero. The fitted
    :meth:`beta_values` are the direct test of that prediction.
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
        max_scale: float = 1.0,
        per_label: bool = True,
        pretrained_checkpoint: str | None = None,
        checkpoint_key: str = "model",
    ) -> None:
        super().__init__()
        self.n_labels = n_labels
        self.max_scale = max_scale
        self.per_label = per_label

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
        self.oct_head = MultiLabelHead(image_embedding_dim, n_labels, dropout=dropout)
        self.clinical_head = MultiLabelHead(clinical_embedding_dim, n_labels, dropout=dropout)

        # Zero init => tanh(0) = 0 => the clinical branch contributes nothing at
        # the first step, so training begins from the OCT baseline exactly.
        self.beta = nn.Parameter(torch.zeros(n_labels if per_label else 1))

    # ------------------------------------------------------------------
    def effective_beta(self) -> torch.Tensor:
        """Bounded per-label coefficient actually applied to clinical logits."""
        return self.max_scale * torch.tanh(self.beta)

    def _fuse(self, image: torch.Tensor, clinical: torch.Tensor) -> dict[str, torch.Tensor]:
        image_embedding = self.image_encoder(image)
        clinical_embedding = self.clinical_encoder(clinical)
        oct_logits = self.oct_head(image_embedding)
        clinical_logits = self.clinical_head(clinical_embedding)
        beta = self.effective_beta()
        return {
            "image": image_embedding,
            "clinical": clinical_embedding,
            "oct_logits": oct_logits,
            "clinical_logits": clinical_logits,
            "beta": beta,
            "fused": oct_logits + beta * clinical_logits,
        }

    def forward(
        self,
        image: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image is None or clinical is None:
            raise ValueError("ResidualLogitFusionModel requires image and clinical inputs")
        return self._fuse(image, clinical)["fused"]

    def embeddings(
        self,
        image: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if image is None or clinical is None:
            raise ValueError("ResidualLogitFusionModel requires image and clinical inputs")
        return self._fuse(image, clinical)

    @property
    def feature_layer(self) -> nn.Module:
        """Grad-CAM attaches to the OCT branch, which still carries the imaging path."""
        return self.image_encoder.feature_layer

    def regularization_loss(self) -> torch.Tensor:
        """L1 on beta, so clinical terms must earn their place.

        Wired through ``training.model_regularization_weight``; left at 0 it has
        no effect.
        """
        return self.effective_beta().abs().mean()

    @torch.no_grad()
    def regularization_value(self) -> float:
        """Detached view of :meth:`regularization_loss`, for logging."""
        return float(self.effective_beta().abs().mean())

    @torch.no_grad()
    def beta_values(self, label_names: list[str] | None = None) -> dict[str, float]:
        """Fitted clinical coefficient per biomarker.

        This is the interpretable output of the model. A coefficient near zero
        means the model chose to ignore clinical input for that biomarker; a
        large one means BCVA/CST shifted its decision. Reporting these is more
        informative than a macro metric that moved by 0.005.
        """
        beta = self.effective_beta().detach().cpu().numpy().ravel()
        if not self.per_label:
            beta = beta.repeat(self.n_labels)
        names = label_names or [f"label_{i}" for i in range(self.n_labels)]
        return {name: float(value) for name, value in zip(names, beta)}

    @torch.no_grad()
    def contribution_summary(
        self, image: torch.Tensor, clinical: torch.Tensor
    ) -> dict[str, float]:
        """How much of the final logit the clinical branch actually supplies."""
        parts = self._fuse(image, clinical)
        oct_magnitude = parts["oct_logits"].abs().mean()
        clinical_magnitude = (parts["beta"] * parts["clinical_logits"]).abs().mean()
        total = oct_magnitude + clinical_magnitude
        return {
            "mean_abs_oct_logit": float(oct_magnitude),
            "mean_abs_clinical_term": float(clinical_magnitude),
            "clinical_share": float(clinical_magnitude / total) if float(total) > 0 else 0.0,
            "max_abs_beta": float(parts["beta"].abs().max()),
        }
