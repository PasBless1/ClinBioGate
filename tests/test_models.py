"""Model shape, gating and optimisation-sanity tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from olives_biomarkers.config import ModelConfig  # noqa: E402
from olives_biomarkers.models import (  # noqa: E402
    ClinicalOnlyModel,
    ConcatFusionModel,
    GatedFusionModel,
    ModelFactory,
    OCTOnlyModel,
)
from olives_biomarkers.training.losses import LossFactory, MaskedBCEWithLogitsLoss  # noqa: E402

BATCH = 4
CLINICAL_DIM = 4
IMAGE = (BATCH, 3, 64, 64)


def _inputs(n_labels: int = 16):
    return (
        torch.randn(*IMAGE),
        torch.randn(BATCH, CLINICAL_DIM),
        torch.randint(0, 2, (BATCH, n_labels)).float(),
    )


class TestOutputShapes:
    """Every model must emit (batch, n_labels) for both target sets."""

    @pytest.mark.parametrize("n_labels", [6, 16])
    def test_clinical_only(self, n_labels: int) -> None:
        model = ClinicalOnlyModel(clinical_dim=CLINICAL_DIM, n_labels=n_labels)
        _, clinical, _ = _inputs(n_labels)
        assert model(clinical=clinical).shape == (BATCH, n_labels)

    @pytest.mark.parametrize("n_labels", [6, 16])
    def test_oct_only(self, n_labels: int) -> None:
        model = OCTOnlyModel(n_labels=n_labels, pretrained=False)
        image, _, _ = _inputs(n_labels)
        assert model(image=image).shape == (BATCH, n_labels)

    @pytest.mark.parametrize("n_labels", [6, 16])
    def test_concat_fusion(self, n_labels: int) -> None:
        model = ConcatFusionModel(clinical_dim=CLINICAL_DIM, n_labels=n_labels, pretrained=False)
        image, clinical, _ = _inputs(n_labels)
        assert model(image=image, clinical=clinical).shape == (BATCH, n_labels)

    @pytest.mark.parametrize("n_labels", [6, 16])
    def test_gated_fusion(self, n_labels: int) -> None:
        model = GatedFusionModel(clinical_dim=CLINICAL_DIM, n_labels=n_labels, pretrained=False)
        image, clinical, _ = _inputs(n_labels)
        assert model(image=image, clinical=clinical).shape == (BATCH, n_labels)

    def test_single_channel_stem_is_supported(self) -> None:
        model = OCTOnlyModel(n_labels=6, pretrained=False, in_channels=1)
        assert model(image=torch.randn(BATCH, 1, 64, 64)).shape == (BATCH, 6)


class TestRawLogits:
    """The head must not squash its output; BCEWithLogitsLoss needs logits."""

    def test_outputs_leave_the_zero_one_range(self) -> None:
        model = OCTOnlyModel(n_labels=16, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(image=torch.randn(64, 3, 64, 64) * 6)
        assert logits.min() < 0.0, "outputs look like probabilities, not logits"

    def test_loss_accepts_raw_logits(self) -> None:
        model = ClinicalOnlyModel(clinical_dim=CLINICAL_DIM, n_labels=16)
        _, clinical, targets = _inputs()
        loss = MaskedBCEWithLogitsLoss()(model(clinical=clinical), targets)
        assert torch.isfinite(loss)


class TestModelFactory:
    """Configuration-driven construction."""

    @pytest.mark.parametrize("name", ModelFactory.available())
    def test_builds_every_registered_model(self, name: str) -> None:
        config = ModelConfig(name=name, pretrained=False)
        model = ModelFactory().build(config, n_labels=6, clinical_dim=CLINICAL_DIM)
        assert model.n_parameters() > 0
        assert "class" in model.describe()

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown model"):
            ModelFactory().build(ModelConfig(name="nope"), n_labels=6)


class TestEmbeddings:
    """Intermediate representations must be exposed for analysis."""

    def test_gated_fusion_exposes_all_stages(self) -> None:
        model = GatedFusionModel(clinical_dim=CLINICAL_DIM, n_labels=16, pretrained=False)
        image, clinical, _ = _inputs()
        embeddings = model.embeddings(image=image, clinical=clinical)
        assert set(embeddings) == {"image", "clinical", "gate", "gated_image", "fused"}
        assert embeddings["gate"].shape == embeddings["image"].shape

    def test_concat_fusion_exposes_both_modalities(self) -> None:
        model = ConcatFusionModel(clinical_dim=CLINICAL_DIM, n_labels=16, pretrained=False)
        image, clinical, _ = _inputs()
        embeddings = model.embeddings(image=image, clinical=clinical)
        assert embeddings["fused"].shape[1] == (
            embeddings["image"].shape[1] + embeddings["clinical"].shape[1]
        )


class TestClinicalGate:
    """The gate must not erase the OCT signal at initialisation."""

    @pytest.mark.parametrize("residual", [True, False])
    def test_applied_scale_starts_at_identity(self, residual: bool) -> None:
        """Whatever the mode, the default bias must make the gate a pass-through."""
        model = GatedFusionModel(
            clinical_dim=CLINICAL_DIM, n_labels=16, pretrained=False, gate_residual=residual
        )
        stats = model.gate_statistics(torch.randn(32, CLINICAL_DIM))
        assert stats["scale_mean"] == pytest.approx(1.0, abs=0.05 if residual else 0.02)
        assert stats["scale_std"] == pytest.approx(0.0, abs=1e-5)
        assert stats["fraction_scale_below_0.1"] == 0.0

    @pytest.mark.parametrize("residual", [True, False])
    def test_gating_preserves_the_image_embedding_at_init(self, residual: bool) -> None:
        model = GatedFusionModel(
            clinical_dim=CLINICAL_DIM, n_labels=16, pretrained=False, gate_residual=residual
        )
        model.eval()
        image, clinical, _ = _inputs()
        with torch.no_grad():
            embeddings = model.embeddings(image=image, clinical=clinical)
        torch.testing.assert_close(
            embeddings["gated_image"], embeddings["image"], rtol=0.05, atol=0.05
        )

    def test_default_bias_matches_the_mode(self) -> None:
        from olives_biomarkers.models.fusion import ClinicalGate

        assert ClinicalGate.default_bias_init(residual=True) == 0.0
        assert ClinicalGate.default_bias_init(residual=False) > 0.0

    def test_gate_output_is_bounded(self) -> None:
        model = GatedFusionModel(clinical_dim=CLINICAL_DIM, n_labels=6, pretrained=False)
        with torch.no_grad():
            gate = model.gate(model.clinical_encoder(torch.randn(16, CLINICAL_DIM)))
        assert gate.min() >= 0.0 and gate.max() <= 1.0

    def test_residual_scale_is_symmetric_about_one(self) -> None:
        from olives_biomarkers.models.fusion import ClinicalGate

        gate = ClinicalGate(CLINICAL_DIM, 8, residual=True, alpha=1.0)
        closed = gate.scale(torch.zeros(1, 8))
        open_ = gate.scale(torch.ones(1, 8))
        assert closed.mean() == pytest.approx(0.0)
        assert open_.mean() == pytest.approx(2.0)

    def test_gate_can_learn_away_from_identity(self) -> None:
        """The zeroed init must not be a dead end -- gradients have to flow."""
        model = GatedFusionModel(clinical_dim=CLINICAL_DIM, n_labels=6, pretrained=False)
        image, clinical, targets = _inputs(6)
        MaskedBCEWithLogitsLoss()(model(image=image, clinical=clinical), targets).backward()
        gate_grad = model.gate.projection[-1].weight.grad
        assert gate_grad is not None and gate_grad.abs().sum() > 0


class TestOptimisation:
    """A model that cannot overfit one batch has a wiring bug."""

    @pytest.mark.parametrize("name", ["clinical_only", "oct_only", "concat_fusion", "gated_fusion"])
    def test_overfits_a_single_batch(self, name: str) -> None:
        torch.manual_seed(0)
        config = ModelConfig(name=name, pretrained=False, dropout=0.0)
        model = ModelFactory().build(config, n_labels=6, clinical_dim=CLINICAL_DIM)
        model.train()

        image = torch.randn(BATCH, 3, 64, 64)
        clinical = torch.randn(BATCH, CLINICAL_DIM)
        targets = torch.randint(0, 2, (BATCH, 6)).float()

        criterion = LossFactory().build("bce")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        first = None
        for step in range(60):
            optimizer.zero_grad()
            logits = model(
                image=image if model.uses_image else None,
                clinical=clinical if model.uses_clinical else None,
            )
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            if step == 0:
                first = float(loss.detach())
        assert float(loss.detach()) < first * 0.5, f"{name} failed to overfit one batch"

    def test_gradients_reach_both_encoders_in_the_gated_model(self) -> None:
        model = GatedFusionModel(clinical_dim=CLINICAL_DIM, n_labels=6, pretrained=False)
        image, clinical, targets = _inputs(6)
        loss = MaskedBCEWithLogitsLoss()(model(image=image, clinical=clinical), targets)
        loss.backward()

        image_grad = model.image_encoder.projection[2].weight.grad
        clinical_grad = model.clinical_encoder.network[0].weight.grad
        assert image_grad is not None and image_grad.abs().sum() > 0
        assert clinical_grad is not None and clinical_grad.abs().sum() > 0


class TestLosses:
    """Masking and weighting."""

    def test_mask_excludes_entries_from_the_loss(self) -> None:
        # Column 2 is confidently wrong; masking it must lower the loss.
        logits = torch.tensor([[2.0, 2.0, -8.0], [2.0, 2.0, -8.0]])
        targets = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
        criterion = MaskedBCEWithLogitsLoss()
        assert criterion(logits, targets, mask) < criterion(logits, targets)

    def test_a_fully_open_mask_matches_the_unmasked_loss(self) -> None:
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()
        criterion = MaskedBCEWithLogitsLoss()
        torch.testing.assert_close(
            criterion(logits, targets, torch.ones(4, 3, dtype=torch.bool)),
            criterion(logits, targets),
        )

    def test_pos_weight_increases_the_penalty_on_missed_positives(self) -> None:
        logits = torch.full((4, 1), -3.0)
        targets = torch.ones(4, 1)
        plain = MaskedBCEWithLogitsLoss()(logits, targets)
        weighted = MaskedBCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))(logits, targets)
        assert weighted > plain

    def test_focal_loss_is_available_as_an_ablation(self) -> None:
        criterion = LossFactory().build("focal")
        loss = criterion(torch.randn(4, 6), torch.randint(0, 2, (4, 6)).float())
        assert torch.isfinite(loss)
