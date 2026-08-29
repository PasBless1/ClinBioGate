"""Optimiser groups, learning-rate schedule and progressive unfreezing."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from olives_biomarkers.config import ModelConfig, TrainingConfig  # noqa: E402
from olives_biomarkers.models import ClinicalOnlyModel, ModelFactory  # noqa: E402
from olives_biomarkers.training.engine import Trainer  # noqa: E402
from olives_biomarkers.training.losses import MaskedBCEWithLogitsLoss  # noqa: E402
from olives_biomarkers.training.schedule import (  # noqa: E402
    FineTuneSchedule,
    ParameterGroupBuilder,
    WarmupCosineSchedule,
)


def _oct_model(dropout: float = 0.3):
    return ModelFactory().build(
        ModelConfig(name="oct_only", pretrained=False, dropout=dropout),
        n_labels=6,
        clinical_dim=4,
    )


class TestParameterGroupBuilder:
    """Backbone and head must receive different learning rates."""

    def test_splits_backbone_from_head(self) -> None:
        groups = ParameterGroupBuilder().build(
            _oct_model(), backbone_lr=1e-5, head_lr=1e-3, weight_decay=1e-3
        )
        by_name = {g["name"]: g for g in groups}
        assert {"backbone", "head"} <= set(by_name)
        assert by_name["backbone"]["lr"] == pytest.approx(1e-5)
        assert by_name["head"]["lr"] == pytest.approx(1e-3)

    def test_every_parameter_lands_in_exactly_one_group(self) -> None:
        model = _oct_model()
        groups = ParameterGroupBuilder().build(model, 1e-5, 1e-3, 1e-3)
        grouped = sum(len(g["params"]) for g in groups)
        assert grouped == len(list(model.parameters()))
        seen = {id(p) for g in groups for p in g["params"]}
        assert len(seen) == grouped

    def test_norm_and_bias_parameters_get_no_weight_decay(self) -> None:
        groups = ParameterGroupBuilder().build(_oct_model(), 1e-5, 1e-3, weight_decay=1e-3)
        for group in groups:
            if group["name"].endswith("no_decay"):
                assert group["weight_decay"] == 0.0
                assert group["params"]

    def test_clinical_only_model_has_no_backbone_group(self) -> None:
        model = ClinicalOnlyModel(clinical_dim=4, n_labels=6)
        groups = ParameterGroupBuilder().build(model, 1e-5, 1e-3, 1e-3)
        assert all(not g["name"].startswith("backbone") for g in groups)


class TestFineTuneSchedule:
    """Phase timeline."""

    def test_disabled_by_default(self) -> None:
        schedule = FineTuneSchedule()
        assert not schedule.enabled
        assert schedule.mode_for_epoch(1) == "all"

    def test_three_phase_timeline(self) -> None:
        schedule = FineTuneSchedule(freeze_epochs=4, gradual_epochs=6)
        assert [schedule.mode_for_epoch(e) for e in (1, 4)] == ["frozen", "frozen"]
        assert [schedule.mode_for_epoch(e) for e in (5, 10)] == ["last", "last"]
        assert [schedule.mode_for_epoch(e) for e in (11, 40)] == ["all", "all"]

    def test_freeze_without_gradual_goes_straight_to_all(self) -> None:
        schedule = FineTuneSchedule(freeze_epochs=3, gradual_epochs=0)
        assert schedule.mode_for_epoch(3) == "frozen"
        assert schedule.mode_for_epoch(4) == "all"


class TestWarmupCosineSchedule:
    """Learning-rate curve."""

    @staticmethod
    def _schedule(total=40, warmup=3, min_ratio=0.01):
        model = _oct_model()
        optimizer = torch.optim.AdamW(
            ParameterGroupBuilder().build(model, 1e-5, 1e-4, 1e-3)
        )
        return WarmupCosineSchedule(optimizer, total, warmup, min_ratio), optimizer

    def test_warmup_ramps_up_then_decays(self) -> None:
        schedule, _ = self._schedule()
        warm = [schedule.factor_for_epoch(e) for e in (1, 2, 3)]
        assert warm == sorted(warm), "warmup should increase"
        assert warm[-1] == pytest.approx(1.0)
        assert schedule.factor_for_epoch(40) < schedule.factor_for_epoch(20)

    def test_decays_to_the_configured_floor(self) -> None:
        schedule, _ = self._schedule(total=40, warmup=3, min_ratio=0.01)
        assert schedule.factor_for_epoch(40) == pytest.approx(0.01, abs=1e-6)

    def test_group_ratio_is_preserved_throughout(self) -> None:
        """The backbone must stay 10x below the head at every epoch."""
        schedule, optimizer = self._schedule()
        for epoch in (1, 5, 20, 40):
            schedule.step(epoch)
            rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
            assert rates["head"] / rates["backbone"] == pytest.approx(10.0)

    def test_no_warmup_starts_at_full_rate(self) -> None:
        schedule, _ = self._schedule(warmup=0)
        assert schedule.factor_for_epoch(1) == pytest.approx(1.0, abs=1e-3)


class TestTrainerIntegration:
    """The knobs must actually reach the optimiser."""

    @staticmethod
    def _trainer(**overrides):
        config = TrainingConfig(
            epochs=40,
            learning_rate=1e-4,
            backbone_learning_rate=1e-5,
            head_learning_rate=1e-4,
            weight_decay=1e-3,
            scheduler="cosine",
            warmup_epochs=3,
            freeze_backbone_epochs=4,
            gradual_unfreeze_epochs=6,
            **overrides,
        )
        return Trainer(_oct_model(), MaskedBCEWithLogitsLoss(), config, device="cpu")

    def test_discriminative_rates_are_applied(self) -> None:
        trainer = self._trainer()
        rates = {g["name"]: g["lr"] for g in trainer.optimizer.param_groups}
        assert rates["backbone"] == pytest.approx(1e-5)
        assert rates["head"] == pytest.approx(1e-4)

    def test_falls_back_to_a_single_rate_when_unset(self) -> None:
        config = TrainingConfig(learning_rate=3e-4)
        trainer = Trainer(_oct_model(), MaskedBCEWithLogitsLoss(), config, device="cpu")
        for group in trainer.optimizer.param_groups:
            assert group["lr"] == pytest.approx(3e-4)
        assert trainer.scheduler is None
        assert not trainer.fine_tune_schedule.enabled

    def test_freezing_reduces_then_restores_trainable_parameters(self) -> None:
        trainer = self._trainer()
        total = sum(p.numel() for p in trainer.model.parameters())

        trainer.apply_trainability("frozen")
        frozen = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        trainer.apply_trainability("last")
        partial = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        trainer.apply_trainability("all")
        everything = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

        assert frozen < partial < everything == total

    def test_repeated_mode_is_a_no_op(self) -> None:
        trainer = self._trainer()
        assert trainer.apply_trainability("frozen") is True
        assert trainer.apply_trainability("frozen") is False

    def test_optimizer_is_not_rebuilt_on_unfreeze(self) -> None:
        """Rebuilding would discard AdamW's momentum state mid-training."""
        trainer = self._trainer()
        before = id(trainer.optimizer)
        trainer.apply_trainability("frozen")
        trainer.apply_trainability("all")
        assert id(trainer.optimizer) == before

    def test_frozen_backbone_batchnorm_stays_in_eval(self) -> None:
        trainer = self._trainer()
        trainer.apply_trainability("frozen")
        trainer.model.train(True)
        trainer.model.image_encoder_module().enforce_frozen_eval()
        norms = [
            m for m in trainer.model.image_encoder_module().backbone.modules()
            if isinstance(m, torch.nn.BatchNorm2d)
        ]
        assert norms and all(not m.training for m in norms)

    def test_a_frozen_encoder_still_trains_the_head(self) -> None:
        trainer = self._trainer()
        trainer.apply_trainability("frozen")
        image = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 2, (2, 6)).float()
        loss = trainer.criterion(trainer.model(image=image), target)
        loss.backward()

        encoder = trainer.model.image_encoder_module()
        backbone_grads = [p.grad for p in encoder.backbone.parameters() if p.grad is not None]
        head_grads = [p.grad for p in trainer.model.head.parameters() if p.grad is not None]
        assert not backbone_grads, "frozen backbone received gradients"
        assert head_grads and any(g.abs().sum() > 0 for g in head_grads)
