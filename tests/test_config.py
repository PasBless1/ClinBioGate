"""Configuration loading, inheritance and round-tripping."""

from __future__ import annotations

import pytest

from olives_biomarkers.config import ConfigLoader, ExperimentConfig, ModelConfig, SplitConfig
from olives_biomarkers.utils.io import YamlIO


class TestConfigLoader:
    """Loading the shipped configs."""

    @pytest.mark.parametrize(
        "stem",
        ["data", "baseline_clinical", "baseline_oct", "fusion_concat", "fusion_gated",
         "local_cpu", "colab_gpu",
         "oct_improved", "oct_adjacent", "oct_retfound", "loss_asl",
         "fusion_residual_logit", "fusion_film",
         "fusion_longitudinal", "fusion_delta_only",
         "control_patient_mean", "control_within_shuffle",
         "control_across_shuffle", "control_quantise"],
    )
    def test_every_shipped_config_loads(self, repo_root, stem: str) -> None:
        config = ConfigLoader(repo_root).load(repo_root / "configs" / f"{stem}.yaml")
        assert config.source_path is not None
        assert config.data.target_set in {"six", "sixteen"}

    def test_defaults_are_inherited_then_overridden(self, repo_root) -> None:
        base = ConfigLoader(repo_root).load(repo_root / "configs" / "data.yaml")
        derived = ConfigLoader(repo_root).load(repo_root / "configs" / "fusion_gated.yaml")
        # Inherited from data.yaml
        assert derived.data.root == base.data.root
        # Set by the derived file
        assert derived.model.name == "gated_fusion"

    def test_unknown_section_is_rejected(self, repo_root, tmp_path) -> None:
        path = tmp_path / "bad.yaml"
        YamlIO.write({"nonsense": {"a": 1}}, path)
        with pytest.raises(ValueError, match="unknown config sections"):
            ConfigLoader(repo_root).load(path)

    def test_unknown_key_inside_a_section_is_rejected(self, repo_root, tmp_path) -> None:
        path = tmp_path / "bad.yaml"
        YamlIO.write({"model": {"name": "oct_only", "not_a_real_knob": 1}}, path)
        with pytest.raises(ValueError, match="unknown keys for ModelConfig"):
            ConfigLoader(repo_root).load(path)

    def test_missing_file_names_the_path(self, repo_root, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="config not found"):
            ConfigLoader(repo_root).load(tmp_path / "absent.yaml")

    def test_circular_defaults_are_caught(self, repo_root, tmp_path) -> None:
        a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
        YamlIO.write({"defaults": str(b)}, a)
        YamlIO.write({"defaults": str(a)}, b)
        with pytest.raises(ValueError, match="circular defaults"):
            ConfigLoader(repo_root).load(a)


class TestSavedConfigRoundTrip:
    """A config written beside a run must reload without editing.

    Regression guard: ``to_dict`` emits ``source_path``, which is metadata rather
    than a config section. Before this was handled, evaluating a finished run
    failed with "unknown config sections: ['source_path']".
    """

    def test_saved_config_reloads(self, repo_root, tmp_path) -> None:
        original = ConfigLoader(repo_root).load(repo_root / "configs" / "fusion_gated.yaml")
        saved = original.save(tmp_path / "resolved_config.yaml")

        reloaded = ConfigLoader(repo_root).load(saved)
        assert reloaded.model.name == original.model.name
        assert reloaded.data.target_set == original.data.target_set
        assert reloaded.training.epochs == original.training.epochs
        assert reloaded.model.gate_residual == original.model.gate_residual

    def test_round_trip_is_stable_across_two_saves(self, repo_root, tmp_path) -> None:
        first = ConfigLoader(repo_root).load(repo_root / "configs" / "baseline_oct.yaml")
        once = ConfigLoader(repo_root).load(first.save(tmp_path / "one.yaml"))
        twice = ConfigLoader(repo_root).load(once.save(tmp_path / "two.yaml"))
        assert once.to_dict()["model"] == twice.to_dict()["model"]
        assert once.to_dict()["training"] == twice.to_dict()["training"]


class TestValidation:
    """Config objects reject impossible values at construction."""

    def test_split_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitConfig(train_fraction=0.8, val_fraction=0.3, test_fraction=0.3)

    def test_unknown_target_set_is_rejected(self) -> None:
        from olives_biomarkers.config import DataConfig

        with pytest.raises(ValueError, match="target_set"):
            DataConfig(target_set="twelve")

    def test_unknown_duplicate_policy_is_rejected(self) -> None:
        from olives_biomarkers.config import DuplicateConfig

        with pytest.raises(ValueError, match="duplicate policy"):
            DuplicateConfig(policy="keep_some")

    def test_gate_bias_defaults_to_none(self) -> None:
        # None means "let ClinicalGate pick the identity-preserving value".
        assert ModelConfig().gate_bias_init is None

    def test_experiment_config_to_dict_is_complete(self) -> None:
        payload = ExperimentConfig().to_dict()
        for section in ("project", "data", "model", "training", "evaluation", "split"):
            assert section in payload


class TestImprovementConfigs:
    """The improvement-round configs must express what they claim to."""

    def test_oct_improved_enables_crop_and_320px(self, repo_root) -> None:
        config = ConfigLoader(repo_root).load(repo_root / "configs" / "oct_improved.yaml")
        assert config.data.crop_retina is True
        assert config.data.image_size == (320, 320)
        assert config.data.preserve_aspect_ratio is True
        assert config.data.normalization == "train_fold"

    def test_oct_improved_uses_discriminative_rates_and_cosine(self, repo_root) -> None:
        training = ConfigLoader(repo_root).load(
            repo_root / "configs" / "oct_improved.yaml"
        ).training
        assert training.backbone_learning_rate < training.head_learning_rate
        assert training.scheduler == "cosine"
        assert training.warmup_epochs > 0
        assert training.freeze_backbone_epochs > 0
        assert training.gradual_unfreeze_epochs > 0

    def test_oct_improved_balances_by_patient(self, repo_root) -> None:
        training = ConfigLoader(repo_root).load(
            repo_root / "configs" / "oct_improved.yaml"
        ).training
        assert training.sampler == "patient"
        assert training.pos_weight_unit == "patient"

    def test_adjacent_config_switches_image_mode_only(self, repo_root) -> None:
        loader = ConfigLoader(repo_root)
        base = loader.load(repo_root / "configs" / "oct_improved.yaml")
        adjacent = loader.load(repo_root / "configs" / "oct_adjacent.yaml")
        assert adjacent.data.image_mode == "adjacent"
        assert base.data.image_mode == "repeat"
        # Everything else must match, or the comparison is confounded.
        assert adjacent.training.epochs == base.training.epochs
        assert adjacent.data.image_size == base.data.image_size
        assert adjacent.model.name == base.model.name

    def test_asl_config_switches_loss_only(self, repo_root) -> None:
        loader = ConfigLoader(repo_root)
        base = loader.load(repo_root / "configs" / "oct_improved.yaml")
        asl = loader.load(repo_root / "configs" / "loss_asl.yaml")
        assert asl.training.loss == "asl"
        assert base.training.loss == "bce"
        assert asl.model.name == base.model.name
        assert asl.data.image_size == base.data.image_size

    def test_retfound_config_points_at_a_vit_backbone(self, repo_root) -> None:
        config = ConfigLoader(repo_root).load(repo_root / "configs" / "oct_retfound.yaml")
        assert config.model.image_encoder == "retfound_vit_large_patch16"
        # torchvision weights must be off; the weights come from the checkpoint.
        assert config.model.pretrained is False

    def test_fusion_configs_select_the_bounded_variants(self, repo_root) -> None:
        loader = ConfigLoader(repo_root)
        residual = loader.load(repo_root / "configs" / "fusion_residual_logit.yaml")
        film = loader.load(repo_root / "configs" / "fusion_film.yaml")
        assert residual.model.name == "residual_logit_fusion"
        assert residual.model.clinical_residual_per_label is True
        assert film.model.name == "bounded_film_fusion"
        assert film.model.film_max_scale == 0.25

    def test_fusion_configs_inherit_the_improved_oct_pipeline(self, repo_root) -> None:
        """Fusion must be tested on top of the best OCT setup, not the old one."""
        loader = ConfigLoader(repo_root)
        base = loader.load(repo_root / "configs" / "oct_improved.yaml")
        for stem in ("fusion_residual_logit", "fusion_film"):
            config = loader.load(repo_root / "configs" / f"{stem}.yaml")
            assert config.data.crop_retina == base.data.crop_retina
            assert config.data.image_size == base.data.image_size
            assert config.training.scheduler == base.training.scheduler


class TestLongitudinalAndControlConfigs:
    """The within-eye arm and the control ladder that keeps it honest."""

    def test_longitudinal_arm_requests_the_derived_features(self, repo_root) -> None:
        config = ConfigLoader(repo_root).load(repo_root / "configs" / "fusion_longitudinal.yaml")
        assert config.data.longitudinal_clinical is True
        assert "cst_delta" in config.model.clinical_features
        assert "bcva_delta" in config.model.clinical_features
        assert config.data.clinical_perturbation == "none"

    def test_longitudinal_arm_uses_the_per_label_residual_design(self, repo_root) -> None:
        """The gate cannot express 'use CST for this label only'; beta can."""
        config = ConfigLoader(repo_root).load(repo_root / "configs" / "fusion_longitudinal.yaml")
        assert config.model.name == "residual_logit_fusion"
        assert config.model.clinical_residual_per_label is True

    def test_delta_only_arm_withholds_the_absolute_values(self, repo_root) -> None:
        """This arm has no fingerprint left to exploit, which is the point."""
        config = ConfigLoader(repo_root).load(repo_root / "configs" / "fusion_delta_only.yaml")
        assert config.model.clinical_features == ["bcva_delta", "cst_delta", "is_baseline_visit"]
        assert "cst" not in config.model.clinical_features
        assert "bcva" not in config.model.clinical_features

    @pytest.mark.parametrize(
        "stem,mode",
        [
            ("control_patient_mean", "patient_mean"),
            ("control_within_shuffle", "within_patient_shuffle"),
            ("control_across_shuffle", "across_patient_shuffle"),
            ("control_quantise", "quantise"),
        ],
    )
    def test_each_control_selects_its_perturbation(self, repo_root, stem, mode) -> None:
        config = ConfigLoader(repo_root).load(repo_root / "configs" / f"{stem}.yaml")
        assert config.data.clinical_perturbation == mode
        assert config.data.is_control_arm is True

    def test_controls_match_the_arm_they_are_read_against(self, repo_root) -> None:
        """A control differing in anything but the perturbation proves nothing."""
        loader = ConfigLoader(repo_root)
        base = loader.load(repo_root / "configs" / "fusion_longitudinal.yaml")
        for stem in ("control_patient_mean", "control_within_shuffle",
                     "control_across_shuffle", "control_quantise"):
            config = loader.load(repo_root / "configs" / f"{stem}.yaml")
            assert config.model.name == base.model.name
            assert config.model.clinical_features == base.model.clinical_features
            assert config.data.image_size == base.data.image_size
            assert config.data.crop_retina == base.data.crop_retina
            assert config.training.epochs == base.training.epochs
            assert config.training.scheduler == base.training.scheduler

    def test_unknown_perturbation_is_rejected(self) -> None:
        from olives_biomarkers.config import DataConfig

        with pytest.raises(ValueError, match="unknown clinical_perturbation"):
            DataConfig(clinical_perturbation="scramble")

    def test_default_config_is_not_a_control(self) -> None:
        from olives_biomarkers.config import DataConfig

        assert DataConfig().is_control_arm is False
        assert DataConfig().longitudinal_clinical is False

    def test_quantise_bins_round_trip_through_save(self, repo_root, tmp_path) -> None:
        original = ConfigLoader(repo_root).load(repo_root / "configs" / "control_quantise.yaml")
        reloaded = ConfigLoader(repo_root).load(original.save(tmp_path / "resolved.yaml"))
        assert reloaded.data.clinical_bins == original.data.clinical_bins
        assert reloaded.data.clinical_perturbation == "quantise"
