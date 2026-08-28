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
         "local_cpu", "colab_gpu"],
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
