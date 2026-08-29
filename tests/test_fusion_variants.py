"""Bounded fusion designs and seed ensembling.

The property that matters for both new fusion models is that they *start* as the
OCT baseline exactly. The A100 result showed the unbounded gate settling at a
mean scale of 1.897 out of 2 while losing to OCT-only; a design that cannot
depart from the baseline without evidence is the direct answer to that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from olives_biomarkers.config import ModelConfig  # noqa: E402
from olives_biomarkers.evaluation.ensemble import (  # noqa: E402
    EnsembleMismatchError,
    SeedEnsemble,
)
from olives_biomarkers.evaluation.metrics import ThresholdSet  # noqa: E402
from olives_biomarkers.experiment import RunResult  # noqa: E402
from olives_biomarkers.models import (  # noqa: E402
    BoundedFiLMFusionModel,
    ModelFactory,
    ResidualLogitFusionModel,
)
from olives_biomarkers.training.callbacks import MetricHistory  # noqa: E402

BATCH = 4
CLINICAL_DIM = 4
N_LABELS = 6
LABELS = [f"l{i}" for i in range(N_LABELS)]


def _inputs():
    return torch.randn(BATCH, 3, 64, 64), torch.randn(BATCH, CLINICAL_DIM)


class TestResidualLogitFusion:
    """Clinical evidence as a bounded per-biomarker logit correction."""

    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return ResidualLogitFusionModel(
            clinical_dim=CLINICAL_DIM, n_labels=N_LABELS, pretrained=False
        )

    def test_output_shape(self, model) -> None:
        image, clinical = _inputs()
        assert model(image=image, clinical=clinical).shape == (BATCH, N_LABELS)

    def test_starts_exactly_at_the_oct_baseline(self, model) -> None:
        model.eval()
        image, clinical = _inputs()
        with torch.no_grad():
            parts = model.embeddings(image=image, clinical=clinical)
        torch.testing.assert_close(parts["fused"], parts["oct_logits"])
        assert float(parts["beta"].abs().max()) == 0.0

    def test_clinical_input_cannot_change_the_output_at_init(self, model) -> None:
        """With beta at zero, two different clinical vectors give identical logits."""
        model.eval()
        image = torch.randn(BATCH, 3, 64, 64)
        with torch.no_grad():
            a = model(image=image, clinical=torch.zeros(BATCH, CLINICAL_DIM))
            b = model(image=image, clinical=torch.randn(BATCH, CLINICAL_DIM) * 5)
        torch.testing.assert_close(a, b)

    def test_beta_is_bounded_by_max_scale(self) -> None:
        model = ResidualLogitFusionModel(
            clinical_dim=CLINICAL_DIM, n_labels=N_LABELS, pretrained=False, max_scale=0.5
        )
        with torch.no_grad():
            model.beta.fill_(50.0)  # drive tanh into saturation
        assert float(model.effective_beta().detach().abs().max()) <= 0.5 + 1e-6

    def test_per_label_beta_has_one_value_per_biomarker(self, model) -> None:
        assert model.effective_beta().numel() == N_LABELS
        values = model.beta_values(LABELS)
        assert set(values) == set(LABELS)

    def test_global_beta_is_shared_across_labels(self) -> None:
        model = ResidualLogitFusionModel(
            clinical_dim=CLINICAL_DIM, n_labels=N_LABELS, pretrained=False, per_label=False
        )
        assert model.effective_beta().numel() == 1
        assert len(model.beta_values(LABELS)) == N_LABELS

    def test_beta_receives_gradient(self, model) -> None:
        """A zero init must not be a dead end."""
        image, clinical = _inputs()
        target = torch.randint(0, 2, (BATCH, N_LABELS)).float()
        torch.nn.functional.binary_cross_entropy_with_logits(
            model(image=image, clinical=clinical), target
        ).backward()
        assert model.beta.grad is not None
        assert float(model.beta.grad.abs().sum()) > 0

    def test_clinical_branch_activates_once_beta_moves(self, model) -> None:
        with torch.no_grad():
            model.beta.fill_(1.0)
        model.eval()
        image = torch.randn(BATCH, 3, 64, 64)
        with torch.no_grad():
            a = model(image=image, clinical=torch.zeros(BATCH, CLINICAL_DIM))
            b = model(image=image, clinical=torch.randn(BATCH, CLINICAL_DIM) * 5)
        assert not torch.allclose(a, b)

    def test_contribution_summary_reports_zero_clinical_share_at_init(self, model) -> None:
        image, clinical = _inputs()
        summary = model.contribution_summary(image, clinical)
        assert summary["clinical_share"] == pytest.approx(0.0)
        assert summary["max_abs_beta"] == pytest.approx(0.0)

    def test_regularization_is_zero_at_init(self, model) -> None:
        assert model.regularization_value() == pytest.approx(0.0)

    def test_exposes_a_grad_cam_layer(self, model) -> None:
        assert model.feature_layer is not None


class TestBoundedFiLMFusion:
    """Feature modulation capped well below the old gate's 2x amplification."""

    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return BoundedFiLMFusionModel(
            clinical_dim=CLINICAL_DIM, n_labels=N_LABELS, pretrained=False, max_scale=0.25
        )

    def test_starts_at_identity(self, model) -> None:
        model.eval()
        image, clinical = _inputs()
        with torch.no_grad():
            parts = model.embeddings(image=image, clinical=clinical)
        torch.testing.assert_close(parts["fused"], parts["image"])
        assert float(parts["scale"].min()) == pytest.approx(1.0)

    def test_scale_stays_within_the_configured_bound(self, model) -> None:
        with torch.no_grad():
            model.film.weight.normal_(0, 5)
            model.film.bias.normal_(0, 5)
            parts = model.embeddings(*_inputs())
        assert float(parts["scale"].min()) >= 1.0 - 0.25 - 1e-5
        assert float(parts["scale"].max()) <= 1.0 + 0.25 + 1e-5

    def test_is_far_more_bounded_than_the_legacy_gate(self) -> None:
        """The gate could reach 2.0; FiLM at 0.25 caps at 1.25."""
        film = BoundedFiLMFusionModel(
            clinical_dim=CLINICAL_DIM, n_labels=N_LABELS, pretrained=False, max_scale=0.25
        )
        gated = ModelFactory().build(
            ModelConfig(name="gated_fusion", pretrained=False), N_LABELS, CLINICAL_DIM
        )
        with torch.no_grad():
            film.film.weight.normal_(0, 10)
            film.film.bias.normal_(0, 10)
            film_max = float(film.embeddings(*_inputs())["scale"].max())
            gate_max = float(gated.gate.scale(torch.ones(1, 8)).max())
        assert film_max < 1.3 < gate_max


class TestModelFactoryCoversNewVariants:
    """Every fusion variant must be reachable from config."""

    @pytest.mark.parametrize("name", ["residual_logit_fusion", "bounded_film_fusion"])
    def test_builds_from_config(self, name: str) -> None:
        model = ModelFactory().build(
            ModelConfig(name=name, pretrained=False), n_labels=N_LABELS, clinical_dim=CLINICAL_DIM
        )
        image, clinical = _inputs()
        assert model(image=image, clinical=clinical).shape == (BATCH, N_LABELS)

    def test_registry_lists_them(self) -> None:
        assert {"residual_logit_fusion", "bounded_film_fusion"} <= set(ModelFactory.available())

    def test_per_label_flag_reaches_the_model(self) -> None:
        config = ModelConfig(name="residual_logit_fusion", pretrained=False)
        config.clinical_residual_per_label = False
        model = ModelFactory().build(config, n_labels=N_LABELS, clinical_dim=CLINICAL_DIM)
        assert model.effective_beta().numel() == 1


def _run(run_id: str, seed: int, logits: np.ndarray, targets: np.ndarray,
         row_uid: np.ndarray, patients: np.ndarray, model_name: str = "oct_only") -> RunResult:
    """Minimal RunResult carrying val and test predictions."""
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    payload = {
        "logits": logits,
        "probabilities": probabilities,
        "targets": targets,
        "row_uid": row_uid,
        "patient_id": patients,
    }
    return RunResult(
        run_id=run_id,
        experiment=model_name,
        model_name=model_name,
        seed=seed,
        split_name="holdout",
        target_set="six",
        run_dir="/tmp",
        history=MetricHistory(),
        thresholds=ThresholdSet(thresholds=np.full(N_LABELS, 0.5), label_names=LABELS),
        label_names=LABELS,
        predictions={"test": payload, "val": payload},
    )


class TestSeedEnsemble:
    """Averaging must be aligned, validated and actually helpful."""

    @staticmethod
    def _members(n: int = 3, seed: int = 0):
        """``n`` runs over the same scans, differing only in prediction noise."""
        rng = np.random.default_rng(seed)
        n_rows = 60
        row_uid = np.arange(n_rows)
        patients = np.repeat(np.arange(6), 10)
        targets = rng.integers(0, 2, (n_rows, N_LABELS)).astype(float)
        signal = (targets * 2 - 1) * 1.5

        members = []
        for index in range(n):
            logits = signal + rng.normal(0, 2.5, signal.shape)
            members.append(
                _run(f"run{index}", 42 + index, logits, targets, row_uid, patients)
            )
        return members, targets

    @staticmethod
    def _shuffle_member(member: RunResult, seed: int = 7) -> RunResult:
        """The same run with its rows stored in a different order."""
        payload = member.predictions["test"]
        order = np.random.default_rng(seed).permutation(len(payload["row_uid"]))
        shuffled = {key: value[order] for key, value in payload.items()}
        return _run(
            member.run_id, member.seed, shuffled["logits"], shuffled["targets"],
            shuffled["row_uid"], shuffled["patient_id"],
        )

    def test_combines_to_the_expected_shape(self) -> None:
        members, targets = self._members()
        ensemble = SeedEnsemble().combine(members)
        assert ensemble.logits.shape == targets.shape
        assert ensemble.n_members == 3
        assert list(ensemble.label_names) == LABELS

    def test_realigns_runs_stored_in_different_row_orders(self) -> None:
        """Positional averaging would silently mix scans; alignment is by row_uid."""
        members, _ = self._members()
        reordered = [members[0], self._shuffle_member(members[1]), members[2]]

        baseline = SeedEnsemble().combine(members)
        realigned = SeedEnsemble().combine(reordered)

        np.testing.assert_allclose(realigned.logits, baseline.logits, atol=1e-10)
        assert np.array_equal(realigned.row_uid, np.sort(realigned.row_uid))

    def test_rejects_runs_evaluated_on_different_patients(self) -> None:
        members, _ = self._members(n=2)
        members[1].predictions["test"]["patient_id"] = members[1].predictions["test"]["patient_id"] + 100
        with pytest.raises(EnsembleMismatchError, match="different test patients"):
            SeedEnsemble().combine(members)

    def test_rejects_a_single_run(self) -> None:
        members, _ = self._members(n=1)
        with pytest.raises(EnsembleMismatchError, match="at least two runs"):
            SeedEnsemble().combine(members)

    def test_rejects_mismatched_label_sets(self) -> None:
        members, _ = self._members(n=2)
        members[1].label_names = LABELS[:-1]
        with pytest.raises(EnsembleMismatchError, match="different label sets"):
            SeedEnsemble().combine(members)

    def test_averaging_reduces_noise_against_the_mean_member(self) -> None:
        members, targets = self._members(n=5)
        from olives_biomarkers.evaluation.metrics import MultiLabelMetrics

        metrics = MultiLabelMetrics(label_names=LABELS)
        member_scores = [
            metrics.compute(m.predictions["test"]["targets"], m.predictions["test"]["probabilities"])[
                "macro_auroc"
            ]
            for m in members
        ]
        ensemble = SeedEnsemble().combine(members)
        ensemble_score = metrics.compute(ensemble.targets, ensemble.probabilities)["macro_auroc"]
        assert ensemble_score > np.mean(member_scores)

    def test_probability_space_is_also_supported(self) -> None:
        members, _ = self._members()
        logit_space = SeedEnsemble("logit").combine(members)
        probability_space = SeedEnsemble("probability").combine(members)
        assert probability_space.space == "probability"
        # Jensen's inequality: the two averages are not the same operation.
        assert not np.allclose(logit_space.probabilities, probability_space.probabilities)

    def test_unknown_space_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown ensemble space"):
            SeedEnsemble("median")

    def test_comparison_table_includes_members_and_ensemble(self) -> None:
        members, _ = self._members()
        table = SeedEnsemble().compare_with_members(members, thresholds=0.5)
        assert isinstance(table, pd.DataFrame)
        assert len(table) == len(members) + 1
        assert "ensemble" in str(table.iloc[-1]["model"])

    def test_thresholds_are_fitted_on_ensembled_validation(self) -> None:
        members, _ = self._members()
        thresholds = SeedEnsemble().fit_thresholds(members)
        assert thresholds.fitted_on == "validation"
        assert len(thresholds.as_array()) == N_LABELS
