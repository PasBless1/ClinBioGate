"""MC dropout and selective prediction tests.

The load-bearing test is ``test_batch_norm_stays_in_eval_mode``: if batch-norm
were reactivated alongside dropout, predictions would depend on batch
composition and the "uncertainty" would be an artefact of batching.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import nn  # noqa: E402

from olives_biomarkers.evaluation.uncertainty import (  # noqa: E402
    MCDropoutInference,
    SelectivePredictor,
)
from olives_biomarkers.models import ClinicalOnlyModel, OCTOnlyModel  # noqa: E402

BATCH = 8
CLINICAL_DIM = 4
N_LABELS = 6


class _FakeLoader:
    """Minimal loader yielding fixed batches, so passes are comparable."""

    def __init__(self, n_batches: int = 3, with_image: bool = False) -> None:
        generator = torch.Generator().manual_seed(0)
        self.batches = [
            {
                "image": torch.randn(BATCH, 3, 64, 64, generator=generator)
                if with_image
                else torch.zeros(BATCH),
                "clinical": torch.randn(BATCH, CLINICAL_DIM, generator=generator),
                "target": torch.randint(0, 2, (BATCH, N_LABELS), generator=generator).float(),
                "row_uid": torch.arange(index * BATCH, (index + 1) * BATCH),
                "patient_id": torch.full((BATCH,), index),
            }
            for index in range(n_batches)
        ]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


@pytest.fixture
def clinical_model():
    torch.manual_seed(0)
    return ClinicalOnlyModel(clinical_dim=CLINICAL_DIM, n_labels=N_LABELS, dropout=0.5)


class TestMCDropout:
    """Stochastic inference."""

    def test_enables_only_dropout_modules(self, clinical_model) -> None:
        clinical_model.eval()
        n_enabled = MCDropoutInference.enable_dropout(clinical_model)
        assert n_enabled > 0
        for module in clinical_model.modules():
            if isinstance(module, nn.Dropout):
                assert module.training
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                assert not module.training, "batch-norm was reactivated; predictions would drift"

    def test_batch_norm_stays_in_eval_mode(self, clinical_model) -> None:
        loader = _FakeLoader()
        inference = MCDropoutInference(clinical_model, n_passes=5)
        before = {
            name: buffer.clone()
            for name, buffer in clinical_model.named_buffers()
            if "running_" in name
        }
        inference.run(loader)
        for name, buffer in clinical_model.named_buffers():
            if "running_" in name:
                torch.testing.assert_close(
                    buffer, before[name], msg=f"batch-norm buffer {name} changed during MC dropout"
                )

    def test_returns_the_expected_shapes(self, clinical_model) -> None:
        loader = _FakeLoader(n_batches=3)
        output = MCDropoutInference(clinical_model, n_passes=7).run(loader)
        expected = (3 * BATCH, N_LABELS)
        assert output.mean_probability.shape == expected
        assert output.std_probability.shape == expected
        assert output.predictive_entropy.shape == expected
        assert output.mutual_information.shape == expected
        assert output.n_passes == 7

    def test_probabilities_are_in_range(self, clinical_model) -> None:
        output = MCDropoutInference(clinical_model, n_passes=5).run(_FakeLoader())
        assert output.mean_probability.min() >= 0.0
        assert output.mean_probability.max() <= 1.0

    def test_dropout_produces_non_zero_variance(self, clinical_model) -> None:
        output = MCDropoutInference(clinical_model, n_passes=10).run(_FakeLoader())
        assert output.std_probability.mean() > 1e-4, "MC dropout produced no variance"

    def test_deterministic_evaluation_is_stable(self, clinical_model) -> None:
        clinical_model.eval()
        loader = _FakeLoader()
        with torch.no_grad():
            first = torch.cat([clinical_model(clinical=b["clinical"]) for b in loader])
            second = torch.cat([clinical_model(clinical=b["clinical"]) for b in loader])
        torch.testing.assert_close(first, second)

    def test_zero_dropout_gives_zero_variance(self) -> None:
        model = ClinicalOnlyModel(clinical_dim=CLINICAL_DIM, n_labels=N_LABELS, dropout=0.0)
        output = MCDropoutInference(model, n_passes=5).run(_FakeLoader())
        assert output.std_probability.max() < 1e-6

    def test_mutual_information_is_non_negative(self, clinical_model) -> None:
        output = MCDropoutInference(clinical_model, n_passes=10).run(_FakeLoader())
        assert (output.mutual_information >= -1e-6).all()

    def test_entropy_decomposition_holds(self, clinical_model) -> None:
        output = MCDropoutInference(clinical_model, n_passes=10).run(_FakeLoader())
        np.testing.assert_allclose(
            output.mutual_information,
            output.predictive_entropy - output.expected_entropy,
            atol=1e-6,
        )

    def test_identifiers_are_preserved(self, clinical_model) -> None:
        output = MCDropoutInference(clinical_model, n_passes=3).run(_FakeLoader(n_batches=2))
        assert output.row_uid is not None
        assert len(output.row_uid) == 2 * BATCH
        assert output.targets is not None

    def test_frame_export_has_a_row_per_sample(self, clinical_model) -> None:
        output = MCDropoutInference(clinical_model, n_passes=3).run(_FakeLoader(n_batches=2))
        frame = output.to_frame(label_names=[f"b{i}" for i in range(N_LABELS)])
        assert len(frame) == 2 * BATCH
        assert "total_uncertainty" in frame.columns
        assert "prob_b0" in frame.columns

    def test_works_with_an_image_model(self) -> None:
        torch.manual_seed(0)
        model = OCTOnlyModel(n_labels=N_LABELS, pretrained=False, dropout=0.5)
        output = MCDropoutInference(model, n_passes=3).run(_FakeLoader(n_batches=1, with_image=True))
        assert output.mean_probability.shape == (BATCH, N_LABELS)


class TestSelectivePrediction:
    """Coverage curves and uncertainty/error association."""

    @staticmethod
    def _informative_case(n: int = 400, seed: int = 0):
        """Predictions whose error rate genuinely rises with uncertainty."""
        rng = np.random.default_rng(seed)
        targets = rng.integers(0, 2, (n, 3)).astype(float)
        # Noise must exceed 0.5 for some samples, otherwise every prediction is
        # correct, the error vector is constant, and Spearman is undefined.
        noise = rng.uniform(0.0, 0.9, n)
        probabilities = np.clip(
            targets * (1 - noise[:, None]) + (1 - targets) * noise[:, None]
            + rng.normal(0, 0.02, (n, 3)),
            0.001,
            0.999,
        )
        return targets, probabilities, noise

    def test_coverage_curve_shrinks_the_retained_set(self) -> None:
        targets, probabilities, uncertainty = self._informative_case()
        curve = SelectivePredictor().coverage_curve(
            targets, probabilities, uncertainty, coverage_levels=[1.0, 0.8, 0.5]
        )
        assert curve["n_retained"].is_monotonic_decreasing
        assert curve.loc[0, "n_retained"] == len(targets)

    def test_abstention_improves_the_retained_metric(self) -> None:
        targets, probabilities, uncertainty = self._informative_case()
        curve = SelectivePredictor().coverage_curve(
            targets, probabilities, uncertainty, coverage_levels=[1.0, 0.7]
        )
        full = curve.loc[curve["coverage"] == 1.0, "macro_f1"].iloc[0]
        retained = curve.loc[curve["coverage"] == 0.7, "macro_f1"].iloc[0]
        assert retained >= full, "abstaining on the most uncertain cases did not help"

    def test_uncertainty_error_association_is_detected(self) -> None:
        targets, probabilities, uncertainty = self._informative_case()
        association = SelectivePredictor().uncertainty_error_association(
            targets, probabilities, uncertainty
        )
        assert association["spearman_r"] > 0.2
        assert association["p_value"] < 0.05

    def test_example_selection_returns_the_four_categories(self) -> None:
        targets, probabilities, uncertainty = self._informative_case()
        examples = SelectivePredictor().select_examples(
            targets, probabilities, uncertainty, n_per_category=3
        )
        assert set(examples) == {
            "confidently_correct",
            "confidently_wrong",
            "uncertain_correct",
            "uncertain_wrong",
        }
        for indices in examples.values():
            assert len(indices) <= 3

    def test_coverage_levels_are_reported_verbatim(self) -> None:
        targets, probabilities, uncertainty = self._informative_case()
        levels = [1.0, 0.9, 0.8, 0.7]
        curve = SelectivePredictor().coverage_curve(
            targets, probabilities, uncertainty, coverage_levels=levels
        )
        assert curve["coverage"].tolist() == levels
