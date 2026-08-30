"""Persistent checkpoint and model-bundle behavior."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from olives_biomarkers.config import TrainingConfig  # noqa: E402
from olives_biomarkers.models import ClinicalOnlyModel  # noqa: E402
from olives_biomarkers.training import MaskedBCEWithLogitsLoss, Trainer  # noqa: E402


def _loader() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "clinical": torch.tensor(
                [
                    [0.1, 0.2, 0.0, 0.0],
                    [0.4, -0.1, 0.0, 1.0],
                    [-0.3, 0.8, 1.0, 0.0],
                    [0.9, 0.3, 0.0, 0.0],
                    [-0.5, -0.2, 1.0, 1.0],
                    [0.2, 0.6, 0.0, 0.0],
                ]
            ),
            "target": torch.tensor(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ),
            "row_uid": torch.arange(6),
            "patient_id": torch.arange(6),
        }
    ]


def _trainer(tmp_path, run_id: str = "resume_test") -> Trainer:
    return Trainer(
        model=ClinicalOnlyModel(clinical_dim=4, n_labels=2, dropout=0.1),
        criterion=MaskedBCEWithLogitsLoss(),
        config=TrainingConfig(
            epochs=2,
            batch_size=6,
            amp=False,
            early_stopping_patience=5,
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        run_id=run_id,
        label_names=["label_a", "label_b"],
    )


def test_resume_continues_at_next_epoch_with_full_state(tmp_path) -> None:
    first = _trainer(tmp_path)
    first.fit(_loader(), _loader(), epochs=1)

    last = first.checkpoints.last_path
    best = first.checkpoints.best_path
    assert last.is_file() and best.is_file()
    assert not list(last.parent.glob(".*.tmp"))

    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert payload["checkpoint_version"] == 2
    assert payload["epoch"] == 1
    assert payload["optimizer_state_dict"]
    assert payload["early_stopping_state"]["best_epoch"] == 1
    assert len(payload["history_records"]) == 1
    assert payload["rng_state"]["torch_cpu"] is not None

    resumed = _trainer(tmp_path)
    resumed.fit(_loader(), _loader(), epochs=2, resume=True)

    assert [record["epoch"] for record in resumed.history.records] == [1, 2]
    final = torch.load(last, map_location="cpu", weights_only=False)
    assert final["epoch"] == 2
    assert len(final["history_records"]) == 2


def test_best_model_bundle_is_saved_separately(tmp_path) -> None:
    trainer = _trainer(tmp_path, run_id="bundle_test")
    trainer.fit(_loader(), _loader(), epochs=1)
    trainer.load_best()

    path = trainer.export_model(
        tmp_path / "models" / "bundle_test_model.pt",
        metadata={"label_names": ["label_a", "label_b"]},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["format_version"] == 1
    assert payload["run_id"] == "bundle_test"
    assert payload["metadata"]["label_names"] == ["label_a", "label_b"]
    assert payload["model_state_dict"]
