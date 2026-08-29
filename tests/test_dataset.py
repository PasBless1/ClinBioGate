"""Schema, manifest, preprocessing and dataset tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olives_biomarkers.data.manifests import ManifestBuilder, ParquetShardIndex, VisitInferencer
from olives_biomarkers.data.preprocessing import (
    ClinicalPreprocessor,
    NotFittedError,
    PosWeightCalculator,
)
from olives_biomarkers.data.schema import LabelSchema, SchemaValidationError


class TestLabelSchema:
    """Schema resolution and validation."""

    def test_sixteen_label_set_resolves(self, label_schema_path) -> None:
        schema = LabelSchema.from_yaml(label_schema_path, "sixteen", "disease_classification")
        assert schema.n_labels == 16
        assert "irhrf" in schema.label_keys

    def test_six_label_mvp_resolves(self, label_schema_path) -> None:
        schema = LabelSchema.from_yaml(label_schema_path, "six", "disease_classification")
        assert schema.n_labels == 6
        assert set(schema.label_keys) == {"irhrf", "pavf", "favf", "irf", "drt_me", "vitreous_debris"}

    def test_six_and_sixteen_share_source_columns(self, label_schema_path) -> None:
        six = LabelSchema.from_yaml(label_schema_path, "six", "disease_classification")
        sixteen = LabelSchema.from_yaml(label_schema_path, "sixteen", "disease_classification")
        assert set(six.source_columns).issubset(set(sixteen.source_columns))

    def test_rejects_an_unknown_target_set(self, label_schema_path) -> None:
        with pytest.raises(ValueError, match="target_set"):
            LabelSchema.from_yaml(label_schema_path, "twelve", "disease_classification")

    def test_missing_columns_raise_with_an_actionable_message(self, schema) -> None:
        with pytest.raises(SchemaValidationError, match="missing"):
            schema.validate_columns(["Patient_ID", "BCVA"])

    def test_records_fields_absent_from_this_source(self, schema) -> None:
        assert "fundus_image" in schema.unresolved_fields
        assert schema.unresolved_fields["visit_id_or_week"]["status"] == "absent"


class TestParquetShardIndex:
    """Shard discovery."""

    def test_finds_shards(self, synthetic_root) -> None:
        index = ParquetShardIndex(synthetic_root, "disease_classification")
        assert len(index.shards("train")) == 2

    def test_missing_directory_message_names_the_expected_path(self, tmp_path) -> None:
        index = ParquetShardIndex(tmp_path, "disease_classification")
        with pytest.raises(FileNotFoundError, match="expected directory"):
            index.require_exists()


class TestManifest:
    """Manifest construction and derived columns."""

    def test_row_count_matches_the_parquet(self, manifest, synthetic_root) -> None:
        index = ParquetShardIndex(synthetic_root, "disease_classification")
        assert len(manifest) == index.total_rows("train")

    def test_has_the_expected_columns(self, manifest) -> None:
        for column in ("row_uid", "shard_index", "row_in_shard", "patient_id", "eye_id",
                       "image_hash", "has_biomarkers", "visit_uid", "dup_group_id"):
            assert column in manifest.frame.columns

    def test_row_uid_is_unique(self, manifest) -> None:
        assert manifest.frame["row_uid"].is_unique

    def test_shard_row_pairs_address_rows_uniquely(self, manifest) -> None:
        pairs = manifest.frame[["shard_index", "row_in_shard"]]
        assert not pairs.duplicated().any()

    def test_labelled_rows_have_complete_vectors(self, manifest) -> None:
        labelled = manifest.labelled()
        assert not labelled[manifest.label_columns].isna().any().any()

    def test_labels_are_binary(self, manifest) -> None:
        values = manifest.labelled()[manifest.label_columns].to_numpy()
        assert np.isin(values, [0.0, 1.0]).all()

    def test_detects_the_planted_duplicates(self, manifest) -> None:
        # Patient 1's rows are all emitted twice in the fixture.
        assert manifest.frame["is_duplicate"].sum() > 0
        patient_one = manifest.frame[manifest.frame["patient_id"] == 1]
        assert patient_one["is_duplicate"].all()

    def test_keep_first_halves_the_duplicated_patient(self, manifest) -> None:
        deduplicated = manifest.deduplicated("keep_first")
        assert len(deduplicated) < len(manifest.frame)
        assert (deduplicated["dup_rank"] == 0).all()

    def test_missing_clinical_values_are_flagged(self, manifest) -> None:
        # Patient 3 has no BCVA/CST in the fixture.
        missing = manifest.frame[manifest.frame["bcva_missing"]]
        assert set(missing["patient_id"].unique()) == {3}

    def test_save_and_load_round_trip(self, manifest, tmp_path, schema) -> None:
        from olives_biomarkers.data.manifests import Manifest

        path = manifest.save(tmp_path / "m.parquet")
        restored = Manifest.load(path, schema=schema)
        assert len(restored) == len(manifest)
        pd.testing.assert_frame_equal(restored.frame, manifest.frame)


class TestVisitInferencer:
    """Visit segmentation from BCVA/CST change-points."""

    def test_assigns_visit_columns(self, manifest) -> None:
        for column in ("visit_index", "visit_uid", "is_adjacent_duplicate", "visit_size"):
            assert column in manifest.frame.columns

    def test_finds_the_expected_visits_per_eye(self, manifest) -> None:
        # The fixture writes 3 visits per eye.
        per_eye = manifest.frame.groupby(["patient_id", "eye_id"])["visit_index"].nunique()
        assert set(per_eye.unique()) == {3}

    def test_a_visit_holds_49_unique_images(self, manifest) -> None:
        unique_per_visit = manifest.frame.groupby("visit_uid")["image_hash"].nunique()
        assert (unique_per_visit == 49).all()

    def test_flags_adjacent_duplicates_without_splitting_the_visit(self, manifest) -> None:
        patient_one = manifest.frame[manifest.frame["patient_id"] == 1]
        assert patient_one["is_adjacent_duplicate"].sum() > 0
        assert patient_one["visit_index"].nunique() == 3

    def test_requires_identifier_columns(self) -> None:
        with pytest.raises(SchemaValidationError):
            VisitInferencer().assign(pd.DataFrame({"bcva": [1.0]}))


class TestVisitUidWithMissingIdentifiers:
    """Building a visit id must not depend on the installed pandas version.

    Regression guard: ``eye_id`` is float64 and NaN for the one patient with no
    clinical measurements. ``astype("Int64")`` on that column raises
    ``IntCastingNaNError`` on older pandas but succeeds on 2.2, so the manifest
    built locally and failed on Colab against identical data.
    """

    @pytest.mark.parametrize(
        "series,expected",
        [
            (pd.Series([58.0, np.nan, 60.0]), ["58", "<NA>", "60"]),
            (pd.Series([1, 2], dtype="int64"), ["1", "2"]),
            (pd.Series([1, pd.NA], dtype="Int64"), ["1", "<NA>"]),
            (pd.Series([np.nan, np.nan]), ["<NA>", "<NA>"]),
        ],
    )
    def test_id_component_renders_every_dtype(self, series, expected) -> None:
        assert VisitInferencer._id_component(series).tolist() == expected

    def test_assign_survives_a_missing_eye_id(self) -> None:
        frame = pd.DataFrame(
            {
                "patient_id": [79, 79, 79, 79],
                "eye_id": [np.nan] * 4,
                "bcva": [np.nan, np.nan, 70.0, 70.0],
                "cst": [np.nan, np.nan, 250.0, 250.0],
                "scan_number": [1, 2, 1, 2],
            }
        )
        out = VisitInferencer().assign(frame)
        assert list(out["visit_uid"]) == ["79_E<NA>_V0"] * 2 + ["79_E<NA>_V1"] * 2

    def test_missing_clinical_pair_does_not_open_a_new_visit(self) -> None:
        """NaN must compare equal to NaN, or every unmeasured scan starts a visit."""
        frame = pd.DataFrame(
            {
                "patient_id": [79] * 4,
                "eye_id": [1.0] * 4,
                "bcva": [np.nan] * 4,
                "cst": [np.nan] * 4,
                "scan_number": [1, 2, 3, 4],
            }
        )
        out = VisitInferencer().assign(frame)
        assert out["visit_index"].nunique() == 1


class TestClinicalPreprocessor:
    """Fit-on-train-only enforcement."""

    def test_transform_before_fit_raises(self, modelling_frame) -> None:
        with pytest.raises(NotFittedError):
            ClinicalPreprocessor().transform(modelling_frame)

    def test_silent_refit_is_blocked(self, modelling_frame) -> None:
        pre = ClinicalPreprocessor().fit(modelling_frame)
        with pytest.raises(RuntimeError, match="already fitted"):
            pre.fit(modelling_frame)
        pre.fit(modelling_frame, allow_refit=True)

    def test_output_width_includes_missingness_indicators(self, modelling_frame) -> None:
        pre = ClinicalPreprocessor(["bcva", "cst"], use_missingness_indicators=True)
        features = pre.fit_transform(modelling_frame)
        assert features.shape == (len(modelling_frame), 4)
        assert pre.output_dim == 4

    def test_indicators_can_be_disabled(self, modelling_frame) -> None:
        pre = ClinicalPreprocessor(["bcva", "cst"], use_missingness_indicators=False)
        assert pre.fit_transform(modelling_frame).shape[1] == 2

    def test_statistics_come_only_from_the_fitted_rows(self, modelling_frame) -> None:
        patients = sorted(modelling_frame["patient_id"].unique())
        train = modelling_frame[modelling_frame["patient_id"].isin(patients[:4])]
        held_out = modelling_frame[modelling_frame["patient_id"].isin(patients[4:])]

        pre = ClinicalPreprocessor().fit(train)
        expected = float(pd.to_numeric(train["bcva"], errors="coerce").median())
        assert pre.state is not None
        assert pre.state.medians["bcva"] == pytest.approx(expected)

        # Transforming held-out rows must not change the fitted statistics.
        pre.transform(held_out)
        assert pre.state.medians["bcva"] == pytest.approx(expected)

    def test_missing_values_are_imputed_and_flagged(self, modelling_frame) -> None:
        pre = ClinicalPreprocessor(["bcva", "cst"])
        features = pre.fit_transform(modelling_frame)
        assert not np.isnan(features).any()
        missing_rows = modelling_frame["bcva"].isna().to_numpy()
        if missing_rows.any():
            assert (features[missing_rows, 2] == 1.0).all()

    def test_save_and_load_round_trip(self, modelling_frame, tmp_path) -> None:
        pre = ClinicalPreprocessor().fit(modelling_frame)
        path = pre.save(tmp_path / "clinical.json")
        restored = ClinicalPreprocessor.load(path)
        np.testing.assert_allclose(
            pre.transform(modelling_frame), restored.transform(modelling_frame)
        )


class TestPosWeightCalculator:
    """Class weights from the training fold."""

    def test_weight_is_negatives_over_positives(self) -> None:
        labels = np.array([[1, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
        weights = PosWeightCalculator(cap=100).compute(labels)
        assert weights[0] == pytest.approx(3.0)

    def test_weights_are_capped(self) -> None:
        labels = np.zeros((1000, 1), dtype=np.float32)
        labels[0] = 1
        assert PosWeightCalculator(cap=20).compute(labels)[0] == pytest.approx(20.0)

    def test_label_with_no_positives_gets_weight_one(self) -> None:
        labels = np.zeros((10, 2), dtype=np.float32)
        labels[:, 0] = 1
        assert PosWeightCalculator().compute(labels)[1] == pytest.approx(1.0)


class TestOlivesDataset:
    """Dataset item construction."""

    def test_items_align_image_clinical_and_target(self, manifest, modelling_frame) -> None:
        from olives_biomarkers.data.dataset import OlivesDataset, ParquetImageReader
        from olives_biomarkers.data.preprocessing import ImageTransformFactory

        pytest.importorskip("torch")
        pytest.importorskip("torchvision")

        reader = ParquetImageReader(
            manifest.frame.attrs.get("data_root", ""), "disease_classification"
        ) if False else None

        pre = ClinicalPreprocessor().fit(modelling_frame)
        dataset = OlivesDataset(
            frame=modelling_frame,
            label_columns=manifest.label_columns,
            clinical_preprocessor=pre,
            transform=ImageTransformFactory((32, 32)).build(train=False),
            reader=reader,
            return_image=False,
        )
        item = dataset[0]
        assert item["target"].shape == (len(manifest.label_columns),)
        assert item["clinical"].shape == (4,)
        assert item["row_uid"] == int(modelling_frame.iloc[0]["row_uid"])
        assert item["patient_id"] == int(modelling_frame.iloc[0]["patient_id"])

    def test_requires_an_image_source_when_images_are_requested(
        self, manifest, modelling_frame
    ) -> None:
        from olives_biomarkers.data.dataset import OlivesDataset

        with pytest.raises(ValueError, match="cache_path"):
            OlivesDataset(
                frame=modelling_frame,
                label_columns=manifest.label_columns,
                return_image=True,
            )


class TestParquetImageReader:
    """Random access into the shards."""

    def test_reads_a_decodable_image(self, synthetic_root, manifest) -> None:
        from olives_biomarkers.data.dataset import ParquetImageReader

        reader = ParquetImageReader(synthetic_root, "disease_classification", "train")
        row = manifest.frame.iloc[5]
        image = reader.read_image(int(row["shard_index"]), int(row["row_in_shard"]))
        assert image.size == (32, 32)

    def test_bytes_match_the_manifest_hash(self, synthetic_root, manifest) -> None:
        import hashlib

        from olives_biomarkers.data.dataset import ParquetImageReader

        reader = ParquetImageReader(synthetic_root, "disease_classification", "train")
        for position in (0, 50, 120):
            row = manifest.frame.iloc[position]
            payload = reader.read_bytes(int(row["shard_index"]), int(row["row_in_shard"]))
            digest = hashlib.blake2b(payload, digest_size=12).hexdigest()
            assert digest == row["image_hash"], "reader returned a different image than indexed"


class TestExperimentRunnerWiring:
    """Config knobs must actually reach the transform factory and the loader.

    Regression guard: ``build_data_module`` previously constructed
    ``ImageTransformFactory(image_size)`` and nothing else, so ``crop_retina``,
    ``image_mode``, ``normalization`` and the sampler settings were silently
    ignored no matter what the config said.
    """

    @staticmethod
    def _runner(synthetic_root, repo_root, **data_overrides):
        from olives_biomarkers.config import ConfigLoader
        from olives_biomarkers.experiment import ExperimentRunner
        from olives_biomarkers.pipeline import OlivesPipeline

        config = ConfigLoader(repo_root).load(repo_root / "configs" / "data.yaml")
        for key, value in data_overrides.items():
            target = config.training if hasattr(config.training, key) else config.data
            setattr(target, key, value)

        from olives_biomarkers.utils.environment import RuntimeEnvironment

        pipeline = OlivesPipeline.__new__(OlivesPipeline)
        pipeline.config = config
        pipeline.data_root = synthetic_root
        pipeline.env = RuntimeEnvironment.detect(repo_root)
        pipeline.output_dir = repo_root / "outputs"
        return ExperimentRunner(pipeline, config)

    def test_transform_options_are_propagated(self, synthetic_root, repo_root) -> None:
        runner = self._runner(
            synthetic_root,
            repo_root,
            crop_retina=True,
            preserve_aspect_ratio=True,
            image_mode="adjacent",
            normalization="olives",
            horizontal_flip=True,
            image_size=(320, 320),
        )
        from olives_biomarkers.data.preprocessing import ImageTransformFactory

        factory = ImageTransformFactory(
            image_size=runner.config.data.image_size,
            image_mode=runner.config.data.image_mode,
            crop_retina=runner.config.data.crop_retina,
            preserve_aspect_ratio=runner.config.data.preserve_aspect_ratio,
            normalization=runner.config.data.normalization,
            horizontal_flip=runner.config.data.horizontal_flip,
        )
        assert factory.crop_retina is True
        assert factory.preserve_aspect_ratio is True
        assert factory.image_mode == "adjacent"
        assert factory.image_size == (320, 320)

    def test_data_module_receives_sampler_settings(
        self, synthetic_root, repo_root, manifest, modelling_frame
    ) -> None:
        from olives_biomarkers.data.splits import PatientGroupedSplitter

        runner = self._runner(
            synthetic_root, repo_root, sampler="patient", rare_positive_sampling_power=0.5
        )
        runner.config.model.name = "clinical_only"   # skip image I/O
        assignment = PatientGroupedSplitter(seed=42).split(modelling_frame)
        module = runner.build_data_module(
            modelling_frame, assignment, manifest.label_columns
        )
        assert module.sampler == "patient"
        assert module.rare_positive_sampling_power == 0.5

    def test_patient_sampler_is_actually_used_by_the_train_loader(
        self, synthetic_root, repo_root, manifest, modelling_frame
    ) -> None:
        from olives_biomarkers.data.dataset import GroupBalancedSampler
        from olives_biomarkers.data.splits import PatientGroupedSplitter

        runner = self._runner(synthetic_root, repo_root, sampler="patient")
        runner.config.model.name = "clinical_only"
        assignment = PatientGroupedSplitter(seed=42).split(modelling_frame)
        module = runner.build_data_module(modelling_frame, assignment, manifest.label_columns)

        train_loader = module.dataloader("train")
        assert isinstance(train_loader.sampler, GroupBalancedSampler)
        # Evaluation partitions must stay deterministic and unbalanced.
        assert not isinstance(module.dataloader("test", shuffle=False).sampler, GroupBalancedSampler)

    def test_asl_config_reaches_the_criterion(self, synthetic_root, repo_root) -> None:
        from olives_biomarkers.training.losses import AsymmetricLoss

        runner = self._runner(synthetic_root, repo_root)
        runner.config.training.loss = "asl"
        runner.config.training.asl_gamma_negative = 3.0
        runner.config.training.asl_clip = 0.1
        criterion = runner.build_criterion(pos_weight=None)
        assert isinstance(criterion, AsymmetricLoss)
        assert criterion.gamma_negative == 3.0
        assert criterion.clip == 0.1

    def test_bce_still_receives_class_weights(self, synthetic_root, repo_root) -> None:
        from olives_biomarkers.training.losses import MaskedBCEWithLogitsLoss

        runner = self._runner(synthetic_root, repo_root)
        runner.config.training.loss = "bce"
        criterion = runner.build_criterion(pos_weight=np.full(16, 3.0, dtype=np.float32))
        assert isinstance(criterion, MaskedBCEWithLogitsLoss)
        assert criterion.pos_weight is not None


class TestClinicalFramePreparation:
    """The within-eye features and the control ladder must reach the loader.

    ``prepare_clinical_frame`` runs inside ``build_data_module``, so a config
    that asks for longitudinal features but never receives them would train an
    ordinary fusion model while claiming to test the longitudinal hypothesis.
    """

    @staticmethod
    def _runner(synthetic_root, repo_root, **data_overrides):
        return TestExperimentRunnerWiring._runner(synthetic_root, repo_root, **data_overrides)

    def test_default_config_is_a_no_op(self, synthetic_root, repo_root, modelling_frame) -> None:
        runner = self._runner(synthetic_root, repo_root)
        out = runner.prepare_clinical_frame(modelling_frame)
        assert "cst_delta" not in out.columns
        assert out["cst"].equals(modelling_frame["cst"])

    def test_longitudinal_flag_adds_the_contrast_columns(
        self, synthetic_root, repo_root, modelling_frame
    ) -> None:
        runner = self._runner(synthetic_root, repo_root, longitudinal_clinical=True)
        out = runner.prepare_clinical_frame(modelling_frame)
        for column in ("cst_delta", "bcva_delta", "cst_baseline", "is_baseline_visit"):
            assert column in out.columns

    def test_unknown_clinical_feature_names_the_available_ones(
        self, synthetic_root, repo_root, modelling_frame
    ) -> None:
        runner = self._runner(synthetic_root, repo_root, longitudinal_clinical=True)
        runner.config.model.clinical_features = ["bcva", "cst_slope"]
        with pytest.raises(KeyError, match="cst_delta"):
            runner.prepare_clinical_frame(modelling_frame)

    def test_perturbation_reaches_the_frame(
        self, synthetic_root, repo_root, modelling_frame
    ) -> None:
        runner = self._runner(synthetic_root, repo_root, clinical_perturbation="patient_mean")
        out = runner.prepare_clinical_frame(modelling_frame)
        # One value per patient. The synthetic cohort includes a patient whose
        # clinical values are entirely missing, and whose mean is therefore NaN;
        # nunique() is 0 there, not 1, and that is the correct outcome.
        distinct = out.groupby("patient_id")["cst"].nunique()
        assert (distinct <= 1).all()
        assert (distinct == 1).sum() >= out["patient_id"].nunique() - 1

    def test_perturbation_is_applied_before_the_deltas(
        self, synthetic_root, repo_root, modelling_frame
    ) -> None:
        """A control arm must not keep honest deltas beside perturbed absolutes."""
        runner = self._runner(
            synthetic_root,
            repo_root,
            longitudinal_clinical=True,
            clinical_perturbation="patient_mean",
        )
        out = runner.prepare_clinical_frame(modelling_frame)
        assert np.allclose(out["cst_delta"].dropna(), 0.0)
