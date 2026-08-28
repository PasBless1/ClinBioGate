"""Shared fixtures: a synthetic OLIVES-shaped dataset that needs no real data.

The fixture reproduces the structural traps of the real mirror on purpose:
patients with two eyes, 49-scan visits, repeated visits, byte-identical duplicate
images, a patient with missing clinical values, and a label that occurs in only
one patient. Tests that pass on clean synthetic data prove nothing.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

SCANS_PER_VISIT = 49

LABEL_COLUMNS = [
    "Atrophy / thinning of retinal layers",
    "Disruption of EZ",
    "DRIL",
    "IR hemorrhages",
    "IR HRF",
    "Partially attached vitreous face",
    "Fully attached vitreous face",
    "Preretinal tissue/hemorrhage",
    "Vitreous debris",
    "VMT",
    "DRT/ME",
    "Fluid (IRF)",
    "Fluid (SRF)",
    "Disruption of RPE",
    "PED (serous)",
    "SHRM",
]


def _png_bytes(seed: int, size: tuple[int, int] = (32, 32)) -> bytes:
    """Deterministic small PNG, so identical seeds give identical bytes."""
    rng = np.random.default_rng(seed)
    array = (rng.random(size) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def build_synthetic_frame(
    n_patients: int = 8,
    visits_per_eye: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Build an OLIVES-shaped table with the real dataset's structural quirks."""
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    image_seed = 0

    for patient in range(1, n_patients + 1):
        disease = float(patient % 2)
        n_eyes = 2 if patient % 4 == 0 else 1
        # Patient 3 mirrors the real patient 79: clinical values missing.
        clinical_missing = patient == 3

        for eye_offset in range(n_eyes):
            eye = float(patient * 10 + eye_offset)
            visit_images: list[list[bytes]] = []

            for visit in range(visits_per_eye):
                bcva = None if clinical_missing else float(70 + 2 * visit + patient)
                cst = None if clinical_missing else float(300 - 5 * visit + patient)
                labelled = visit in (0, visits_per_eye - 1)

                # Visit 2 of patient 5 repeats visit 1's images byte-for-byte.
                repeats_earlier = patient == 5 and visit == 2 and visit_images
                images = (
                    visit_images[1]
                    if repeats_earlier
                    else [_png_bytes(image_seed + i) for i in range(SCANS_PER_VISIT)]
                )
                if not repeats_earlier:
                    image_seed += SCANS_PER_VISIT
                visit_images.append(images)

                for scan in range(SCANS_PER_VISIT):
                    row: dict = {
                        "Image": {"bytes": images[scan], "path": f"{scan}.png"},
                        "Scan (n/49)": float(scan + 1) if labelled else None,
                        "Eye_ID": eye,
                        "BCVA": bcva,
                        "CST": cst,
                        "Patient_ID": patient,
                        "Disease Label": disease,
                    }
                    for index, column in enumerate(LABEL_COLUMNS):
                        if not labelled:
                            row[column] = None
                        elif column == "PED (serous)":
                            # Rare label confined to one patient.
                            row[column] = 1.0 if patient == 2 else 0.0
                        elif column == "IR HRF":
                            row[column] = 1.0 if rng.random() < 0.7 else 0.0
                        else:
                            row[column] = 1.0 if rng.random() < 0.25 * (1 - index / 40) else 0.0
                    records.append(row)

                    # Patient 1 has every row emitted twice (adjacent duplication).
                    if patient == 1:
                        records.append(dict(row))

    return pd.DataFrame(records)


def write_synthetic_dataset(
    root: Path,
    config_name: str = "disease_classification",
    n_shards: int = 2,
    **kwargs,
) -> Path:
    """Write the synthetic frame as sharded parquet under ``root/config_name``."""
    frame = build_synthetic_frame(**kwargs)
    directory = root / config_name
    directory.mkdir(parents=True, exist_ok=True)

    chunks = np.array_split(np.arange(len(frame)), n_shards)
    for index, chunk in enumerate(chunks):
        table = pa.Table.from_pandas(frame.iloc[chunk].reset_index(drop=True), preserve_index=False)
        pq.write_table(
            table,
            directory / f"train-{index:05d}-of-{n_shards:05d}.parquet",
            row_group_size=64,
        )
    return directory


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def label_schema_path(repo_root: Path) -> Path:
    """The real label schema, which the synthetic data deliberately matches."""
    return repo_root / "configs" / "label_schema.yaml"


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """A synthetic OLIVES parquet dataset written once per test session."""
    root = tmp_path_factory.mktemp("olives_synthetic")
    write_synthetic_dataset(root)
    return root


@pytest.fixture(scope="session")
def schema(label_schema_path: Path):
    """16-label schema over the disease_classification config."""
    from olives_biomarkers.data.schema import LabelSchema

    return LabelSchema.from_yaml(label_schema_path, "sixteen", "disease_classification")


@pytest.fixture(scope="session")
def manifest(synthetic_root: Path, schema):
    """Manifest built from the synthetic dataset."""
    from olives_biomarkers.data.manifests import ManifestBuilder

    builder = ManifestBuilder(synthetic_root, schema, compute_image_hashes=True)
    return builder.build(split="train", progress=False)


@pytest.fixture(scope="session")
def modelling_frame(manifest) -> pd.DataFrame:
    """Deduplicated, biomarker-labelled rows."""
    return manifest.modelling_frame(policy="keep_first", labelled_only=True)
