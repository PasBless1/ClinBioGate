# Data directory

**Nothing in `raw/` or `processed/` is tracked by git.** OLIVES is de-identified clinical trial
data; keep it local.

```
data/
├── raw/OLIVES/                        # the downloaded dataset (git-ignored)
│   ├── disease_classification/        # 32 parquet shards, 78,822 rows, 16 biomarkers
│   └── biomarker_detection/           # VIP-Cup 6-biomarker subset, own train/test split
├── processed/                         # exported image caches (git-ignored)
│   └── images_labelled/               # 9,396 PNGs of the modelling subset, 2.3 GB
└── manifests/                         # metadata-only manifests and split assignments
    ├── disease_classification_train.parquet
    └── splits/
        ├── split_holdout.json
        ├── split_holdout_patients.csv
        ├── split_holdout_prevalence.csv
        └── split_fold_{0..4}.json
```

## Getting the data

```bash
pip install huggingface_hub
huggingface-cli download gOLIVES/OLIVES_Dataset --repo-type dataset \
    --local-dir data/raw/OLIVES
```

Roughly 30 GB across both configs. To take only what the modelling needs:

```bash
huggingface-cli download gOLIVES/OLIVES_Dataset --repo-type dataset \
    --include "disease_classification/*" --local-dir data/raw/OLIVES
```

If the data is elsewhere, point `data.root` in `configs/data.yaml` at it rather than moving files.

## Which config to use

**`disease_classification`** — the default, and the one to use. All 16 biomarkers, BCVA, CST,
patient/eye IDs, disease label, every visit. It ships a single `train` split, so partitioning is
ours to define and can be patient-grouped.

**`biomarker_detection`** — the VIP-Cup 2023 subset (`B1`–`B6`). It ships its own train/test split,
but **that test split contains one patient**, which makes patient-level bootstrap CIs impossible
and produces a large prevalence shift against train (for example DRT/ME: 30.5% → 3.7%). Use it only
to reproduce the challenge, never as this project's evaluation split. The same six concepts are
available inside `disease_classification` via `target_set: six`, where our own grouped split
applies.

## Manifests

The manifest is a metadata-only view of the parquet: one row per scan carrying identifiers,
clinical values, labels, an image hash and derived columns (`has_biomarkers`, `visit_uid`,
`dup_group_id`). It holds **no pixels**, so it is safe to keep locally and cheap to analyse — about
3 MB standing in for 15 GB.

Each row is addressed by `(shard_index, row_in_shard)`, which is how `ParquetImageReader` and
`ImageCacheExporter` fetch the actual image later.

```bash
python scripts/build_manifest.py --config configs/data.yaml   # ~90 s, hashes every image
python scripts/audit_data.py --config configs/data.yaml       # audit + report
python scripts/make_splits.py --config configs/data.yaml --folds
```

Manifests are git-ignored too: they contain patient identifiers, and they are reproducible from the
raw data in about ninety seconds.

## Known data issues

Confirmed by running the audit against the local copy:

| Issue | Detail |
|---|---|
| Duplicate images | 16,562 rows (21%) are byte-identical to another row, in 8,281 groups. No group crosses a patient. Handled by `duplicates.policy`. |
| Missing clinical values | 637 rows have no BCVA/CST — all patient 79. Impute on train, keep the indicator. |
| Sparse `Scan (n/49)` | Present on only 22% of rows; it belongs to the annotation block. Filter on `has_biomarkers`. |
| No visit column | Visits are inferred from BCVA/CST change-points; 5 of 1,444 deviate from 49 scans. Inferred, not authoritative. |
| Absent modalities | No fundus, no 3D volumes, no DRSS/BMI/age/race/HbA1c. Those extensions need the Zenodo release. |
| Normalisation constants | Measured pixel mean ≈ 0.17, not the paper's 0.482. Do not reuse the paper's values. |

## Licence

CC BY 4.0. Derived from the PRIME and TREX-DME trials (Retina Consultants of Texas), collected
under IRB approval and de-identified per HIPAA. Single U.S. clinic, no untreated control group —
nothing here supports a diagnostic or deployment claim. Cite Prabhushankar et al., NeurIPS Datasets
and Benchmarks 2022.
