# Clinically Grounded, Uncertainty-Aware Multimodal Retinal Biomarker Detection

Multilabel detection of retinal biomarkers from OCT B-scans on the **OLIVES** dataset, fusing the
scan with the clinical measurements recorded at the same visit (BCVA, CST), with calibrated
probabilities, uncertainty estimates and per-label explanations.

**Owner:** Blessing Asare · **Framework:** PyTorch · **Status:** Phases 0–5 implemented; headline numbers pending a GPU run

---

## Research question

> Can clinically grounded fusion of OCT features with BCVA and CST improve multilabel retinal
> biomarker detection, and produce better-calibrated, more reliable predictions than an OCT-only
> model on OLIVES?

A rigorous negative result — clinical fusion failing to beat a well-controlled OCT baseline — is a
legitimate and useful outcome. The evaluation is built so that outcome would be believable.

---

## Quick start

```bash
# 1. Install
python -m pip install -e ".[dev]"

# 2. Audit the data (builds the manifest on first run, ~90 s)
python scripts/audit_data.py --config configs/data.yaml

# 3. Patient-grouped splits
python scripts/make_splits.py --config configs/data.yaml --folds

# 4. Export the image cache once (~2.3 GB; also what you upload to Drive for Colab)
python -c "from olives_biomarkers import OlivesPipeline; OlivesPipeline.from_config('configs/data.yaml').export_image_cache()"

# 5. Run the mandatory A-D comparison
python scripts/run_comparison.py --budget local_cpu --evaluate      # CPU: reduced budget
python scripts/run_comparison.py --budget colab_gpu --seeds 42 43 44 --evaluate

# 6. Explore
jupyter lab notebooks/
```

### Notebooks

| Notebook | Phase | Covers |
|---|---|---|
| `01_data_audit_and_eda.ipynb` | 0-1 | Schema, manifest, audit, extensive EDA, split design |
| `02_baselines.ipynb` | 2 | Models A, B, C with pre-flight wiring checks |
| `03_gated_fusion.ipynb` | 3 | Model D, gate analysis, ablations |
| `04_uncertainty_calibration.ipynb` | 4 | MC dropout, temperature scaling, selective prediction |
| `05_explainability_and_report.ipynb` | 5 | Grad-CAM, attention sanity, bootstrap CIs, report |

Each notebook imports classes from `src/`; none reimplements logic. With `%autoreload 2`, editing
a `.py` in VS Code and re-running a cell picks up the change.

Expected data location, configurable via `data.root` in `configs/data.yaml`:

```
data/raw/OLIVES/
├── disease_classification/    # 32 parquet shards, 78,822 rows, 16 biomarkers
└── biomarker_detection/       # VIP-Cup 6-biomarker subset with its own train/test split
```

Download from [`gOLIVES/OLIVES_Dataset`](https://huggingface.co/datasets/gOLIVES/OLIVES_Dataset).
If the data is absent, every entry point exits with the exact path it expected.

---

## What the audit found

Established by running the pipeline over the local copy, not assumed:

| Finding | Consequence |
|---|---|
| 78,822 rows, but only **9,396** deduplicated biomarker-labelled scans from **87 patients** | The effective sample size is the patient count. Patient-level bootstrap CIs only. |
| **21% of rows are byte-identical duplicates** (8,281 groups) | Deduplicate before counting or splitting. No group crosses a patient, so patient-grouping contains them. |
| Biomarkers graded on exactly **2 visits per eye** (first and last) | Matches the paper's protocol; the first/last contrast is a treatment-induced domain shift. |
| Prevalence spans **67% (IRHRF) to 0.06% (serous PED)** | Per-label thresholds, not 0.5. Five labels cannot support a test-set metric. |
| `Scan (n/49)` is present on only **22%** of rows | It belongs to the annotation block. Filter modelling rows on `has_biomarkers`. |
| **No visit/week column**; no fundus; no 3D volumes | Visits are *inferred* from BCVA/CST change-points. Fundus and volume extensions are blocked. |
| BCVA/CST missing for **patient 79 only** | Concentrated, so missingness is informative — impute on train, keep an indicator. |
| Measured pixel mean ≈ **0.17**, not the paper's 0.482 | Do not reuse the paper's normalisation constants. |

Full report: `outputs/reports/data_audit.md` (regenerate with `scripts/audit_data.py`).

Two independent cross-checks that the pipeline agrees with the published dataset: 9,396 labelled
scans against the paper's 9,408, and 292 unique label vectors against the paper's 286.

---

## Design decisions that matter

**Leakage control is structural, not procedural.** Splitting partitions patients; `SplitValidator`
re-verifies disjointness and duplicate containment on every split, and the test suite constructs
deliberately leaky splits to confirm the validator rejects them.

**Preprocessing cannot silently leak.** `ClinicalPreprocessor` raises if you transform before
fitting, and raises again if you re-fit on a second partition without saying so explicitly.

**Rare labels return `nan`, never a number.** A label with no positives in a fold has an undefined
AUROC; macro averages skip it and report how many were skipped.

**The gate cannot erase the image.** `ClinicalGate` zeroes its output weights and picks the bias
that makes the applied scale exactly 1.0, so Model D starts as a pure pass-through of the OCT
embedding and has to *learn* to modulate. `GateAnalyzer` then checks after training whether it
actually did — a gate stuck at identity means Model D is Model C with extra parameters, whatever
the metric says.

**MC dropout reactivates dropout only.** Batch-norm stays in eval mode; a test asserts the running
buffers do not move during stochastic passes.

---

## Repository layout

```
configs/                     YAML: data, label schema, one file per model
├── label_schema.yaml        Verbatim parquet columns → canonical keys; absent fields recorded
notebooks/
└── 01_data_audit_and_eda.ipynb    Phase 0 analysis front-end
src/olives_biomarkers/
├── config.py                Typed config dataclasses + `defaults:` inheritance
├── pipeline.py              OlivesPipeline — data, audit, splits, image cache
├── experiment.py            ExperimentRunner · RunEvaluator · ExperimentSuite · ResultsAggregator
├── data/                    schema · manifests · audit · splits · preprocessing · dataset
├── models/                  encoders · heads · baselines (A–C) · gated fusion (D) · registry
├── training/                losses · callbacks · engine
├── evaluation/              metrics · bootstrap · calibration · uncertainty · gating ·
│                            explainability · plots
├── eda/                     analyzer · plots
└── utils/                   environment · logging · reproducibility · io
scripts/                     audit_data · build_manifest · make_splits · train ·
                             run_comparison · evaluate · generate_report
tests/                       synthetic fixture reproducing the real dataset's traps
```

`ExperimentRunner` owns the ordering that keeps results honest: preprocessing and class weights
fit on train, thresholds fit on validation and then frozen, calibration fit on a patient-disjoint
partition, and test evaluated exactly once. Scripts and notebooks both call it, so they cannot
drift apart.

Everything is a class, and the notebook imports rather than reimplements. With `%autoreload 2`,
editing a `.py` file and re-running a cell picks up the change.

---

## Models

| ID | Model | Purpose |
|---|---|---|
| A | `clinical_only` | How much biomarker signal is in BCVA + CST alone |
| B | `oct_only` | The imaging baseline every fusion claim is measured against |
| C | `concat_fusion` | Ordinary feature concatenation — the control for D |
| D | `gated_fusion` | **Proposed:** clinical embedding gates the OCT features |

The hypothesis behind D is that clinical measurements say *how to read* a scan rather than what the
answer is — high CST should raise the weight on fluid-related image features. Concatenation cannot
express that interaction; a multiplicative gate can.

One caveat the EDA makes concrete: CST is a single value per *volume*, identical across all 49
B-scans of a visit, while the biomarkers vary slice to slice. Clinical features can shift a scan's
prior but cannot explain within-volume variation, which bounds the effect size to expect from C
and D.

---

## Running on Colab from VS Code

The code detects the runtime and resolves paths accordingly, so nothing needs editing.

```python
pipeline = OlivesPipeline.from_config("configs/data.yaml")
pipeline.env.mount_drive()          # no-op locally
```

Do not upload the 30 GB of parquet. Export just the modelling subset once, locally:

```python
from olives_biomarkers.data.dataset import ImageCacheExporter

exporter = ImageCacheExporter(pipeline.data_root, "disease_classification",
                              output_dir="data/processed/images_labelled")
exporter.export(eda.labelled)       # 9,396 PNGs, 2.3 GB
```

Upload that folder to Drive and set `data.colab.drive_data_subpath`.

---

## Testing

```bash
pytest                    # full suite
pytest tests/test_splits.py -v      # the leakage tests
```

Tests run against a synthetic dataset built by `tests/conftest.py` that deliberately reproduces the
real traps: two-eyed patients, 49-scan visits, a repeated visit, adjacent duplicate rows, a patient
with missing clinical values, and a label confined to one patient. No real data required.

---

## Status

- [x] **Phase 0** — repository, config, schema resolution, manifest, audit, EDA notebook
- [x] **Phase 1** — patient-grouped holdout and 5-fold splits, verified leakage-free
- [x] **Phase 2** — baselines A, B, C (`02_baselines.ipynb`, `scripts/run_comparison.py`)
- [x] **Phase 3** — gated fusion D, gate analysis, ablations (`03_gated_fusion.ipynb`)
- [x] **Phase 4** — MC dropout, temperature scaling, selective prediction (`04_uncertainty_calibration.ipynb`)
- [x] **Phase 5** — Grad-CAM, attention sanity, bootstrap CIs, report (`05_explainability_and_report.ipynb`)
- [ ] **Phase 6** — self-supervised pretraining *(fundus and volume extensions blocked: not in this mirror)*

The pipeline is complete and exercised end to end. **Headline numbers still need a GPU run**:
local CPU results use a reduced budget (128px, few epochs, one seed) and are a plumbing check,
not a result. Run `scripts/run_comparison.py --budget colab_gpu --seeds 42 43 44 --evaluate`
before reporting anything.

---

## Data use and citation

OLIVES is released under CC BY 4.0 and derives from the PRIME and TREX-DME clinical trials
(Retina Consultants of Texas), collected under IRB approval and de-identified per HIPAA. Raw
images, manifests and checkpoints are git-ignored and must not be committed.

The cohort comes from a single U.S. clinic with no untreated control group, so nothing here
supports a diagnostic or deployment claim. Biomarkers are retrospectively graded indicators, and
the dataset's authors are explicit that they are **not causal** to disease — this project makes no
causal claims.

```bibtex
@inproceedings{prabhushankarolives2022,
  title     = {OLIVES Dataset: Ophthalmic Labels for Investigating Visual Eye Semantics},
  author    = {Prabhushankar, Mohit and Kokilepersaud, Kiran and Logan, Yash-yee and
               Trejo Corona, Stephanie and AlRegib, Ghassan and Wykoff, Charles},
  booktitle = {NeurIPS Track on Datasets and Benchmarks},
  year      = {2022}
}
```

- Paper: <https://arxiv.org/abs/2209.11195>
- Dataset: <https://doi.org/10.5281/zenodo.7105232> · [Hugging Face mirror](https://huggingface.co/datasets/gOLIVES/OLIVES_Dataset)
- Official code: <https://github.com/olivesgatech/OLIVES_Dataset>
