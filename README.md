# Clinically Grounded, Uncertainty-Aware Multimodal Retinal Biomarker Detection

Multilabel detection of retinal biomarkers from OCT B-scans on the **OLIVES** dataset, fusing the
scan with the clinical measurements recorded at the same visit (BCVA, CST), with calibrated
probabilities, uncertainty estimates and per-label explanations.

**Owner:** Blessing Asare · **Framework:** PyTorch · **Status:** Phases 0–5 executed on an NVIDIA A100; the proposed gated fusion did not outperform the OCT-only baseline

---

## Research question

> Can clinically grounded fusion of OCT features with visit-level BCVA and CST improve multilabel
> retinal biomarker detection — and where it appears to, can that gain be shown to reflect clinical
> signal rather than visit identity?

A rigorous negative result — clinical fusion failing to beat a well-controlled OCT baseline — is a
legitimate and useful outcome. The evaluation is built so that outcome would be believable.

The second clause is not decoration. In this cohort the pair `(BCVA, CST)` identifies the visit
uniquely in **97–100% of cases in every fold**, so a fusion model can improve by memorising which
visit it is looking at. That would not transfer to another clinic and is not a clinical finding.
Distinguishing the two is what the control ladder below exists for.

---

## A100 results

Models were evaluated across seeds 42, 43 and 44 on one fixed patient-grouped holdout:
52 training, 13 validation, 9 calibration and 13 test patients (1,372 test scans). Thresholds
were fitted on validation only. Macro metrics omit the three labels with no positive test
examples (DRIL, VMT and serous PED).

| Model | Macro F1 | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|
| Clinical only | 0.3644 | 0.6500 | 0.3541 |
| OCT only | **0.5026** | 0.8522 | 0.5478 |
| Concatenation | 0.4971 | **0.8660** | **0.5492** |
| Gated fusion | 0.4979 | 0.8551 | 0.5413 |

OCT clearly outperformed BCVA+CST alone. Concatenation was effectively tied with OCT, while
gated fusion was lower by 0.0065 macro AUPRC and 0.0047 macro F1. The proposed gate therefore
did not improve the primary endpoint on this cohort. This is a negative model result, not a
pipeline failure.

![Three-seed model comparison](assets/readme/02_model_comparison.png)

### Gate mechanism

The gate responded to the intended clinical signal: its Spearman correlation was 0.8768 with
CST and -0.3875 with BCVA, and the largest biomarker effects occurred for DRT/ME, SRF and IRF.
However, its mean scale was 1.8971 out of a maximum of 2; 97.78% of channels were amplified,
20.31% were essentially constant, and across-sample variation was small. It learned a
clinically associated but mostly amplifying transformation without improving prediction.

![Learned gate distribution](assets/readme/03_gate_distribution.png)

![Implemented gated-fusion architecture](assets/readme/07_gated_fusion_architecture.png)

### Single-seed follow-up experiments

The additional Phase 3 runs used the same patient-grouped holdout. The gate ablations tested
whether missingness indicators or the residual-safe formulation explained the negative result.
The longitudinal arms used a bounded per-biomarker logit correction driven by within-eye clinical
change; these are seed-42 exploratory runs, not three-seed estimates.

| Experiment | Seed | Macro F1 | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|---:|
| Gated fusion | 42 | 0.4988 | 0.8528 | 0.5335 |
| Gated, no missingness indicators | 42 | 0.4574 | 0.8319 | 0.5338 |
| Gated, raw multiplicative scale | 42 | 0.4779 | 0.8604 | 0.5146 |
| Longitudinal absolute + within-eye change | 42 | 0.4856 | — | **0.5721** |
| Within-eye change only | 42 | 0.4643 | — | **0.5616** |

Against the seed-42 OCT reference (macro F1 0.5189, macro AUPRC 0.5476), the longitudinal arm
gained 0.0245 AUPRC while losing 0.0333 F1; the change-only arm gained 0.0140 AUPRC while losing
0.0546 F1. This is an encouraging ranking result, but it does **not** establish that clinical
change improves the model: only one seed was run, the contender checkpoints were not available
to Phase 5 for a paired patient bootstrap, and the threshold-dependent metric moved in the
opposite direction. Persistent paired reruns are required before claiming the objective was met.

### Calibration and selective prediction

Post-hoc analysis used the persistent OCT-only seed-42 run. Temperature scaling reduced mean
ECE from 0.0720 to 0.0609 and Brier score from 0.0902 to 0.0855 across the 12 labels with enough
positives to interpret. After refitting validation thresholds on the calibrated scale, macro F1
was 0.5261; ranking metrics remained unchanged, as expected.

MC dropout uncertainty tracked scan error (Spearman rho 0.5319). Referring the most uncertain
20% of scans increased macro F1 from 0.5261 to 0.5983 and macro AUPRC from 0.5476 to 0.6183.
The highest observed macro F1 was 0.5994 at 70% coverage. This is a retrospective
selective-prediction result, not a deployment claim.

| Calibration | Selective prediction |
|---|---|
| ![Calibration before and after temperature scaling](assets/readme/04_calibration_comparison.png) | ![Performance against retained coverage](assets/readme/05_selective_prediction.png) |

### Confidence intervals and explainability

Patient-level bootstrap intervals for the persistent OCT run were: macro F1 0.5004
[0.4590, 0.5795], macro AUROC 0.8505 [0.7948, 0.8905], and macro AUPRC 0.5476
[0.5120, 0.6584]. At evaluation time, only this OCT checkpoint was available in persistent Drive
storage, so these intervals do not establish a difference between models.

The attention audit examined 80 IRF and DRT/ME heatmaps from 40 test scans. Twenty maps (25%)
concentrated suspiciously on borders or background. Grad-CAM therefore provides case-review
material, not proof that the model reasons clinically.

![Attention sanity audit](assets/readme/06_attention_sanity.png)

## Improvement plan (next round)

The A100 result points at the input pipeline and the fine-tuning schedule, not at
model capacity: OCT-only already matches both fusion models, and the clinical gate was
near-saturated (mean scale 1.897/2.0, 97.78% of channels amplified), so it was acting as
global gain rather than selective modulation.

### What changed in the code

| # | Change | Evidence it addresses | Where |
|---|---|---|---|
| 1 | Deterministic retinal crop before resize | ~45% of pixels near-black; 25% of audited heatmaps on border/background | `RetinalTissueCrop`, `data.crop_retina` |
| 2 | 320px input, aspect-ratio padding, train-fold normalisation | Small biomarkers vanish at 224px from 504x496; measured mean is 0.17 not 0.482 | `configs/oct_improved.yaml` |
| 3 | RETFound ViT-L retinal MAE weights | 52 training patients; retinal pretraining beats ImageNet on label efficiency | `ImageEncoder`, `configs/oct_retfound.yaml` |
| 4 | 2.5D adjacent B-scans as the three channels | Grayscale was copied 3x, wasting the stem; neighbours add volumetric context | `data.image_mode: adjacent`, `configs/oct_adjacent.yaml` |
| 5 | Patient/visit-balanced sampling and class weights | Scan-level shuffling lets high-volume patients dominate the gradient | `GroupBalancedSampler`, `training.sampler` |
| 6 | Discriminative LRs, warmup+cosine, progressive unfreeze | Train loss fell while validation degraded; one flat LR for the whole net | `training/schedule.py` |
| 7 | Asymmetric loss as an ablation | 5 labels below 1% prevalence; weighted BCE still swamped by easy negatives | `AsymmetricLoss`, `configs/loss_asl.yaml` |
| 8 | Bounded fusion replacing the gate | Gate reached ~2x global amplification and lost to OCT-only | `ResidualLogitFusionModel`, `BoundedFiLMFusionModel` |
| 9 | Seed ensembling | 13 test patients makes any single seed unstable | `SeedEnsemble`, `--ensemble` |

### The two replacement fusion designs

Both start as the OCT baseline **exactly**, so clinical input has to earn its influence
against validation rather than being switched on at initialisation:

```
residual_logit_fusion:  logits = oct_logits + beta * clinical_logits
                        beta init 0, one bounded coefficient per biomarker

bounded_film_fusion:    scale = 1 + 0.25 * tanh(gamma)
                        modulated = scale * oct_features + shift
```

`ResidualLogitFusionModel.beta_values()` is the interpretable output: CST should be
allowed to inform DRT/ME, IRF and SRF and stay near zero for the vitreous-face labels.
Report those coefficients whether or not the macro metric moves — they answer the
research question more directly than a 0.005 difference in AUPRC.

### Recommended experiment order

```bash
# 1. Current OCT baseline (already have it)
# 2. Retinal crop + 320px + staged fine-tuning
python scripts/run_comparison.py --arms B E --seeds 42 43 44 --evaluate --ensemble

# 3-5. 2.5D input, then RETFound (set model.pretrained_checkpoint first)
python scripts/train.py --config configs/oct_adjacent.yaml --seed 42
python scripts/train.py --config configs/oct_retfound.yaml --seed 42

# 6. Loss ablation against the BCE reference
python scripts/train.py --config configs/loss_asl.yaml --seed 42

# 7. Bounded clinical fusion over the best OCT configuration
python scripts/run_comparison.py --arms E G H --seeds 42 43 44 --evaluate --ensemble

# 8. Within-eye clinical context, the arm with a measured mechanism.
#    Pre-register the target labels FIRST; it reads training patients only.
python scripts/preregister_targets.py --folds
python scripts/run_comparison.py --arms E I J --seeds 42 43 44 --evaluate     --preregistered ir_hemorrhages

# 9. The control ladder. Only meaningful if step 8 showed a gain.
python scripts/run_comparison.py --controls --seeds 42 --evaluate

# 10. Five-fold grouped confirmation of the winner
python scripts/make_splits.py --config configs/data.yaml --folds
```

Do not start with a larger backbone or a longer schedule. The evidence points at wasted
image area, ImageNet-only pretraining, missing inter-slice context, and patient-level
data scarcity.

The holdout versions of arms I and J have now been completed for seed 42; their exploratory
results are reported above. The remaining work is the three-seed rerun, paired patient bootstrap,
control ladder and five-fold confirmation.

### Within-eye clinical context, and the controls for it

Absolute BCVA and CST at the current visit are largely redundant with the B-scan — CST is a
thickness measurement of the very volume being classified, and across patients its variance is
dominated by differences in baseline retinal thickness rather than disease state. That is the
mechanical reason the A100 gate collapsed into near-global amplification.

What a single B-scan does not contain is how the eye has **moved**. Centring both the clinical value
and the biomarker state within an eye removes patient identity by construction, so the visit
fingerprint cannot contribute. Measured on training patients only, several biomarkers retain a
clinically coherent association — fluid appears as thickness rises, atrophy as it falls.

`configs/fusion_longitudinal.yaml` (arm I) adds the signed change from each eye's own baseline visit.
`configs/fusion_delta_only.yaml` (arm J) withholds the absolute values entirely: a delta is zero at
every baseline visit and a small signed number elsewhere, so that arm has no fingerprint left to
exploit. If J beats the OCT reference, the gain cannot be memorisation.

| arm | control | what survives | reading if the gain holds up |
|---|---|---|---|
| K | `patient_mean` | patient severity, no visit contrast | the gain was never longitudinal |
| L | `within_patient_shuffle` | identity and marginals | the gain was a fingerprint |
| M | `across_patient_shuffle` | marginals only | floor; anything here is a bug |
| N | `quantise` | clinical meaning, not exact values | the gain is a clinical effect |

`quantise` bins CST to 25 µm and BCVA to 5 letters, which drops exact-value visit uniqueness from
98.4% to 55.6% while keeping everything a clinician would act on. It is both the diagnostic and the
remedy: if the gain survives binning, the binned features are the defensible way to report it.

**A control that retains most of the gain has not been passed.** It means the thing it destroys was
not what the fusion arm was using.

### Pre-registered targets

`scripts/preregister_targets.py` measures the within-eye association on each fold's **training
patients only** and names the labels where fusion is predicted to help, before any test data is
touched. A label qualifies on two independent grounds: an association strong enough to be a
mechanism, and an OCT baseline low enough to leave room.

For the executed holdout run, the training-only procedure pre-registered `ir_hemorrhages` and
`ez_disruption`. Across the planned five-fold analysis, only `ir_hemorrhages` qualifies in every
fold:

| fold | qualifying labels |
|---|---|
| 0, 2, 4 | `ir_hemorrhages`, `ez_disruption` |
| 1, 3 | `ir_hemorrhages` |
| **every fold** | **`ir_hemorrhages`** |

`ir_hemorrhages` is therefore the confirmatory set; everything else is exploratory and must be
reported as such. The procedure earns its keep immediately: `atrophy_thinning` looks strongly
associated (r = −0.49, p = 0.006) when all 87 patients are pooled, and qualifies in **no fold** once
held-out patients are excluded. That is precisely the artefact the guard exists to catch.

### Comparing two arms

Use `ResultsAggregator.paired_difference`, not `intervals_overlap`. Two independently computed
intervals overlap mostly because patient difficulty is wide, and that component is shared by both
arms and cancels when the difference itself is resampled. `intervals_overlap` is kept for the
descriptive table only; as a test it is both wrong and the least powerful option available.

### Self-supervised pretraining on OLIVES

If you pretrain on the ~70k unlabelled scans rather than using RETFound, use
`pipeline.pretraining_frame(assignment)` **per fold**. "Unlabelled" is not "safe":
pretraining on a held-out patient's scans teaches the encoder that patient's anatomy and
the result stops being an inductive estimate. That method restricts the pool to the
fold's training patients and raises if a held-out patient slips in.

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
| Prevalence spans **67% (IRHRF) to 0.06% (serous PED)** | Per-label thresholds, not 0.5. DRIL, VMT and serous PED had no positive test examples; RPE disruption had one. |
| `Scan (n/49)` is present on only **22%** of rows | It belongs to the annotation block. Filter modelling rows on `has_biomarkers`. |
| **No visit/week column**; no fundus; no 3D volumes | Visits are *inferred* from BCVA/CST change-points. Fundus and volume extensions are blocked. |
| BCVA/CST missing for **patient 79 only** | Concentrated, so missingness is informative — impute on train, keep an indicator. |
| Measured pixel mean ≈ **0.17**, not the paper's 0.482 | Do not reuse the paper's normalisation constants. |

![Biomarker prevalence and patient support](assets/readme/01_label_prevalence.png)

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

| ID | Model | Clinical influence on the OCT path |
|---|---|---|
| A | `clinical_only` | n/a — how much signal is in BCVA + CST alone |
| B | `oct_only` | none — the imaging baseline every fusion claim is measured against |
| C | `concat_fusion` | additive, in feature space, via a shared head |
| D | `gated_fusion` | multiplicative gate over every channel, unbounded toward 2x |
| G | `residual_logit_fusion` | additive in *logit* space; one bounded coefficient per biomarker |
| H | `bounded_film_fusion` | multiplicative scale + shift, capped at ±25% |

D is the original proposal and the A100 run did not support it. G and H are the bounded
replacements; both begin as an exact copy of B.

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

### Persistent checkpoints and model bundles

The Colab training notebooks write runs directly to
`/content/drive/MyDrive/olives/outputs/runs/colab_gpu`. Every model/seed run keeps:

```text
<run_id>/
├── checkpoints/<run_id>_last.pt   # full state, replaced atomically after every epoch
├── checkpoints/<run_id>_best.pt   # best validation checkpoint
├── models/<run_id>_model.pt       # best weights + architecture/label metadata
├── resolved_config.yaml
├── clinical_preprocessor.json
└── run_state.json
```

`training.resume_from_checkpoint: true` restores model weights, AdamW state, AMP scaler,
early-stopping patience, metric history, schedule position and random-number states from `last`.
`training.reuse_completed_run: true` loads a finished run without training. Keep the same
model/split/seed run ID when reconnecting to Colab; Phase 2–4 do this automatically.

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
- [x] **Improvement round implemented** — crop/320px, 2.5D input, RETFound support,
      balanced sampling, staged fine-tuning, ASL, two bounded fusion designs, seed
      ensembling. The original-gate ablations and longitudinal bounded-fusion arms have
      seed-42 results; the crop/320px, 2.5D, RETFound, ASL and ensemble arms remain untrained.
- [x] **Within-eye clinical context implemented** — longitudinal features, the four-rung control
      ladder, pre-registration of target labels, and paired patient-level difference testing.
      The longitudinal and delta-only holdout arms are trained for seed 42; controls, repeated
      seeds, paired bootstrap and five-fold confirmation remain.
- [x] **Persistent Colab recovery** — atomic best/last checkpoints, full optimiser-state resume,
      completed-run reuse and exported best-model bundles under Google Drive.
- [ ] **Phase 6** — self-supervised pretraining *(fundus and volume extensions blocked: not in this mirror)*

The pipeline has been exercised end to end on an NVIDIA A100 at the full 224 px, 50-epoch
budget across three seeds. The current evidence supports OCT over the clinical-only baseline,
but not the original gated fusion over OCT. The single-seed longitudinal AUPRC gains are
exploratory. Final inferential comparison still requires Drive-backed reruns for every contender,
paired patient-level bootstrap differences, repeated seeds and five-fold patient-grouped
cross-validation.

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
