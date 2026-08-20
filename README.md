# LEAP — Leukemia End-to-End Analysis Platform

LEAP recognises acute promyelocytic leukemia (APL) from bone marrow aspirate smear whole-slide
images, and extends the same architecture to other acute myeloid leukemia labels and to overall
survival. It is a weakly supervised model: the only supervision is one label per slide, and the
model learns for itself which cells on the smear carry the signal. This repository contains the
model and the code to train it, to aggregate the trained Experts into the LEAP score, and to fit
the survival models on top of them.

## Architecture

A slide is represented as a bag of single-cell patches. Each patch passes through an
ImageNet-pretrained convolutional feature extractor whose early stages stay frozen and whose later
stages train with the rest of the network. A patch-wise projection maps those features into the
attention space, gated attention pooling collapses the bag into one slide vector by a learned
weighted average, and a small MLP turns that vector into a slide-level logit. Extractor and MIL
head train together end to end, so the features adapt to the task instead of being fixed.

Three Experts are trained independently on the same folds, one per backbone — VGG19, ResNet50 and
DenseNet121 — and their predictions are combined into the LEAP score. Aggregation is fitted on
out-of-fold predictions from the discovery cohort only and is cross-fitted, so no slide is scored
by a combiner that was fitted using it, and external cohorts are scored without refitting.

Because pooling is a weighted average over cells, the attention weight on a cell is directly the
contribution that cell made to the slide-level call, which is what makes the model interpretable
at the cell level.

**Patch extraction is upstream and is not part of this repository.** LEAP begins where segmented
single-cell patches already exist on disk. Cell detection, segmentation and cropping are done
separately, and this repository neither performs nor stubs them.

## Installation

```
conda env create -f environment.yml
conda activate leap
```

Run everything from the repository root, so that `import leap` resolves.

## Input data contract

Three inputs. Nothing else about your filesystem is assumed, and every path is a config key.

**Patch archives.** One zip per slide, named for its slide identifier:

```
<image_folder>/
├── SLIDE_0001.zip
├── SLIDE_0002.zip
└── ...
```

Each archive holds one PNG per cell, at the top level or in subdirectories; only the `.png`
suffix matters and filenames are otherwise free. Names are sorted before sampling, so ordering is
stable. Patches are resized to 96×96 and normalised with ImageNet statistics, so the crops
themselves may be any size. Each bag draws `patches_per_slide` patches; a slide with fewer is
zero-padded to that length, and a slide with more is subsampled — deterministically at evaluation
time, from a seed derived from `(base_seed, Slide_ID)`, and re-drawn each epoch during training.

**Label file.** One spreadsheet (`.xlsx`), one row per slide:

| column | type | meaning |
|---|---|---|
| `Slide_ID` | string | must match the zip basename exactly, and be unique |
| *your label column* | integer, 0 or 1 | the classification target; name it in `data.label_column` |

For survival, the same format with two columns instead of a label:

| column | type | meaning |
|---|---|---|
| `Slide_ID` | string | as above |
| *time column* | float | follow-up time, in whatever unit you report |
| *event column* | 0 or 1 | 1 if the event was observed, 0 if censored |

Every **other** column in a survival label file is treated as a clinical covariate for the
second-stage Cox model, and rows with any missing covariate are dropped from that model. So point
`cohorts.*.label_file` at a file containing only the covariates you intend to adjust for. All
cohorts must share the same covariate columns.

**Split file.** JSON, keyed by slide identifier so that an edit to the spreadsheet cannot
reshuffle folds. `scripts/train_expert.py` creates it on first run if `data.split_json` does not
exist, using stratified k-fold with a stratified inner validation split; write your own to control
the partition:

```json
{
  "label_file": "labels.xlsx",
  "label_column": "TARGET",
  "n_folds": 5,
  "seed": 42,
  "val_frac": 0.25,
  "folds": {
    "0": {"train": ["SLIDE_0002", "..."], "val": ["SLIDE_0031", "..."], "test": ["SLIDE_0007", "..."]},
    "1": {"train": ["..."], "val": ["..."], "test": ["..."]}
  }
}
```

Every identifier listed must be present in the label file; a mismatch aborts the run rather than
silently dropping slides. Only the `folds` block is read.

## Usage

Each task is one script and one config. The configs in `configs/` are blank templates: fill in
the keys you need — every key carries an inline comment saying what belongs there — and a key you
forget is reported by name rather than failing deep inside a library. Any key can be overridden on
the command line with `--set key=value`.

### 1. Train an Expert

```
python scripts/train_expert.py --config configs/train_expert.yaml
```

Trains one backbone across all folds. Per fold it selects the best epoch on validation, writes
that checkpoint to `output.checkpoint_dir` as `<experiment_id>_fold<k>.pt` — where
`<experiment_id>` is `<experiment_name>.<extractor class>.<head class>` — chooses the
operating threshold on validation, and applies it unchanged to the held-out fold and to every
cohort listed in `data.external_cohorts`. Under `output.results_dir` it writes, per cohort,
`<experiment_id>_fold_<k>.csv` with one row per slide (train and validation rows carry the label
only; test rows carry the probability, the thresholded call and the exact patch filenames scored),
a per-fold metrics JSON, the resolved config, and two append-only trackers,
`metrics_per_fold.csv` and `experiment_metrics.csv`.

Run it once per backbone, changing `extractor._target_`, `head.in_features`,
`head.d_model_attention` and `experiment_name`. `in_features` must equal the extractor output
dimension: 4096 for VGG19, 2048 for ResNet50, 1024 for DenseNet121.

### 2. Aggregate the Experts into the LEAP score

```
python scripts/ensemble.py --config configs/ensemble.yaml
```

CPU only — it reads the Experts' prediction CSVs, not their weights. Evaluates each single Expert
and three combiners (mean, logistic, non-negative logistic) on the discovery cohort and on each
zero-shot cohort, and reports the one named by `headline`. The run directory is tagged with the
`expert_gate_margin`, so runs that differ only in the gate never collide. Writes, under
`output.results_dir/<experiment_name>/`, `combiner_comparison.csv` (every model × cohort, with
bootstrap confidence intervals), `ensemble_recap.csv`, the reported model's per-fold predictions
in `<cohort>/ensemble_fold_<k>.csv`, and every other model's under `by_model/`.

To sweep several targets, loop in the shell:

```
for target in TARGET_A TARGET_B; do
  python scripts/ensemble.py --config configs/ensemble.yaml --set data.label_column=$target
done
```

### 3. Survival modelling

```
python scripts/survival.py --config configs/survival.yaml
```

Loads each Expert's checkpoint, embeds every slide of every cohort, standardises on the discovery
cohort, fits one Cox-loss MLP per Expert, combines their risks in a Cox model to give the
pathology risk score, then adds the clinical covariates in a second Cox model. Writes tables
only, under `output.results_dir/<experiment_name>/tables/`: `patients.csv` (one row per cohort ×
fold × slide, with every risk score and covariate), `cindex.csv`, `cox_stage2.csv`,
`univariate.csv`, `leapsurv_emb.parquet` and `results.json`.

The discovery cohort's test folds are disjoint, so each patient is counted once and pooled.
External cohorts are scored by every fold model on the *same* patients, which gives one estimate
of the model per fold but not one patient per fold: those scores are rank-normalised within each
fold and averaged into a single score per patient before any log-rank test, so an event is never
counted more than once.

If a Cox model fails to converge, the Experts are ordering slides almost identically; set
`cox_penalizer` to a small value such as 0.1.

## Hardware and runtime

A CUDA GPU is required in practice for steps 1 and 3; step 2 is CPU only. Every script falls back
to CPU if CUDA is unavailable, which is useful for a smoke test on a handful of slides and
impractical for a real run.

Memory is driven by bag size, not by slide count. One bag of 500 patches is
500 × 3 × 96 × 96 floats, about 55 MB, and the backbone sees all of them in one forward pass — so
`data.batch_size` is bounded by GPU memory and is small. Use `data.accumulated_batch_size` to
recover the effective batch size you want: with `batch_size: 4` and
`accumulated_batch_size: 32`, gradients accumulate over eight forward passes before each step.
Host memory is dominated by `data.num_workers` decompressing archives in parallel.

For bit-reproducible runs, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python. Seeding
enables PyTorch's deterministic algorithms, and deterministic cuBLAS matrix multiplication will
otherwise raise an error. Keep `data.num_workers` fixed across runs you intend to reproduce: the
per-epoch patch draw is seeded per worker, so changing the worker count changes which patches each
slide sees.

## Data and model availability

No data and no model weights are distributed with this repository. Trained checkpoints, slide and
cell embeddings, cross-validation splits, and patient data are **not** included, and no slide or
patient identifier appears anywhere in it. Those materials are available to reviewers via the
Zenodo record [10.5281/zenodo.17728421](https://doi.org/10.5281/zenodo.17728421), or from the
corresponding author on reasonable request.

## Citation

See `CITATION.cff`.

## License

MIT, `LICENSE`. The attention-MIL head in `leap/nn/` is adapted from Owkin's HistoSSLscaling
(<https://github.com/owkin/HistoSSLscaling>) and is governed by that project's terms; see the
module docstrings in `leap/nn/`.

## Intended use

LEAP is a research tool. It is not a medical device, has not been approved or cleared by any
regulatory body, and must not be used to inform clinical decisions or patient care.
