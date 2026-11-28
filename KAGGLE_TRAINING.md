# Training on the Kaggle 20k+ Crop Disease Image Dataset

Dataset: https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images

This workflow creates a separate research model bundle in `models/kaggle_20k/`. It does not overwrite the existing 12-class production model or Android assets.

## 1. Timeline-Driven Selective Rabi Crops

`python/rabi_timeline.py` resolves which Rabi crops are active from:
- **`ROADMAP.md`** (Nov 2026 – Nov 2027 quarterly milestones):
  - **Q1 (`Dec 2026 – Feb 2027`, incl. Nov 2026 kickoff, `v1.1`)**: `wheat`, `mustard`, `chickpea`, `potato`
  - **Q2 (`Mar – May 2027`, `v1.2`)**: `wheat`, `mustard`, `chickpea`, `potato`
  - **Q3 (`Jun – Aug 2027`, `v1.3`)**: adds `barley`, `lentil`, `field_pea`
  - **Q4 (`Sep – Nov 2027`, `v2.0`)**: all 7 Rabi crops (`wheat`, `mustard`, `chickpea`, `potato`, `barley`, `lentil`, `field_pea`)
- **`data/rabi_crops_knowledge.json` & `data/rabi_field_incidence_india.csv`**: monthly disease peaks (`peak-month` mode).

When selecting from `jawadali1045/20k-multi-class-crop-disease-images`, the trainer:
- Selects only matching Rabi crop classes (`Healthy Wheat` → `wheat_healthy`, `Flag Smut` → `wheat_flag_smut`, plus all `Wheat *` classes, merging nested `Train/Healthy Wheat/Wheat___Healthy`).
- Excludes non-Rabi crops (`Cotton`, `Maize`, `Rice`, `Sugarcane` including `Yellow Rust Sugarcane` and Cotton `Wilt` / `Leaf Curl`).
- Explicitly reports any timeline crops absent from this Kaggle source (`mustard`, `chickpea`, `potato`, etc.) in `models/kaggle_20k/evaluation_metrics.json`.
- Holds out the official `Validation/` folder unchanged as the evaluation split when `Train/` and `Validation/` are present.

## 2. Selective Download & Training Locally

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install kaggle six
python python/train_kaggle_dataset.py --download --timeline
```

When `--timeline` or `--crops` is supplied, `--download` uses selective HTTP Range extraction over the Kaggle archive so only the selected Rabi crop images (~1.15 GB for Wheat instead of the 2.51 GB full archive) are downloaded into `data/kaggle_20k_crop_disease_images/`.

Useful flags:
- Inspect the timeline resolution: `python python/rabi_timeline.py --mode roadmap`
- Monthly disease-peak filter: `python python/train_kaggle_dataset.py --download --timeline-mode peak-month --as-of 2027-01-15`
- Full dataset download fallback: `python python/train_kaggle_dataset.py --download --full-download`

If downloading privately or hitting rate limits, set `KAGGLE_API_TOKEN` (raw token or `kaggle.json` contents) — for the public CC0-1.0 dataset, anonymous selective range download also works automatically. Never commit `kaggle.json`, API tokens, or downloaded images to Git.

## 3. Fully Automated GitHub Actions Pipeline

`.github/workflows/kaggle-training.yml` automates the complete selective download, training, evaluation summary, and artifact publishing process via:
- **Monthly schedule (`cron: "30 2 1 * *"`)** aligned to `ROADMAP.md` milestones and Indian Rabi scouting months.
- **Automatic `push` triggers** on `main` when `ROADMAP.md`, `data/rabi_crops_knowledge.json`, `data/rabi_field_incidence_india.csv`, `python/rabi_timeline.py`, `python/train_kaggle_dataset.py`, or `.github/workflows/kaggle-training.yml` change.
- **Manual `workflow_dispatch`** with selectable `timeline_mode` (`roadmap`, `peak-month`, `all-rabi`, `custom`), `quarter`, `as_of`, `crops`, and `cv_folds`.

Outputs uploaded in the `rabi-crop-doctor-kaggle-training` workflow artifact:
- `models/kaggle_20k/trained_models.pkl` — fitted sklearn models and scaler
- `models/kaggle_20k/evaluation_metrics.json` — timeline provenance, crop coverage, class counts, split protocol, CV and held-out metrics
- `models/kaggle_20k/training_summary.md` — Markdown benchmark table (also published to the GitHub Actions run summary)
- `models/kaggle_20k/held_out_predictions.csv` — held-out true labels and model predictions

## Important methodological notes

- `Validation/` is held out unchanged for final evaluation and never used for fitting or cross-validation.
- Exact-duplicate hashing detects byte-identical images only, not near-duplicates or resized/recompressed copies.
- The 28-dimensional descriptor is hand-engineered computer vision features, not an end-to-end CNN.
- This separate research bundle does not overwrite `models/android_models.json` or `RabiCropDoctor.apk`.
