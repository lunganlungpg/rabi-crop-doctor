# Training on the Kaggle 20k+ Crop Disease Image Dataset

Dataset: https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images

This workflow creates a separate research model bundle in models/kaggle_20k/. It does not overwrite the existing 12-class production model or Android assets.

## 1. Obtain Kaggle access

1. Sign in to Kaggle and open the dataset page.
2. Accept the dataset's terms/rules if Kaggle requests it.
3. Create a Kaggle API token in Account settings.
4. Install the Kaggle CLI: python -m pip install kaggle.
5. Configure credentials using Kaggle's documented method. Keep credentials private; never commit kaggle.json, API tokens, or the downloaded dataset to Git.

## 2. Download the complete dataset

From the repository root:

    python -m pip install -r requirements.txt
    python -m pip install kaggle
    python python/train_kaggle_dataset.py --download

The download is extracted to data/kaggle_20k_crop_disease_images/. The raw images are ignored by Git and must not be pushed to this repository unless the dataset licence explicitly permits redistribution.

## 3. Train

The script scans nested class folders, uses official train/test folders when both are present, removes byte-identical duplicate files to reduce train/test leakage, extracts the project's deterministic 28 image features, and trains Random Forest, RBF-SVM, MLP-ANN, and a soft-voting probability ensemble.

    python python/train_kaggle_dataset.py --dataset-dir data/kaggle_20k_crop_disease_images --output-dir models/kaggle_20k

To train only on selected crop prefixes (the prefix is inferred from the class directory name):

    python python/train_kaggle_dataset.py --download --crops wheat rice

Do not assume the dataset contains the project's four existing crops or the same disease taxonomy. Inspect the discovered class counts and excluded/unpaired classes in the metrics before interpreting results.

## 4. Outputs

- models/kaggle_20k/trained_models.pkl — fitted sklearn models and scaler
- models/kaggle_20k/evaluation_metrics.json — dataset provenance, class counts, split protocol, cross-validation and held-out metrics
- models/kaggle_20k/held_out_predictions.csv — held-out true labels and model predictions

## Important methodological notes

- If the dataset supplies official train/test folders, the test folder is held out and not used for fitting or cross-validation.
- If no official split exists, a stratified random split is used; report this clearly.
- The duplicate audit detects byte-identical images only. It does not detect resized, recompressed, augmented, or near-duplicate images. For publication-grade evaluation, perform perceptual-hash and source/plant-level grouping where metadata permits.
- The current feature extractor is a hand-engineered 28-dimensional descriptor, not an end-to-end CNN. Performance reflects this representation and should not be described as deep-learning performance.
- A high random-split score does not establish field generalization. Prefer location-, season-, cultivar-, camera-, and plant-grouped external validation where source metadata supports it.
- This dataset's labels are not automatically mapped to the Rabi Crop Doctor advisory knowledge base. Do not issue crop-specific treatment advice for a class unless its identity and agronomic meaning have been verified.
- The separate model bundle is not wired into the Android/PWA classifier. Integrate it only after class mapping, independent validation, parity tests, bundle-size checks, and on-device latency tests.
