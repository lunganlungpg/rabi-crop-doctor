"""
Train Rabi Crop Doctor's real-image benchmark from the Kaggle 20k+ dataset.

The dataset is downloaded locally (never committed to Git), its official train/test
directories are kept separate, and the project's existing 28-dimensional feature
extractor is used so the results remain comparable with the current offline stack.

This creates a separate research model bundle under models/kaggle_20k/; it does
not overwrite the production 12-class Android model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from .feature_extractor import FEATURE_NAMES, extract_features_and_diagnostics
except ImportError:
    from feature_extractor import FEATURE_NAMES, extract_features_and_diagnostics


KAGGLE_DATASET = "jawadali1045/20k-multi-class-crop-disease-images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "training_set": "train",
    "test": "test",
    "testing": "test",
    "testing_set": "test",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
}
GENERIC_DIRS = {"image", "images", "img", "imgs", "data", "dataset", "raw", "raw_images"}


def _slug(value: str) -> str:
    value = value.lower().replace("___", " ").replace("__", " ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def _record_from_path(path: Path, root: Path) -> dict[str, str] | None:
    parts = path.relative_to(root).parts
    dirs = list(parts[:-1])
    if not dirs:
        return None

    split_index = next(
        (i for i, part in enumerate(dirs) if part.lower() in SPLIT_ALIASES),
        None,
    )
    if split_index is None:
        split = "unsplit"
        class_parts = dirs
    else:
        split = SPLIT_ALIASES[dirs[split_index].lower()]
        class_parts = dirs[split_index + 1 :]

    class_parts = [
        p for p in class_parts
        if p.lower() not in GENERIC_DIRS and not p.startswith(".")
    ]
    label = _slug("_".join(class_parts))
    if not label:
        return None
    return {"path": str(path), "label": label, "split": split}


def _collect_records(dataset_dir: Path, crops: list[str] | None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        record = _record_from_path(path, dataset_dir)
        if record is None:
            continue
        if crops:
            crop = record["label"].split("_", 1)[0]
            if crop not in crops:
                continue
        records.append(record)

    if not records:
        raise RuntimeError(
            f"No labelled images found under {dataset_dir}. Expected folders such as "
            "train/<class_name>/*.jpg and test/<class_name>/*.jpg."
        )
    return records


def _deduplicate_records(
    records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Remove byte-identical duplicates and prevent exact train/test leakage."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        digest = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        groups[digest].append(record)

    kept: list[dict[str, str]] = []
    duplicate_count = 0
    conflicting_hashes = 0
    for group in groups.values():
        labels = {item["label"] for item in group}
        if len(labels) > 1:
            conflicting_hashes += 1
            duplicate_count += len(group)
            continue
        # Prefer the official test copy so an identical training copy is dropped.
        chosen = next((x for x in group if x["split"] == "test"), group[0])
        kept.append(chosen)
        duplicate_count += len(group) - 1

    return kept, {
        "duplicate_images_removed": duplicate_count,
        "conflicting_label_hash_groups_removed": conflicting_hashes,
    }


def _download_dataset(destination: Path) -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI is not installed. Run: pip install kaggle"
        )
    has_legacy_credentials = bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    config_path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not has_legacy_credentials and not config_path.exists():
        raise RuntimeError(
            "Kaggle credentials are missing. Configure KAGGLE_USERNAME and KAGGLE_KEY "
            "environment variables or place kaggle.json in ~/.kaggle/."
        )
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(destination),
        "--unzip",
    ]
    print("Downloading full Kaggle dataset:", KAGGLE_DATASET, flush=True)
    subprocess.run(command, check=True)


def _make_split(
    records: list[dict[str, str]],
    test_size: float,
    random_state: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str, list[str]]:
    official_train = [r for r in records if r["split"] in {"train", "validation"}]
    official_test = [r for r in records if r["split"] == "test"]

    if official_train and official_test:
        train_labels = {r["label"] for r in official_train}
        test_labels = {r["label"] for r in official_test}
        common = sorted(train_labels & test_labels)
        excluded = sorted((train_labels | test_labels) - set(common))
        train = [r for r in official_train if r["label"] in common]
        test = [r for r in official_test if r["label"] in common]
        source = "official train/validation and test folders; test held out unchanged"
    else:
        all_records = [r for r in records if r["split"] == "unsplit"]
        if not all_records:
            raise RuntimeError(
                "Could not construct a train/test split. Check the dataset folder structure."
            )
        counts = Counter(r["label"] for r in all_records)
        eligible = {label for label, n in counts.items() if n >= 2}
        excluded = sorted(set(counts) - eligible)
        all_records = [r for r in all_records if r["label"] in eligible]
        train, test = train_test_split(
            all_records,
            test_size=test_size,
            random_state=random_state,
            stratify=[r["label"] for r in all_records],
        )
        source = f"stratified {1-test_size:.0%}/{test_size:.0%} split (no official test folder)"

    train_counts = Counter(r["label"] for r in train)
    test_counts = Counter(r["label"] for r in test)
    valid = sorted(
        label for label in set(train_counts) & set(test_counts)
        if train_counts[label] >= 2 and test_counts[label] >= 1
    )
    dropped = sorted((set(train_counts) | set(test_counts)) - set(valid))
    excluded = sorted(set(excluded) | set(dropped))
    train = [r for r in train if r["label"] in valid]
    test = [r for r in test if r["label"] in valid]

    if len(valid) < 2:
        raise RuntimeError(
            "Fewer than two classes have usable examples in both train and test. "
            "Inspect the folder names and dataset structure before training."
        )
    return train, test, source, excluded


def _extract_matrix(
    records: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    feature_rows: list[list[float]] = []
    labels: list[str] = []
    skipped = 0
    for i, record in enumerate(records, start=1):
        try:
            # Verify that Pillow can decode the image before feature extraction.
            with Image.open(record["path"]) as im:
                im.verify()
            result = extract_features_and_diagnostics(record["path"])
            feature_rows.append([float(x) for x in result["features"]])
            labels.append(record["label"])
        except Exception as exc:
            skipped += 1
            print(f"[skip] {record['path']}: {exc}", flush=True)
        if i % 500 == 0:
            print(f"Extracted features from {i:,}/{len(records):,} images", flush=True)

    if not feature_rows:
        raise RuntimeError("All selected images failed decoding or feature extraction.")
    return np.asarray(feature_rows, dtype=np.float64), np.asarray(labels), FEATURE_NAMES, skipped


def _score(y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    prediction = np.argmax(probabilities, axis=1)
    report = classification_report(
        y_true, prediction, labels=np.arange(len(class_names)),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, prediction, labels=np.arange(len(class_names))
        ).tolist(),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
    }
    if len(class_names) > 2:
        try:
            result["macro_roc_auc_ovr"] = float(
                roc_auc_score(
                    y_true, probabilities,
                    labels=np.arange(len(class_names)),
                    multi_class="ovr", average="macro",
                )
            )
        except ValueError as exc:
            result["macro_roc_auc_ovr"] = None
            result["roc_auc_note"] = str(exc)
    return result


def train(
    dataset_dir: Path,
    output_dir: Path,
    crops: list[str] | None,
    test_size: float,
    cv_folds: int,
    random_state: int,
    skip_hash_dedup: bool,
) -> dict[str, Any]:
    records = _collect_records(dataset_dir, crops)
    raw_counts = Counter(r["label"] for r in records)
    print(f"Discovered {len(records):,} images across {len(raw_counts)} class folders.")

    dedup_report = {"duplicate_images_removed": 0, "conflicting_label_hash_groups_removed": 0}
    if not skip_hash_dedup:
        records, dedup_report = _deduplicate_records(records)
        print(
            f"Exact-duplicate audit: removed {dedup_report['duplicate_images_removed']:,}; "
            f"conflicting-label groups: {dedup_report['conflicting_label_hash_groups_removed']:,}"
        )

    train_records, test_records, split_description, excluded = _make_split(
        records, test_size, random_state
    )
    train_counts = Counter(r["label"] for r in train_records)
    test_counts = Counter(r["label"] for r in test_records)
    class_names = sorted(train_counts)
    print("Split:", split_description)
    print(f"Training images: {len(train_records):,}; held-out test images: {len(test_records):,}")
    print("Classes:", len(class_names))
    for name in class_names:
        print(f"  {name:<36} train={train_counts[name]:>5,}  test={test_counts[name]:>5,}")

    print("\nExtracting 28-dimensional image features...")
    X_train, y_train_names, feature_names, skipped_train = _extract_matrix(train_records)
    X_test, y_test_names, _, skipped_test = _extract_matrix(test_records)

    # Decoding failures can alter the class support; retain only classes present in both matrices.
    common = sorted(set(y_train_names) & set(y_test_names))
    train_mask = np.isin(y_train_names, common)
    test_mask = np.isin(y_test_names, common)
    X_train, y_train_names = X_train[train_mask], y_train_names[train_mask]
    X_test, y_test_names = X_test[test_mask], y_test_names[test_mask]
    class_names = common
    if len(class_names) < 2:
        raise RuntimeError("Fewer than two classes remain after image decoding.")

    encoder = LabelEncoder().fit(class_names)
    y_train = encoder.transform(y_train_names)
    y_test = encoder.transform(y_test_names)
    min_train_class = min(Counter(y_train).values())
    if min_train_class < 2:
        raise RuntimeError("Every training class needs at least two decodable images.")

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    estimators = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=4.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            cache_size=2048,
            random_state=random_state,
        ),
        "ann_mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.12,
            random_state=random_state,
        ),
    }

    min_cv = min(args_count for args_count in Counter(y_train).values())
    folds = min(cv_folds, min_cv)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state) if folds >= 2 else None
    metrics: dict[str, Any] = {
        "dataset": {
            "name": "20k+ Multi-Class Crop Disease Images",
            "kaggle_id": KAGGLE_DATASET,
            "dataset_root": str(dataset_dir.resolve()),
            "selected_crops": crops or "all",
            "split_protocol": split_description,
            "raw_image_count": len(records),
            "training_images_after_decode": int(len(X_train)),
            "held_out_test_images_after_decode": int(len(X_test)),
            "num_classes": len(class_names),
            "classes": class_names,
            "training_class_counts": {
                name: int(Counter(y_train_names)[name]) for name in class_names
            },
            "test_class_counts": {
                name: int(Counter(y_test_names)[name]) for name in class_names
            },
            "excluded_or_unpaired_classes": excluded,
            "feature_names": feature_names,
            "num_features": len(feature_names),
            "exact_duplicate_audit": dedup_report,
            "images_skipped_train": skipped_train,
            "images_skipped_test": skipped_test,
            "cross_validation_folds": folds if cv else 0,
        },
        "models": {},
    }

    probabilities: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, estimator in estimators.items():
        print(f"\nTraining {name}...", flush=True)
        cv_summary: dict[str, Any] = {}
        if cv:
            pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
            cv_result = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
                n_jobs=1,
                error_score="raise",
            )
            cv_summary = {
                "accuracy_mean": float(np.mean(cv_result["test_accuracy"])),
                "accuracy_sd": float(np.std(cv_result["test_accuracy"], ddof=1)),
                "macro_f1_mean": float(np.mean(cv_result["test_macro_f1"])),
                "macro_f1_sd": float(np.std(cv_result["test_macro_f1"], ddof=1)),
            }

        started = time.perf_counter()
        estimator.fit(X_train_scaled, y_train)
        fit_seconds = time.perf_counter() - started
        probs = estimator.predict_proba(X_test_scaled)
        probabilities[name] = probs
        fitted[name] = estimator
        predictions[name] = np.argmax(probs, axis=1)
        model_metrics = _score(y_test, probs, class_names)
        model_metrics.update({
            "cross_validation": cv_summary,
            "fit_seconds": float(fit_seconds),
        })
        if hasattr(estimator, "loss_curve_"):
            model_metrics["loss_curve"] = [float(x) for x in estimator.loss_curve_]
        metrics["models"][name] = model_metrics
        print(
            f"{name}: held-out accuracy={model_metrics['accuracy']:.4f}; "
            f"macro-F1={model_metrics['macro_f1']:.4f}; fit={fit_seconds:.1f}s",
            flush=True,
        )

    # Equal-weight probability average is reported as a benchmark, not assumed superior.
    ensemble_probs = np.mean(np.stack(list(probabilities.values()), axis=0), axis=0)
    ensemble_metrics = _score(y_test, ensemble_probs, class_names)
    metrics["models"]["soft_vote_ensemble"] = ensemble_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "trained_models.pkl"
    import pickle
    with model_path.open("wb") as handle:
        pickle.dump({
            "scaler": scaler,
            "models": fitted,
            "class_names": class_names,
            "feature_names": feature_names,
            "source_dataset": KAGGLE_DATASET,
        }, handle)

    metrics["environment"] = {
        "python": os.sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    metrics_path = output_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_path = output_dir / "held_out_predictions.csv"
    pred_frame = pd.DataFrame({
        "true_label": encoder.inverse_transform(y_test),
    })
    for name, pred in predictions.items():
        pred_frame[f"{name}_prediction"] = encoder.inverse_transform(pred)
    pred_frame.to_csv(pred_path, index=False)

    print("\nTraining complete.")
    print("Model bundle:", model_path)
    print("Metrics:", metrics_path)
    print("Held-out predictions:", pred_path)
    print(
        "\nIMPORTANT: This is a separate external-dataset research bundle. "
        "The production Android model was not overwritten."
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and train Rabi Crop Doctor on the Kaggle 20k+ crop disease image dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/kaggle_20k_crop_disease_images"),
        help="Directory containing the extracted Kaggle image folders.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract the full dataset using the Kaggle CLI.",
    )
    parser.add_argument(
        "--crops",
        nargs="+",
        default=None,
        help="Optional crop prefixes to include, e.g. --crops wheat rice. Default: all classes.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/kaggle_20k"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-hash-dedup",
        action="store_true",
        help="Skip exact-file duplicate audit (faster for very large downloads).",
    )
    args = parser.parse_args()

    if not 0.05 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.05 and 0.50")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        _download_dataset(args.dataset_dir)
    elif not any(args.dataset_dir.rglob("*")):
        parser.error(
            f"No data found in {args.dataset_dir}. Supply --download with Kaggle credentials "
            "or point --dataset-dir to an extracted dataset."
        )

    crop_filter = [_slug(x) for x in args.crops] if args.crops else None
    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        crops=crop_filter,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        skip_hash_dedup=args.skip_hash_dedup,
    )


if __name__ == "__main__":
    main()
