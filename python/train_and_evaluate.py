"""
train_and_evaluate.py
---------------------
Trains and benchmarks Machine Learning models (Random Forest, RBF SVM)
and Artificial Neural Network (ANN) architectures (MLP-64-32 and Deep-ANN-128-64-32)
plus a Hybrid Ensemble on the 28-dimensional Rabi leaf pathology features.

Supports training from data/crop_features_dataset.csv or directly extracting
features from a directory of labeled leaf image folders (--image-dir).
Exports compact JSON weights for zero-dependency offline Android APK inference.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

try:
    from .feature_extractor import FEATURE_NAMES, extract_features_and_diagnostics
    from .generate_rabi_dataset import CLASSES
except ImportError:
    from feature_extractor import FEATURE_NAMES, extract_features_and_diagnostics
    from generate_rabi_dataset import CLASSES


def _serialize_tree(tree) -> Dict[str, Any]:
    """Export a fitted sklearn DecisionTreeClassifier into flat arrays for JS traversal."""
    t = tree.tree_
    classes_probs = []
    for node_val in t.value:
        counts = node_val[0]
        total = float(counts.sum())
        probs = (counts / total).tolist() if total > 0 else [1.0 / len(counts)] * len(counts)
        classes_probs.append([round(float(p), 5) for p in probs])

    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": [round(float(th), 6) for th in t.threshold.tolist()],
        "probs": classes_probs,
    }


def _export_mlp(mlp: MLPClassifier) -> Dict[str, Any]:
    """Export trained MLP ANN layer weights & biases (rounded slightly for compact mobile bundle)."""
    return {
        "activation": "relu",
        "layers": [
            {
                "W": np.round(W, 6).tolist(),
                "b": np.round(b, 6).tolist(),
            }
            for W, b in zip(mlp.coefs_, mlp.intercepts_)
        ],
    }


def _export_svm(svc: SVC) -> Dict[str, Any]:
    """Export RBF SVM support vectors, dual coefficients, intercepts, and Platt calibration."""
    return {
        "gamma": round(float(svc._gamma), 8),
        "n_support": svc.n_support_.tolist(),
        "support_vectors": np.round(svc.support_vectors_, 5).tolist(),
        "dual_coef": np.round(svc.dual_coef_, 6).tolist(),
        "intercept": np.round(svc.intercept_, 6).tolist(),
        "probA": np.round(svc.probA_, 6).tolist(),
        "probB": np.round(svc.probB_, 6).tolist(),
    }


def _load_image_folder(image_dir: str) -> pd.DataFrame:
    """Extract features from any labeled directory tree image_dir/<class_name>/*.png|jpg."""
    rows = []
    idx = 0
    for cls_name in sorted(os.listdir(image_dir)):
        cls_path = os.path.join(image_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        for fname in sorted(os.listdir(cls_path)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            fpath = os.path.join(cls_path, fname)
            diag = extract_features_and_diagnostics(fpath)
            row = {
                "sample_id": idx,
                "filename": os.path.join(cls_name, fname),
                "crop": cls_name.split("_")[0],
                "label": cls_name,
                "extracted_severity_pct": diag["estimated_severity_pct"],
            }
            for k, val in zip(FEATURE_NAMES, diag["features"]):
                row[k] = float(val)
            rows.append(row)
            idx += 1
    if not rows:
        raise RuntimeError(f"No images found inside {image_dir}")
    return pd.DataFrame(rows)


def _crop_filtered_accuracy(y_true_idx: np.ndarray, probs: np.ndarray, class_names: List[str]) -> float:
    """Measure diagnostic accuracy assuming farmer selects the crop first (Wheat/Mustard/Chickpea/Potato)."""
    correct = 0
    for i, true_c in enumerate(y_true_idx):
        crop = class_names[true_c].split("_")[0]
        candidate_indices = [idx for idx, cname in enumerate(class_names) if cname.startswith(crop + "_")]
        best_c = max(candidate_indices, key=lambda idx: probs[i, idx])
        if best_c == true_c:
            correct += 1
    return float(correct) / float(len(y_true_idx))


def train_pipeline(image_dir: str | None = None) -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    if image_dir:
        print(f"Extracting 28 pathology features from custom folder: {image_dir}")
        df = _load_image_folder(image_dir)
    else:
        csv_path = os.path.join(root_dir, "data", "crop_features_dataset.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found at {csv_path}. Run generate_rabi_dataset.py first.")
        df = pd.read_csv(csv_path)

    class_names = [c for c in CLASSES if c in set(df["label"].unique())]
    if not class_names:
        class_names = sorted(df["label"].unique().tolist())

    le = LabelEncoder()
    le.fit(class_names)
    # Preserve explicit order in CLASSES for crisp grouping by crop
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y = df["label"].map(class_to_idx).values
    X = df[FEATURE_NAMES].values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=45,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=4.0,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
        "ann_mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=32,
            learning_rate_init=0.0025,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.12,
            random_state=42,
        ),
        "ann_deep": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=3e-3,
            batch_size=32,
            learning_rate="adaptive",
            learning_rate_init=0.002,
            max_iter=600,
            early_stopping=True,
            n_iter_no_change=25,
            validation_fraction=0.12,
            random_state=42,
        ),
    }

    display_names = {
        "random_forest": "Random Forest (ML)",
        "svm_rbf": "RBF Kernel SVM (ML)",
        "ann_mlp": "MLP Neural Network (ANN 64-32)",
        "ann_deep": "Deep Regularized ANN (128-64-32)",
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_report: Dict[str, Any] = {
        "dataset": {
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "num_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "classes": class_names,
        },
        "models": {},
    }

    fitted_models: Dict[str, Any] = {}
    test_probs: Dict[str, np.ndarray] = {}
    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))

    print("=" * 76)
    print("Rabi Crop Disease Classification: ML vs ANN Benchmark (12 Classes)")
    print("=" * 76)

    for key, estimator in models.items():
        cv_res = cross_validate(
            estimator, X_train_s, y_train, cv=cv, scoring=["accuracy", "f1_macro"]
        )

        t0 = time.perf_counter()
        estimator.fit(X_train_s, y_train)
        fit_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        y_pred = estimator.predict(X_test_s)
        probs = estimator.predict_proba(X_test_s)
        inf_ms = (time.perf_counter() - t1) * 1000.0

        fitted_models[key] = estimator
        test_probs[key] = probs

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))
        crop_acc = _crop_filtered_accuracy(y_test, probs, class_names)
        macro_auc = float(roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovr"))
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names)))).tolist()
        cls_rep = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        m_entry: Dict[str, Any] = {
            "display_name": display_names[key],
            "cv_accuracy_mean": round(float(cv_res["test_accuracy"].mean()), 4),
            "cv_accuracy_std": round(float(cv_res["test_accuracy"].std()), 4),
            "cv_f1_macro_mean": round(float(cv_res["test_f1_macro"].mean()), 4),
            "test_accuracy": round(acc, 4),
            "test_f1_macro": round(f1, 4),
            "crop_filtered_accuracy": round(crop_acc, 4),
            "macro_roc_auc": round(macro_auc, 4),
            "train_time_ms": round(fit_ms, 2),
            "test_batch_infer_ms": round(inf_ms, 2),
            "confusion_matrix": cm,
            "per_class": {
                c: {
                    "precision": round(cls_rep[c]["precision"], 4),
                    "recall": round(cls_rep[c]["recall"], 4),
                    "f1": round(cls_rep[c]["f1-score"], 4),
                    "support": int(cls_rep[c]["support"]),
                }
                for c in class_names
            },
        }
        if hasattr(estimator, "loss_curve_"):
            m_entry["loss_curve"] = [round(float(x), 5) for x in estimator.loss_curve_]
        if hasattr(estimator, "validation_scores_") and estimator.validation_scores_:
            m_entry["val_scores"] = [round(float(x), 4) for x in estimator.validation_scores_]
        if hasattr(estimator, "feature_importances_"):
            m_entry["feature_importances"] = {
                fn: round(float(imp), 5) for fn, imp in zip(FEATURE_NAMES, estimator.feature_importances_)
            }

        metrics_report["models"][key] = m_entry
        print(
            f"{display_names[key]:<34} | CV: {m_entry['cv_accuracy_mean']*100:5.2f}% "
            f"| Test Acc: {acc*100:5.2f}% | Crop-Mode: {crop_acc*100:5.2f}% | AUC: {macro_auc:.4f}"
        )

    # Soft-Vote Hybrid Ensemble (Deep ANN + MLP ANN + RBF SVM + Random Forest)
    weights = {"ann_deep": 0.35, "ann_mlp": 0.25, "svm_rbf": 0.20, "random_forest": 0.20}
    ens_probs = sum(weights[k] * test_probs[k] for k in weights)
    ens_pred = np.argmax(ens_probs, axis=1)
    ens_acc = float(accuracy_score(y_test, ens_pred))
    ens_f1 = float(f1_score(y_test, ens_pred, average="macro"))
    ens_crop_acc = _crop_filtered_accuracy(y_test, ens_probs, class_names)
    ens_auc = float(roc_auc_score(y_test_bin, ens_probs, average="macro", multi_class="ovr"))
    ens_cm = confusion_matrix(y_test, ens_pred, labels=list(range(len(class_names)))).tolist()
    ens_rep = classification_report(
        y_test, ens_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    metrics_report["models"]["hybrid_ensemble"] = {
        "display_name": "Hybrid Ensemble (ANN + SVM + RF)",
        "weights": weights,
        "test_accuracy": round(ens_acc, 4),
        "test_f1_macro": round(ens_f1, 4),
        "crop_filtered_accuracy": round(ens_crop_acc, 4),
        "macro_roc_auc": round(ens_auc, 4),
        "confusion_matrix": ens_cm,
        "per_class": {
            c: {
                "precision": round(ens_rep[c]["precision"], 4),
                "recall": round(ens_rep[c]["recall"], 4),
                "f1": round(ens_rep[c]["f1-score"], 4),
                "support": int(ens_rep[c]["support"]),
            }
            for c in class_names
        },
    }
    print("-" * 76)
    print(
        f"{'Hybrid Ensemble (ANN+SVM+RF)':<34} | Test Acc: {ens_acc*100:5.2f}% "
        f"| Crop-Mode: {ens_crop_acc*100:5.2f}% | Macro F1: {ens_f1:.4f} | AUC: {ens_auc:.4f}"
    )

    # Save Python binary bundle
    pkl_path = os.path.join(models_dir, "trained_models.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "models": fitted_models,
                "weights": weights,
                "class_names": class_names,
                "feature_names": FEATURE_NAMES,
            },
            f,
        )

    # Save Evaluation Metrics JSON
    metrics_path = os.path.join(models_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    # Export compact Android Offline JSON bundle
    android_bundle = {
        "version": "1.0.0",
        "img_size": 128,
        "feature_names": FEATURE_NAMES,
        "classes": class_names,
        "scaler": {
            "mean": np.round(scaler.mean_, 6).tolist(),
            "scale": np.round(scaler.scale_, 6).tolist(),
        },
        "ensemble_weights": weights,
        "ann_mlp": _export_mlp(fitted_models["ann_mlp"]),
        "ann_deep": _export_mlp(fitted_models["ann_deep"]),
        "svm_rbf": _export_svm(fitted_models["svm_rbf"]),
        "random_forest": {
            "n_estimators": len(fitted_models["random_forest"].estimators_),
            "trees": [_serialize_tree(t) for t in fitted_models["random_forest"].estimators_],
        },
        "metrics_summary": {
            k: {
                "display_name": v["display_name"],
                "test_accuracy": v["test_accuracy"],
                "crop_filtered_accuracy": v["crop_filtered_accuracy"],
                "test_f1_macro": v["test_f1_macro"],
            }
            for k, v in metrics_report["models"].items()
        },
    }

    android_json_path = os.path.join(models_dir, "android_models.json")
    with open(android_json_path, "w", encoding="utf-8") as f:
        json.dump(android_bundle, f, separators=(",", ":"))

    bundle_kb = os.path.getsize(android_json_path) / 1024.0
    print(f"\nExported Android Offline Model Bundle -> {android_json_path} ({bundle_kb:.1f} KB)")
    print(f"Saved Evaluation Metrics -> {metrics_path}")
    return metrics_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML & ANN models for Rabi Crop Disease Classification.")
    parser.add_argument("--image-dir", type=str, default=None, help="Optional folder of real images grouped by class subfolder.")
    args = parser.parse_args()
    train_pipeline(image_dir=args.image_dir)
