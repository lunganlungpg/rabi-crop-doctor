"""
predict_image.py
----------------
CLI and library predictor for classifying Rabi crop leaf images using trained ML
and ANN models. Supports scikit-learn pickles locally and falls back cleanly to
the exported JSON weights (android_models.json) when installed via pip.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from typing import Any, Dict, List
import numpy as np

try:
    from ._paths import resolve_asset_path
    from .disease_incidence import severity_to_grade
    from .feature_extractor import extract_features_and_diagnostics
except ImportError:
    from _paths import resolve_asset_path
    from disease_incidence import severity_to_grade
    from feature_extractor import extract_features_and_diagnostics


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax activation for ANN output layers."""
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _predict_mlp_json(x_scaled: np.ndarray, mlp_spec: Dict[str, Any]) -> np.ndarray:
    """Evaluate an exported ReLU MLP Artificial Neural Network using pure NumPy."""
    act = x_scaled.astype(np.float64)
    layers = mlp_spec["layers"]
    for idx, layer in enumerate(layers):
        W = np.asarray(layer["W"], dtype=np.float64)
        b = np.asarray(layer["b"], dtype=np.float64)
        z = act @ W + b
        act = np.maximum(0.0, z) if idx < len(layers) - 1 else _softmax(z)
    return act


def _predict_rf_json(x_scaled: np.ndarray, rf_spec: Dict[str, Any], n_classes: int) -> np.ndarray:
    """Traverse serialized Decision Trees and average class posterior probabilities."""
    probs = np.zeros(n_classes, dtype=np.float64)
    trees = rf_spec["trees"]
    for tree in trees:
        node = 0
        while tree["children_left"][node] != -1:
            f_idx = tree["feature"][node]
            node = (
                tree["children_left"][node]
                if x_scaled[f_idx] <= tree["threshold"][node]
                else tree["children_right"][node]
            )
        probs += np.asarray(tree["probs"][node], dtype=np.float64)
    return probs / float(len(trees))


def _multiclass_coupling(k: int, r: np.ndarray) -> np.ndarray:
    """Wu, Lin & Weng (2004) Algorithm 2 iterative solver for multiclass Platt probabilities."""
    p = np.full(k, 1.0 / k, dtype=np.float64)
    Q = np.zeros((k, k), dtype=np.float64)
    for t in range(k):
        for j in range(t):
            Q[t, t] += r[j, t] ** 2
            Q[t, j] = Q[j, t]
        for j in range(t + 1, k):
            Q[t, t] += r[j, t] ** 2
            Q[t, j] = -r[j, t] * r[t, j]

    max_iter = max(100, k)
    eps = 0.005 / k
    for _ in range(max_iter):
        Qp = Q @ p
        pQp = float(p @ Qp)
        if np.max(np.abs(Qp - pQp)) < eps:
            break
        for t in range(k):
            diff = (-Qp[t] + pQp) / Q[t, t]
            p[t] += diff
            pQp = (pQp + diff * (diff * Q[t, t] + 2.0 * Qp[t])) / (1.0 + diff) ** 2
            Qp = (Qp + diff * Q[t]) / (1.0 + diff)
            p /= 1.0 + diff
    return p


def _predict_svm_json(x_scaled: np.ndarray, svm_spec: Dict[str, Any], n_classes: int) -> np.ndarray:
    """Evaluate 12-class RBF SVM via One-vs-One Platt sigmoids and pairwise coupling."""
    sv = np.asarray(svm_spec["support_vectors"], dtype=np.float64)
    gamma = float(svm_spec["gamma"])
    sq_dists = np.sum((sv - x_scaled[None, :]) ** 2, axis=1)
    k_val = np.exp(-gamma * sq_dists)

    n_sup = svm_spec["n_support"]
    starts: List[int] = []
    off = 0
    for cnt in n_sup:
        starts.append(off)
        off += int(cnt)

    dual = svm_spec["dual_coef"]
    intercept = svm_spec["intercept"]
    prob_a = svm_spec["probA"]
    prob_b = svm_spec["probB"]
    pairwise = np.zeros((n_classes, n_classes), dtype=np.float64)
    min_prob = 1e-7

    p_idx = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            si, ci = starts[i], int(n_sup[i])
            sj, cj = starts[j], int(n_sup[j])
            dec = (
                intercept[p_idx]
                + float(np.dot(dual[j - 1][si : si + ci], k_val[si : si + ci]))
                + float(np.dot(dual[i][sj : sj + cj], k_val[sj : sj + cj]))
            )
            f_apb = dec * prob_a[p_idx] + prob_b[p_idx]
            p_ij = (
                math.exp(-f_apb) / (1.0 + math.exp(-f_apb))
                if f_apb >= 0.0
                else 1.0 / (1.0 + math.exp(f_apb))
            )
            p_ij = min(max(p_ij, min_prob), 1.0 - min_prob)
            pairwise[i, j] = p_ij
            pairwise[j, i] = 1.0 - p_ij
            p_idx += 1
    return _multiclass_coupling(n_classes, pairwise)


def predict_leaf_image(image_path: str, crop_filter: str | None = None) -> Dict[str, Any]:
    """
    Classify a leaf image across all 4 ML & ANN models + Hybrid Ensemble.

    Parameters
    ----------
    image_path : str
        Path to a crop leaf photo (.png, .jpg, .webp).
    crop_filter : str | None
        Optional known crop ('wheat', 'mustard', 'chickpea', or 'potato')
        to restrict classification probabilities within that crop's diseases.
    """
    pkl_path = resolve_asset_path("models", "trained_models.pkl")
    json_model_path = resolve_asset_path("models", "android_models.json")
    kb_path = resolve_asset_path("data", "rabi_crops_knowledge.json")

    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    diag = extract_features_and_diagnostics(image_path)
    if not diag["leaf_detected"]:
        return {
            "leaf_detected": False,
            "message": "Leaf not clearly detected. Center leaf in good daylight.",
        }

    per_model: Dict[str, np.ndarray] = {}

    # Prefer local sklearn pickle when present; fall back cleanly to bundled JSON weights
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        x_scaled_2d = bundle["scaler"].transform(diag["features"].reshape(1, -1))
        class_names = bundle["class_names"]
        weights = bundle["weights"]
        ens_probs = np.zeros(len(class_names), dtype=np.float64)
        for key, est in bundle["models"].items():
            probs = est.predict_proba(x_scaled_2d)[0]
            ens_probs += weights[key] * probs
            per_model[key] = probs
    else:
        with open(json_model_path, "r", encoding="utf-8") as f:
            jbundle = json.load(f)
        class_names = jbundle["classes"]
        weights = jbundle["ensemble_weights"]
        mean = np.asarray(jbundle["scaler"]["mean"], dtype=np.float64)
        scale = np.asarray(jbundle["scaler"]["scale"], dtype=np.float64)
        x_scaled = (diag["features"] - mean) / np.where(scale > 1e-12, scale, 1.0)
        n_cls = len(class_names)

        per_model["random_forest"] = _predict_rf_json(x_scaled, jbundle["random_forest"], n_cls)
        per_model["svm_rbf"] = _predict_svm_json(x_scaled, jbundle["svm_rbf"], n_cls)
        per_model["ann_mlp"] = _predict_mlp_json(x_scaled, jbundle["ann_mlp"])
        per_model["ann_deep"] = _predict_mlp_json(x_scaled, jbundle["ann_deep"])
        ens_probs = sum(weights[k] * per_model[k] for k in weights)

    def _mask_crop(probs_arr: np.ndarray) -> np.ndarray:
        """Renormalize class posterior over a single crop when farmer selects crop upfront."""
        if not crop_filter or crop_filter.lower().strip() == "all":
            return probs_arr
        c_prefix = crop_filter.lower().strip() + "_"
        mask = np.array([1.0 if cn.startswith(c_prefix) else 0.0 for cn in class_names], dtype=np.float64)
        s = float(np.sum(probs_arr * mask))
        return (probs_arr * mask / s) if s > 1e-9 else probs_arr

    ens_masked = _mask_crop(ens_probs)
    best_idx = int(np.argmax(ens_masked))
    best_class = class_names[best_idx]
    cls_info = next((c for c in kb["classes"] if c["id"] == best_class), {})

    model_breakdown = {}
    for key, p in per_model.items():
        mp = _mask_crop(p)
        mi = int(np.argmax(mp))
        model_breakdown[key] = {
            "predicted_class": class_names[mi],
            "confidence_pct": round(float(mp[mi]) * 100.0, 2),
        }

    is_healthy = bool(cls_info.get("is_healthy", False))
    sev_pct = 0.0 if is_healthy else diag["estimated_severity_pct"]
    grade = 0 if is_healthy else severity_to_grade(sev_pct)

    return {
        "leaf_detected": True,
        "image": image_path,
        "predicted_class": best_class,
        "name_en": cls_info.get("name_en", best_class),
        "name_hi": cls_info.get("name_hi", ""),
        "confidence_pct": round(float(ens_masked[best_idx]) * 100.0, 2),
        "estimated_severity_pct": sev_pct,
        "severity_grade": grade,
        "model_breakdown": model_breakdown,
        "advisory_en": cls_info.get("action_en", []),
    }


def cli_main() -> None:
    """Entrypoint for the installed `rabi-doctor` CLI console script."""
    parser = argparse.ArgumentParser(
        description="Classify a Rabi crop leaf photo using trained ML & ANN models."
    )
    parser.add_argument("image", help="Path to leaf image (.png/.jpg/.webp)")
    parser.add_argument(
        "--crop",
        default=None,
        help="Optional crop filter: wheat, mustard, chickpea, potato",
    )
    args = parser.parse_args()
    res = predict_leaf_image(args.image, crop_filter=args.crop)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli_main()
