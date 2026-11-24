"""
predict_image.py
----------------
CLI leaf pathology predictor using saved ML & ANN models.
Outputs diagnosis, disease severity %, 0-5 grade, and side-by-side
Random Forest, RBF-SVM, MLP-ANN, Deep-ANN, and Ensemble confidence scores.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import numpy as np

from feature_extractor import extract_features_and_diagnostics
from disease_incidence import severity_to_grade


def predict_leaf_image(image_path: str, crop_filter: str | None = None) -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    pkl_path = os.path.join(root_dir, "models", "trained_models.pkl")
    kb_path = os.path.join(root_dir, "data", "rabi_crops_knowledge.json")

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    diag = extract_features_and_diagnostics(image_path)
    if not diag["leaf_detected"]:
        return {"leaf_detected": False, "message": "Leaf not clearly detected. Center leaf in good lighting."}

    x_scaled = bundle["scaler"].transform(diag["features"].reshape(1, -1))
    class_names = bundle["class_names"]
    weights = bundle["weights"]

    per_model = {}
    ens_probs = np.zeros(len(class_names), dtype=np.float64)

    for key, est in bundle["models"].items():
        probs = est.predict_proba(x_scaled)[0]
        ens_probs += weights[key] * probs
        per_model[key] = probs

    def _mask_crop(probs_arr: np.ndarray) -> np.ndarray:
        if not crop_filter:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a Rabi crop leaf photo using trained ML & ANN models.")
    parser.add_argument("image", help="Path to leaf image (.png/.jpg)")
    parser.add_argument("--crop", default=None, help="Optional crop filter: wheat, mustard, chickpea, potato")
    args = parser.parse_args()
    res = predict_leaf_image(args.image, crop_filter=args.crop)
    print(json.dumps(res, indent=2, ensure_ascii=False))
