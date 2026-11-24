"""
test_parity_and_pipeline.py
---------------------------
End-to-end verification suite checking:
1. Dataset integrity & class balance (1,440 samples, 12 classes).
2. Python vs Node.js exact feature-extraction parity across curated images.
3. Python vs Node.js model inference probability parity across ALL 4 models
   (Random Forest, RBF SVM, MLP ANN, Deep ANN) and Hybrid Ensemble.
4. Field Disease Incidence (%) & Wheeler PDI math accuracy.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "python"))

from feature_extractor import FEATURE_NAMES, extract_features_and_diagnostics
from disease_incidence import compute_field_survey_metrics
from generate_rabi_dataset import CLASSES


def run_node_inference(rgba_bytes: list[int]) -> dict:
    script = r"""
const fs = require('fs');
const fe = require(process.argv[1]);
const ie = require(process.argv[2]);
const models = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const rgba = new Uint8ClampedArray(input.rgba);
const diag = fe.extractFeaturesFromRGBA(rgba, 128, 128);
const res = ie.predictAllModels(diag.features, models, 'all');
console.log(JSON.stringify({ diag, res }));
"""
    fe_path = os.path.join(ROOT_DIR, "web", "js", "feature_extractor.js")
    ie_path = os.path.join(ROOT_DIR, "web", "js", "inference_engine.js")
    m_path = os.path.join(ROOT_DIR, "models", "android_models.json")

    proc = subprocess.run(
        ["node", "-e", script, fe_path, ie_path, m_path],
        input=json.dumps({"rgba": rgba_bytes}),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_pipeline() -> None:
    print("1. Checking trained bundle & dataset artifacts...")
    with open(os.path.join(ROOT_DIR, "models", "trained_models.pkl"), "rb") as f:
        pkl_bundle = pickle.load(f)
    with open(os.path.join(ROOT_DIR, "models", "evaluation_metrics.json"), "r", encoding="utf-8") as f:
        eval_metrics = json.load(f)

    assert eval_metrics["dataset"]["total_samples"] == 1440
    assert len(eval_metrics["dataset"]["classes"]) == 12
    ens_acc = eval_metrics["models"]["hybrid_ensemble"]["test_accuracy"]
    assert ens_acc >= 0.85, f"Unexpected low ensemble accuracy: {ens_acc}"
    print(f"   -> Dataset: 1440 samples | Ensemble Test Accuracy: {ens_acc*100:.2f}%")

    print("2. Checking Python <-> JavaScript Feature & Model Inference Parity across 12 classes...")
    max_feat_diff = 0.0
    max_prob_diff = {k: 0.0 for k in ["ann_mlp", "ann_deep", "random_forest", "svm_rbf", "hybrid_ensemble"]}

    for cls_name in CLASSES:
        sample_path = os.path.join(ROOT_DIR, "dataset_samples", f"{cls_name}_sample_1.png")
        with Image.open(sample_path) as img:
            img_128 = img.convert("RGBA").resize((128, 128), Image.Resampling.BILINEAR)
            rgba_arr = np.asarray(img_128, dtype=np.uint8)
            rgb_arr = rgba_arr[:, :, :3]

        py_diag = extract_features_and_diagnostics(rgb_arr)
        assert py_diag["leaf_detected"], f"Leaf not detected for {cls_name}"

        py_x_s = pkl_bundle["scaler"].transform(py_diag["features"].reshape(1, -1))
        py_probs = {}
        ens_py = np.zeros(len(CLASSES), dtype=np.float64)
        for k in ["ann_mlp", "ann_deep", "random_forest", "svm_rbf"]:
            p = pkl_bundle["models"][k].predict_proba(py_x_s)[0]
            py_probs[k] = p
            ens_py += pkl_bundle["weights"][k] * p
        py_probs["hybrid_ensemble"] = ens_py

        js_out = run_node_inference(rgba_arr.flatten().tolist())
        js_feats = np.array(js_out["diag"]["features"], dtype=np.float64)

        feat_diff = float(np.max(np.abs(py_diag["features"] - js_feats)))
        max_feat_diff = max(max_feat_diff, feat_diff)
        assert feat_diff < 1e-4, f"Feature mismatch on {cls_name}: {feat_diff}"

        for k in max_prob_diff:
            js_p = np.array(js_out["res"]["rawProbs"][k], dtype=np.float64)
            p_diff = float(np.max(np.abs(py_probs[k] - js_p)))
            max_prob_diff[k] = max(max_prob_diff[k], p_diff)
            assert p_diff < 2e-2, f"Probability mismatch on {cls_name} ({k}): {p_diff}"

    print(f"   -> Max 28-feature diff (Python vs JS): {max_feat_diff:.2e}")
    for k, d in max_prob_diff.items():
        print(f"   -> Max {k:<16} prob diff: {d:.2e}")

    print("3. Verifying Disease Incidence & Wheeler PDI Math...")
    survey = compute_field_survey_metrics([
        {"is_healthy": True, "severity_pct": 0.5, "disease": "wheat_healthy"},
        {"is_healthy": False, "severity_pct": 8.0, "disease": "wheat_yellow_rust"},
        {"is_healthy": False, "severity_pct": 40.0, "disease": "wheat_yellow_rust"},
        {"is_healthy": True, "severity_pct": 1.0, "disease": "wheat_healthy"},
    ])
    # 2/4 infected = 50% incidence; grades = 0, 1, 3, 0 => sum=4 / (4*5=20) = 20.0% PDI
    assert survey["disease_incidence_pct"] == 50.0
    assert survey["pdi_pct"] == 20.0
    print(f"   -> Incidence: {survey['disease_incidence_pct']}% | PDI: {survey['pdi_pct']}% (PASSED)")

    print("4. Verifying Python Pure-NumPy JSON Fallback & Node index.js Wrapper...")
    from predict_image import _predict_mlp_json, _predict_rf_json, _predict_svm_json
    with open(os.path.join(ROOT_DIR, "models", "android_models.json"), "r", encoding="utf-8") as f:
        jbundle = json.load(f)
    sample_w = os.path.join(ROOT_DIR, "dataset_samples", "wheat_yellow_rust_sample_1.png")
    with Image.open(sample_w) as img:
        img_128 = img.convert("RGBA").resize((128, 128), Image.Resampling.BILINEAR)
        rgba_w = np.asarray(img_128, dtype=np.uint8)
    w_diag = extract_features_and_diagnostics(rgba_w[:, :, :3])
    mean = np.asarray(jbundle["scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(jbundle["scaler"]["scale"], dtype=np.float64)
    xs = (w_diag["features"] - mean) / scale
    rf_j = _predict_rf_json(xs, jbundle["random_forest"], len(CLASSES))
    svm_j = _predict_svm_json(xs, jbundle["svm_rbf"], len(CLASSES))
    mlp_j = _predict_mlp_json(xs, jbundle["ann_mlp"])
    deep_j = _predict_mlp_json(xs, jbundle["ann_deep"])
    assert CLASSES[int(np.argmax(deep_j))] == "wheat_yellow_rust"
    assert CLASSES[int(np.argmax(rf_j))] == "wheat_yellow_rust"
    assert CLASSES[int(np.argmax(svm_j))] == "wheat_yellow_rust"
    assert CLASSES[int(np.argmax(mlp_j))] == "wheat_yellow_rust"

    node_pkg_script = r"""
const pkg = require(process.argv[1]);
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = pkg.classifyLeafRGBA(new Uint8ClampedArray(input.rgba), 'wheat');
console.log(JSON.stringify(out));
"""
    pkg_proc = subprocess.run(
        ["node", "-e", node_pkg_script, os.path.join(ROOT_DIR, "index.js")],
        input=json.dumps({"rgba": rgba_w.flatten().tolist()}),
        capture_output=True,
        text=True,
        check=True,
    )
    pkg_out = json.loads(pkg_proc.stdout)
    assert pkg_out["predictedClass"] == "wheat_yellow_rust", f"Node pkg returned {pkg_out}"
    print(f"   -> JSON Fallback + Node Package: {pkg_out['predictedClass']} ({pkg_out['confidencePct']}%) [PASSED]")
    print("\nALL PYTHON, JSON FALLBACK, NODE PKG & EPIDEMIOLOGY TESTS PASSED!")


if __name__ == "__main__":
    test_pipeline()
