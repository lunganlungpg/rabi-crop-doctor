"""
build_web_bundle.py
-------------------
Packages trained offline model weights, bilingual ICAR/KVK crop knowledge base,
India Rabi regional incidence surveillance data, and 12 bundled sample leaf
data-URLs into a single WebView-safe JS bundle (web/js/models_data.js).
"""

from __future__ import annotations

import base64
import json
import os
import pandas as pd

from generate_rabi_dataset import CLASSES


def build_bundle() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))

    with open(os.path.join(root_dir, "models", "android_models.json"), "r", encoding="utf-8") as f:
        models_json = json.load(f)

    with open(os.path.join(root_dir, "data", "rabi_crops_knowledge.json"), "r", encoding="utf-8") as f:
        knowledge_json = json.load(f)

    with open(os.path.join(root_dir, "models", "evaluation_metrics.json"), "r", encoding="utf-8") as f:
        metrics_json = json.load(f)

    inc_df = pd.read_csv(os.path.join(root_dir, "data", "rabi_field_incidence_india.csv"))
    incidence_records = inc_df.to_dict(orient="records")

    # Encode 1 representative 128x128 PNG per class so farmers can demo offline cleanly
    samples = []
    sample_dir = os.path.join(root_dir, "dataset_samples")
    for cls_name in CLASSES:
        fpath = os.path.join(sample_dir, f"{cls_name}_sample_1.png")
        if os.path.exists(fpath):
            with open(fpath, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("ascii")
            samples.append({
                "id": cls_name,
                "crop": cls_name.split("_")[0],
                "data_url": f"data:image/png;base64,{b64}",
            })

    compact_metrics = {
        "dataset": metrics_json["dataset"],
        "models": {
            k: {
                "display_name": v["display_name"],
                "test_accuracy": v["test_accuracy"],
                "test_f1_macro": v["test_f1_macro"],
                "crop_filtered_accuracy": v["crop_filtered_accuracy"],
                "macro_roc_auc": v["macro_roc_auc"],
            }
            for k, v in metrics_json["models"].items()
        },
        "top_features": sorted(
            metrics_json["models"]["random_forest"]["feature_importances"].items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:10],
    }

    payload = {
        "models": models_json,
        "knowledge": knowledge_json,
        "incidence_india": incidence_records,
        "metrics": compact_metrics,
        "samples": samples,
    }

    js_out = os.path.join(root_dir, "web", "js", "models_data.js")
    os.makedirs(os.path.dirname(js_out), exist_ok=True)
    with open(js_out, "w", encoding="utf-8") as f:
        f.write("window.RABI_BUNDLE = ")
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
        f.write(";\n")

    kb_size = os.path.getsize(js_out) / 1024.0
    print(f"Built offline JS bundle -> {js_out} ({kb_size:.1f} KB)")


if __name__ == "__main__":
    build_bundle()
