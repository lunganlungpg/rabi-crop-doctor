"""
test_rabi_timeline_and_kaggle.py
--------------------------------
Offline verification tests for:
1. Project timeline resolution (ROADMAP.md quarters + data/rabi_crops_knowledge.json
   + data/rabi_field_incidence_india.csv).
2. Selective Rabi crop label parsing & exclusion of non-Rabi crops (e.g., ensuring
   'Healthy Wheat' and 'Flag Smut' map to 'wheat' while 'Yellow Rust Sugarcane'
   and Cotton 'Wilt' are excluded).
3. Merging nested subfolders such as Train/Healthy Wheat/Wheat___Healthy.
4. Official Train + Validation split handling (Validation held out unchanged).
5. End-to-end selective training over a synthetic Train/Validation tree.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageEnhance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "python"))

from rabi_timeline import resolve_rabi_timeline
from train_kaggle_dataset import _canonical_label, _collect_records, _infer_crop, _make_split, train


def test_timeline_resolution() -> None:
    q1_nov = resolve_rabi_timeline(as_of="2026-11-28", mode="roadmap")
    assert q1_nov["quarter"] == "Q1"
    assert q1_nov["crops"] == ["wheat", "mustard", "chickpea", "potato"]

    q2_apr = resolve_rabi_timeline(as_of="2027-04-15", mode="roadmap")
    assert q2_apr["quarter"] == "Q2"
    assert q2_apr["crops"] == ["wheat", "mustard", "chickpea", "potato"]

    q3_jul = resolve_rabi_timeline(as_of="2027-07-10", mode="roadmap")
    assert q3_jul["quarter"] == "Q3"
    assert q3_jul["crops"] == ["wheat", "mustard", "chickpea", "potato", "barley", "lentil", "field_pea"]

    q4_oct = resolve_rabi_timeline(as_of="2027-10-05", mode="roadmap")
    assert q4_oct["quarter"] == "Q4"
    assert "barley" in q4_oct["crops"] and "lentil" in q4_oct["crops"] and "field_pea" in q4_oct["crops"]

    peak_jan = resolve_rabi_timeline(as_of="2027-01-15", mode="peak-month")
    assert set(peak_jan["crops"]) == {"wheat", "mustard", "chickpea", "potato"}

    peak_mar = resolve_rabi_timeline(as_of="2027-03-15", mode="peak-month")
    assert peak_mar["crops"] == ["wheat"]
    print("1. Timeline resolution tests PASSED")


def test_crop_inference_and_split() -> None:
    assert _infer_crop("Healthy Wheat") == "wheat"
    assert _canonical_label("Healthy Wheat") == "wheat_healthy"
    assert _infer_crop("Flag Smut") == "wheat"
    assert _canonical_label("Flag Smut") == "wheat_flag_smut"
    assert _infer_crop("Wheat___Yellow_Rust") == "wheat"
    assert _infer_crop("Wheat Brown leaf Rust") == "wheat"
    assert _infer_crop("Yellow Rust Sugarcane") == "sugarcane"
    assert _infer_crop("Wilt") == "cotton"
    assert _infer_crop("Leaf Curl") == "cotton"
    assert _infer_crop("Common_Rust") == "maize"
    assert _infer_crop("Brownspot") == "rice"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample_healthy = Path(ROOT_DIR) / "dataset_samples" / "wheat_healthy_sample_1.png"
        sample_rust = Path(ROOT_DIR) / "dataset_samples" / "wheat_yellow_rust_sample_1.png"
        sample_brown = Path(ROOT_DIR) / "dataset_samples" / "wheat_brown_rust_sample_1.png"

        # Simulate Train + Validation with nested Wheat___Healthy + Flag Smut + non-Rabi classes
        with Image.open(sample_healthy) as im_h, Image.open(sample_rust) as im_r, Image.open(sample_brown) as im_b:
            for i in range(6):
                d = root / "Train" / "Healthy Wheat"
                d.mkdir(parents=True, exist_ok=True)
                ImageEnhance.Brightness(im_h.rotate(i * 15)).enhance(0.9 + 0.02 * i).save(d / f"h_{i}.png")

            nested = root / "Train" / "Healthy Wheat" / "Wheat___Healthy"
            nested.mkdir(parents=True, exist_ok=True)
            for i in range(4):
                ImageEnhance.Brightness(im_h.rotate(i * 23 + 3)).enhance(0.95 + 0.02 * i).save(nested / f"nh_{i}.png")

            for cls_idx, (cls_dir, src_im) in enumerate([
                ("Wheat___Yellow_Rust", im_r),
                ("Flag Smut", im_b),
                ("Yellow Rust Sugarcane", im_r),
                ("Wilt", im_b),
            ]):
                td = root / "Train" / cls_dir
                vd = root / "Validation" / cls_dir
                td.mkdir(parents=True, exist_ok=True)
                vd.mkdir(parents=True, exist_ok=True)
                for i in range(8):
                    ImageEnhance.Brightness(src_im.rotate(i * 19 + cls_idx * 5)).enhance(0.85 + 0.02 * i + 0.01 * cls_idx).save(td / f"t_{i}.png")
                for i in range(3):
                    ImageEnhance.Brightness(src_im.rotate(i * 29 + 7 + cls_idx * 5)).enhance(0.91 + 0.02 * i + 0.01 * cls_idx).save(vd / f"v_{i}.png")

            vd_h = root / "Validation" / "Healthy Wheat"
            vd_h.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                ImageEnhance.Brightness(im_h.rotate(i * 37 + 11)).enhance(0.92 + 0.03 * i).save(vd_h / f"vh_{i}.png")

        records = _collect_records(root, crops=["wheat", "mustard", "chickpea", "potato"])
        labels = {r["label"] for r in records}
        assert labels == {"wheat_healthy", "wheat_yellow_rust", "wheat_flag_smut"}, labels

        tr, te, split_desc, excluded = _make_split(records, test_size=0.2, random_state=42)
        assert "Validation held out unchanged" in split_desc
        assert len(tr) == 26 and len(te) == 9, (len(tr), len(te))
        assert excluded == []

        out_dir = root / "out"
        timeline_ctx = resolve_rabi_timeline(as_of="2026-11-28", mode="roadmap")
        metrics = train(
            dataset_dir=root,
            output_dir=out_dir,
            crops=timeline_ctx["crops"],
            test_size=0.2,
            cv_folds=3,
            random_state=42,
            skip_hash_dedup=False,
            timeline_context=timeline_ctx,
            download_mode="selective_http_range",
        )
        assert metrics["dataset"]["crops_present_in_dataset"] == ["wheat"]
        assert metrics["dataset"]["crops_missing_from_dataset"] == ["mustard", "chickpea", "potato"]
        assert (out_dir / "trained_models.pkl").exists()
        assert (out_dir / "evaluation_metrics.json").exists()
        assert (out_dir / "training_summary.md").exists()
        assert (out_dir / "held_out_predictions.csv").exists()
    print("2. Crop inference, Train/Validation split & selective training tests PASSED")


if __name__ == "__main__":
    test_timeline_resolution()
    test_crop_inference_and_split()
    print("\nALL RABI TIMELINE & SELECTIVE KAGGLE TESTS PASSED!")
