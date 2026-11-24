"""
generate_rabi_dataset.py
------------------------
Physics-informed botanical leaf benchmark generator (RabiLeaf-1440) for four major
Indian Rabi crops: Wheat, Mustard, Chickpea, and Potato across 12 health/disease classes.

Produces 128x128 RGB PNG images (120 per class = 1,440 images), representative
preview images in dataset_samples/, and extracts the deterministic 28-feature
dataset CSV for reproducible ML and ANN training.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image

from feature_extractor import FEATURE_NAMES, extract_features_and_diagnostics

CLASSES = [
    "wheat_healthy",
    "wheat_yellow_rust",
    "wheat_brown_rust",
    "mustard_healthy",
    "mustard_alternaria_blight",
    "mustard_white_rust",
    "chickpea_healthy",
    "chickpea_ascochyta_blight",
    "chickpea_fusarium_wilt",
    "potato_healthy",
    "potato_early_blight",
    "potato_late_blight",
]

SIZE = 128


def _make_background(rng: np.random.Generator) -> np.ndarray:
    """Create diverse natural field/soil/sky/straw backgrounds so models ignore backdrops."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64) / SIZE
    bg_type = int(rng.integers(0, 5))
    noise = rng.normal(0.0, 0.035, (SIZE, SIZE, 1))

    if bg_type == 0:
        # Loamy Indo-Gangetic alluvial soil
        base = np.array([0.42, 0.34, 0.25]) + rng.uniform(-0.06, 0.06, 3)
        specks = (rng.random((SIZE, SIZE, 1)) > 0.92) * rng.uniform(-0.08, 0.08, (SIZE, SIZE, 1))
        bg = base[None, None, :] + noise + specks
    elif bg_type == 1:
        # Dry wheat/mustard straw mulch
        base = np.array([0.62, 0.54, 0.38]) + rng.uniform(-0.05, 0.05, 3)
        streaks = 0.04 * np.sin((xx * 28.0 + yy * 14.0) * rng.uniform(0.8, 1.4))[:, :, None]
        bg = base[None, None, :] + streaks + noise * 0.8
    elif bg_type == 2:
        # Soft morning overcast sky / out-of-focus background
        grad = 0.08 * (1.0 - yy)[:, :, None]
        base = np.array([0.64, 0.72, 0.78]) + rng.uniform(-0.05, 0.05, 3)
        bg = base[None, None, :] + grad + noise * 0.4
    elif bg_type == 3:
        # Dark tilled furrow / shaded soil
        base = np.array([0.22, 0.19, 0.16]) + rng.uniform(-0.04, 0.04, 3)
        bg = base[None, None, :] + noise * 0.9
    else:
        # Farmer paper / neutral cloth inspection backdrop
        base = np.array([0.82, 0.80, 0.76]) + rng.uniform(-0.05, 0.05, 3)
        bg = base[None, None, :] + noise * 0.3

    return np.clip(bg, 0.03, 0.96)


def _crop_geometry(crop: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build crop-specific leaf mask and normalized anatomical coordinates:
    (mask, u_along, v_across, dist_from_edge, vein_map).
    """
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    cx = SIZE * 0.5 + rng.uniform(-6.0, 6.0)
    cy = SIZE * 0.5 + rng.uniform(-6.0, 6.0)
    theta = rng.uniform(-math.pi, math.pi)
    scale = rng.uniform(0.86, 1.08)

    dx = (xx - cx) / (SIZE * 0.44 * scale)
    dy = (yy - cy) / (SIZE * 0.44 * scale)
    u = dx * math.cos(theta) + dy * math.sin(theta)
    v = -dx * math.sin(theta) + dy * math.cos(theta)

    if crop == "wheat":
        # Long narrow ribbon leaf blade with slight natural arch
        v_arch = v - 0.12 * (u * u - 0.3)
        half_w = 0.21 * np.sqrt(np.maximum(0.0, 1.0 - 0.85 * (u ** 2))) * (1.0 - 0.25 * np.maximum(0.0, u))
        mask = (np.abs(u) <= 0.95) & (np.abs(v_arch) <= np.maximum(half_w, 0.02)) & (half_w > 0.015)
        dist_edge = np.clip((half_w - np.abs(v_arch)) / (half_w + 1e-6), 0.0, 1.0)
        vein = (np.exp(-((v_arch * 45.0) ** 2)) * 0.8 + 0.35 * (np.cos(v_arch * 120.0) ** 8)) * mask
        return mask, u, v_arch, dist_edge, vein

    if crop == "mustard":
        # Broad scalloped Brassica lamina with basal lobes
        ang = np.arctan2(v, u)
        scallop = 1.0 + 0.12 * np.cos(6.0 * ang) + 0.07 * np.sin(10.0 * ang)
        r_ell = np.sqrt((u / 0.88) ** 2 + (v / 0.72) ** 2)
        r_bound = 0.82 * scallop
        mask = r_ell <= r_bound
        dist_edge = np.clip((r_bound - r_ell) / (r_bound + 1e-6), 0.0, 1.0)
        midrib = np.exp(-((v * 35.0) ** 2)) * (u > -0.8) * mask
        side_veins = np.exp(-(((np.abs(v) - 0.48 * (u + 0.55)) * 18.0) ** 2)) * 0.5
        vein = np.clip(midrib + side_veins * 0.35, 0.0, 1.0) * mask
        return mask, u, v, dist_edge, vein

    if crop == "chickpea":
        # Pinnate compound leaf: rachis + 9 small elliptical serrated leaflets
        mask = np.abs(v) < 0.025
        mask &= np.abs(u) <= 0.85
        vein = (np.abs(v) < 0.02) * 0.8
        dist_edge = np.zeros((SIZE, SIZE), dtype=np.float64)
        for idx, u_pos in enumerate([-0.68, -0.36, -0.04, 0.28, 0.60]):
            sides = [0.0] if idx == 4 else [-1.0, 1.0]
            for side in sides:
                lu = u - (u_pos + (0.14 if side == 0 else 0.0))
                lv = v - side * 0.23
                lu_r = lu * math.cos(0.45 * side) + lv * math.sin(0.45 * side)
                lv_r = -lu * math.sin(0.45 * side) + lv * math.cos(0.45 * side)
                ang = np.arctan2(lv_r, lu_r)
                serr = 1.0 + 0.06 * np.cos(12.0 * ang)
                rad = np.sqrt((lu_r / 0.17) ** 2 + (lv_r / 0.115) ** 2)
                lm = rad <= serr
                mask |= lm
                dist_edge = np.maximum(dist_edge, np.clip((serr - rad) / (serr + 1e-6), 0.0, 1.0) * lm)
                vein = np.maximum(vein, np.exp(-((lv_r * 45.0) ** 2)) * 0.55 * lm)
        return mask, u, v, dist_edge, vein

    # Potato: large terminal + 2 lateral ovate ribbed leaflets
    mask = np.zeros((SIZE, SIZE), dtype=np.float64).astype(bool)
    dist_edge = np.zeros((SIZE, SIZE), dtype=np.float64)
    vein = np.zeros((SIZE, SIZE), dtype=np.float64)

    # Terminal leaflet
    ang_t = np.arctan2(v, u - 0.18)
    serr_t = 1.0 + 0.05 * np.cos(14.0 * ang_t)
    r_t = np.sqrt(((u - 0.18) / 0.60) ** 2 + (v / 0.44) ** 2)
    mt = r_t <= serr_t
    mask |= mt
    dist_edge = np.maximum(dist_edge, np.clip((serr_t - r_t) / (serr_t + 1e-6), 0.0, 1.0) * mt)

    # Two basal lateral leaflets
    for side in (-1.0, 1.0):
        lu = u + 0.42
        lv = v - side * 0.34
        r_l = np.sqrt((lu / 0.30) ** 2 + (lv / 0.23) ** 2)
        ml = r_l <= 1.0
        mask |= ml
        dist_edge = np.maximum(dist_edge, np.clip(1.0 - r_l, 0.0, 1.0) * ml)

    rib = np.exp(-((v * 35.0) ** 2)) + 0.45 * (np.sin(u * 26.0 - np.abs(v) * 14.0) > 0.65)
    vein = np.clip(rib * 0.65, 0.0, 1.0) * mask
    return mask, u, v, dist_edge, vein


def synthesize_leaf_image(cls_name: str, seed: int) -> Tuple[np.ndarray, float]:
    """Synthesize a single 128x128 RGB leaf image and return (uint8_rgb, true_severity_pct)."""
    rng = np.random.default_rng(seed)
    crop = cls_name.split("_")[0]

    bg = _make_background(rng)
    mask, u, v, dist_edge, vein = _crop_geometry(crop, rng)
    mask_f = mask.astype(np.float64)

    # Healthy chlorophyll base lamina with natural mottling & lighting gradient
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64) / SIZE
    light = 1.0 + rng.uniform(-0.14, 0.16) * (xx - 0.5) + rng.uniform(-0.12, 0.14) * (yy - 0.5)
    chlorophyll_shift = rng.uniform(-0.04, 0.04, 3)

    if crop == "wheat":
        base_col = np.array([0.21, 0.48, 0.16]) + chlorophyll_shift
    elif crop == "mustard":
        base_col = np.array([0.19, 0.46, 0.18]) + chlorophyll_shift
    elif crop == "chickpea":
        base_col = np.array([0.17, 0.44, 0.19]) + chlorophyll_shift
    else:  # potato
        base_col = np.array([0.16, 0.43, 0.15]) + chlorophyll_shift

    mottle = rng.normal(0.0, 0.02, (SIZE, SIZE, 1))
    leaf = base_col[None, None, :] + mottle
    # Slight brighter/paler veins
    leaf = leaf + vein[:, :, None] * np.array([0.07, 0.09, 0.04])[None, None, :]

    lesion_mask = np.zeros((SIZE, SIZE), dtype=bool)

    if cls_name.endswith("_healthy"):
        # Minor natural blemishes (<1.5%)
        n_specks = int(rng.integers(0, 3))
        for _ in range(n_specks):
            su = rng.uniform(-0.6, 0.6)
            sv = rng.uniform(-0.4, 0.4)
            dist = np.sqrt((u - su) ** 2 + (v - sv) ** 2)
            sp = mask & (dist < 0.025)
            leaf[sp] = np.array([0.45, 0.48, 0.20])

    elif cls_name == "wheat_yellow_rust":
        # Linear yellow-amber pustule stripes parallel to leaf veins (along v slices)
        n_stripes = int(rng.integers(5, 11))
        for _ in range(n_stripes):
            v0 = rng.uniform(-0.16, 0.16)
            u_start = rng.uniform(-0.85, 0.0)
            u_end = min(0.92, u_start + rng.uniform(0.45, 1.35))
            thick = rng.uniform(0.013, 0.022)
            # Dashed uredinial pustules in a crisp linear line
            pulse = 0.5 + 0.5 * np.sin((u + rng.uniform(0, 1)) * 75.0)
            stripe = mask & (np.abs(v - v0) < thick) & (u >= u_start) & (u <= u_end) & (pulse > 0.25)
            halo = mask & (np.abs(v - v0) < thick * 2.1) & (u >= u_start - 0.05) & (u <= u_end + 0.05) & (~stripe)
            leaf[halo] = leaf[halo] * 0.35 + np.array([0.78, 0.78, 0.22]) * 0.65
            leaf[stripe] = np.array([0.94, 0.78, 0.08]) + rng.normal(0.0, 0.03, (int(stripe.sum()), 3))
            lesion_mask |= stripe | halo

    elif cls_name == "wheat_brown_rust":
        # Randomly scattered circular/oval reddish-brown pustules with pale chlorotic halos
        n_spots = int(rng.integers(14, 34))
        for _ in range(n_spots):
            su = rng.uniform(-0.82, 0.82)
            sv = rng.uniform(-0.16, 0.16)
            rad = rng.uniform(0.026, 0.048)
            dist = np.sqrt(((u - su) * 0.85) ** 2 + (v - sv) ** 2)
            core = mask & (dist <= rad)
            halo = mask & (dist > rad) & (dist <= rad * 1.7)
            leaf[halo] = leaf[halo] * 0.45 + np.array([0.76, 0.72, 0.24]) * 0.55
            leaf[core] = np.array([0.62, 0.27, 0.09]) + rng.normal(0.0, 0.04, (int(core.sum()), 3))
            lesion_mask |= core | halo

    elif cls_name == "mustard_alternaria_blight":
        # Dark brown-black concentric target-board spots with yellow halo
        n_spots = int(rng.integers(5, 13))
        for _ in range(n_spots):
            su = rng.uniform(-0.65, 0.65)
            sv = rng.uniform(-0.50, 0.50)
            rad = rng.uniform(0.08, 0.17)
            dist = np.sqrt((u - su) ** 2 + (v - sv) ** 2)
            halo = mask & (dist > rad) & (dist <= rad * 1.48)
            spot = mask & (dist <= rad)
            ring_wave = 0.5 + 0.5 * np.cos((dist / (rad + 1e-6)) * 5.0 * math.pi)
            leaf[halo] = np.array([0.84, 0.78, 0.18]) + rng.normal(0, 0.03, (int(halo.sum()), 3))
            if int(spot.sum()) > 0:
                dark_base = np.array([0.16, 0.11, 0.08])[None, :]
                ring_col = np.array([0.34, 0.23, 0.14])[None, :]
                leaf[spot] = dark_base * (1.0 - 0.55 * ring_wave[spot, None]) + ring_col * (0.55 * ring_wave[spot, None])
            lesion_mask |= spot | halo

    elif cls_name == "mustard_white_rust":
        # Chalky white raised blisters on green lamina with mild yellow halo
        n_clusters = int(rng.integers(3, 7))
        for _ in range(n_clusters):
            cu = rng.uniform(-0.55, 0.55)
            cv = rng.uniform(-0.45, 0.45)
            n_blisters = int(rng.integers(5, 11))
            for _ in range(n_blisters):
                su = cu + rng.normal(0.0, 0.09)
                sv = cv + rng.normal(0.0, 0.09)
                rad = rng.uniform(0.032, 0.068)
                dist = np.sqrt((u - su) ** 2 + (v - sv) ** 2)
                halo = mask & (dist > rad) & (dist <= rad * 1.55)
                pust = mask & (dist <= rad)
                leaf[halo] = leaf[halo] * 0.4 + np.array([0.74, 0.78, 0.26]) * 0.6
                leaf[pust] = np.array([0.90, 0.90, 0.87]) + rng.normal(0, 0.025, (int(pust.sum()), 3))
                lesion_mask |= pust

    elif cls_name == "chickpea_ascochyta_blight":
        # Pale straw/tan lesions with dark brown borders & fine pycnidia dots
        n_spots = int(rng.integers(6, 13))
        for _ in range(n_spots):
            su = rng.uniform(-0.68, 0.68)
            sv = rng.uniform(-0.34, 0.34)
            rad = rng.uniform(0.075, 0.14)
            dist = np.sqrt((u - su) ** 2 + (v - sv) ** 2)
            border = mask & (dist >= rad * 0.74) & (dist <= rad)
            tan_inner = mask & (dist < rad * 0.74)
            pycnidia = tan_inner & (np.sin(dist * 110.0) > 0.78) & (rng.random((SIZE, SIZE)) > 0.35)
            leaf[tan_inner] = np.array([0.72, 0.62, 0.44]) + rng.normal(0, 0.03, (int(tan_inner.sum()), 3))
            leaf[border] = np.array([0.28, 0.17, 0.10])
            leaf[pycnidia] = np.array([0.12, 0.09, 0.07])
            lesion_mask |= border | tan_inner

    elif cls_name == "chickpea_fusarium_wilt":
        # Whole-branch dull yellowing / drooping straw senescence
        wilt_severity = rng.uniform(0.55, 0.95)
        wilt_grad = np.clip((u + 0.9) * 0.6 + rng.uniform(-0.15, 0.25), 0.15, 1.0) * wilt_severity
        wilt_col = np.array([0.76, 0.70, 0.28]) if rng.random() > 0.35 else np.array([0.69, 0.62, 0.36])
        w_f = np.clip(wilt_grad, 0.0, 1.0)[:, :, None]
        leaf = leaf * (1.0 - w_f) + wilt_col[None, None, :] * w_f
        lesion_mask = mask & (wilt_grad > 0.35)

    elif cls_name == "potato_early_blight":
        # Dark brown angular/circular bullseye concentric ring spots mostly interior
        n_spots = int(rng.integers(5, 12))
        for _ in range(n_spots):
            su = rng.uniform(-0.45, 0.62)
            sv = rng.uniform(-0.36, 0.36)
            rad = rng.uniform(0.08, 0.155)
            dist = np.sqrt(((u - su) * 1.1) ** 2 + (v - sv) ** 2)
            halo = mask & (dist > rad) & (dist <= rad * 1.45)
            spot = mask & (dist <= rad)
            rings = 0.5 + 0.5 * np.sin((dist / (rad + 1e-6)) * 6.0 * math.pi)
            leaf[halo] = np.array([0.82, 0.78, 0.20])
            if int(spot.sum()) > 0:
                c1 = np.array([0.18, 0.12, 0.08])[None, :]
                c2 = np.array([0.39, 0.25, 0.14])[None, :]
                leaf[spot] = c1 * (1.0 - 0.55 * rings[spot, None]) + c2 * (0.55 * rings[spot, None])
            lesion_mask |= spot | halo

    elif cls_name == "potato_late_blight":
        # Water-soaked dark olive-brown/purplish blotches advancing from leaf MARGINS & TIPS
        n_lobes = int(rng.integers(2, 5))
        margin_zone = mask & (dist_edge < rng.uniform(0.32, 0.55))
        blotch = np.zeros((SIZE, SIZE), dtype=bool)
        for _ in range(n_lobes):
            # Center foci specifically along leaf perimeter
            ang = rng.uniform(-math.pi, math.pi)
            cu = 0.2 + 0.52 * math.cos(ang)
            cv = 0.42 * math.sin(ang)
            rad = rng.uniform(0.24, 0.44)
            noise_edge = rng.normal(0.0, 0.04, (SIZE, SIZE))
            d = np.sqrt((u - cu) ** 2 + (v - cv) ** 2) + noise_edge
            blotch |= margin_zone & (d <= rad)
        if int(blotch.sum()) < int(0.10 * mask.sum()):
            blotch |= mask & (dist_edge < 0.28) & (u > 0.1)
        # Pale wet border
        halo = mask & (~blotch) & (dist_edge < 0.48)
        for _ in range(1):
            # Subtle olive fade next to blotch
            fade = np.clip(0.55 - dist_edge, 0.0, 1.0) * halo
            fade_3 = fade[:, :, None]
            leaf = leaf * (1.0 - 0.45 * fade_3) + np.array([0.52, 0.56, 0.22])[None, None, :] * (0.45 * fade_3)
        leaf[blotch] = np.array([0.19, 0.17, 0.11]) + rng.normal(0, 0.03, (int(blotch.sum()), 3))
        lesion_mask |= blotch

    # Apply natural lighting + composite leaf softly onto field backdrop
    leaf = np.clip(leaf * light[:, :, None], 0.03, 0.98)
    # Soft 1px edge blend using neighbor smoothing on mask
    edge_blend = np.clip(
        0.5 * mask_f
        + 0.125 * (np.roll(mask_f, 1, 0) + np.roll(mask_f, -1, 0) + np.roll(mask_f, 1, 1) + np.roll(mask_f, -1, 1)),
        0.0,
        1.0,
    )[:, :, None]
    composite = bg * (1.0 - edge_blend) + leaf * edge_blend
    # Slight smartphone camera sensor grain
    composite = np.clip(composite + rng.normal(0.0, 0.012, (SIZE, SIZE, 3)), 0.0, 1.0)
    rgb_uint8 = (composite * 255.0).round().astype(np.uint8)

    true_sev = (float((mask & lesion_mask).sum()) / max(1, int(mask.sum()))) * 100.0
    return rgb_uint8, round(true_sev, 2)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    raw_dir = os.path.join(root_dir, "data", "raw_images")
    sample_dir = os.path.join(root_dir, "dataset_samples")
    os.makedirs(sample_dir, exist_ok=True)

    n_per_class = 120
    rows = []
    class_counts: Dict[str, int] = {}

    print(f"Generating RabiLeaf-1440 dataset ({len(CLASSES)} classes x {n_per_class} images = {len(CLASSES)*n_per_class})...")
    global_idx = 0
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = os.path.join(raw_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        class_counts[cls_name] = 0

        for i in range(n_per_class):
            seed = 10000 + cls_idx * 500 + i
            rgb_u8, true_sev = synthesize_leaf_image(cls_name, seed)
            fname = f"{cls_name}_{i:03d}.png"
            fpath = os.path.join(cls_dir, fname)
            Image.fromarray(rgb_u8).save(fpath, compress_level=6)

            # Save 2 curated samples per class into dataset_samples/ for documentation & app preview
            if i < 2:
                Image.fromarray(rgb_u8).save(os.path.join(sample_dir, f"{cls_name}_sample_{i+1}.png"), compress_level=6)

            diag = extract_features_and_diagnostics(rgb_u8)
            row = {
                "sample_id": global_idx,
                "filename": os.path.join(cls_name, fname),
                "crop": cls_name.split("_")[0],
                "label": cls_name,
                "true_severity_pct": true_sev,
                "extracted_severity_pct": diag["estimated_severity_pct"],
                "leaf_pixel_ratio": round(diag["leaf_pixel_ratio"], 4),
            }
            for fname_col, val in zip(FEATURE_NAMES, diag["features"]):
                row[fname_col] = round(float(val), 6)
            rows.append(row)
            class_counts[cls_name] += 1
            global_idx += 1

    df = pd.DataFrame(rows)
    csv_out = os.path.join(root_dir, "data", "crop_features_dataset.csv")
    df.to_csv(csv_out, index=False)

    summary = {
        "dataset_name": "RabiLeaf-1440 Physics-Informed Botanical Benchmark",
        "total_images": len(df),
        "image_size": f"{SIZE}x{SIZE} RGB",
        "num_classes": len(CLASSES),
        "classes": class_counts,
        "num_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
    }
    summary_path = os.path.join(root_dir, "data", "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {len(df)} feature rows -> {csv_out}")
    print(f"Saved 24 curated sample images -> {sample_dir}")
    print(f"Saved dataset summary -> {summary_path}")


if __name__ == "__main__":
    main()
