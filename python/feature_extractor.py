"""
feature_extractor.py
--------------------
Deterministic 28-dimensional plant pathology computer vision feature extractor
and foreground leaf segmentation engine for 128x128 RGB leaf images.

NOTE: Every calculation here mirrors web/js/feature_extractor.js to ensure
zero training-serving skew between Python model training and offline Android APK
inference.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Union
import numpy as np
from PIL import Image

IMG_SIZE = 128

FEATURE_NAMES: List[str] = [
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    "exg_mean",
    "exg_std",
    "exr_mean",
    "gli_mean",
    "s_mean",
    "v_mean",
    "healthy_green_frac",
    "yellow_chlorosis_frac",
    "rust_orange_brown_frac",
    "white_pustule_frac",
    "dark_necrotic_frac",
    "tan_straw_frac",
    "hue_dispersion",
    "edge_lesion_energy",
    "stripe_coherence",
    "high_freq_spot_ratio",
    "radial_margin_disease_bias",
    "patch_heterogeneity",
    "leaf_elongation",
    "perimeter_area_ratio",
    "rg_diff_std",
    "dog_ring_energy",
]


def _box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """Clamped border-aware box average over window [y-r..y+r, x-r..x+r]."""
    h, w = arr.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(arr.astype(np.float64), axis=0), axis=1)

    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.maximum(0, ys - radius)[:, None]
    y1 = np.minimum(h - 1, ys + radius)[:, None] + 1
    x0 = np.maximum(0, xs - radius)[None, :]
    x1 = np.minimum(w - 1, xs + radius)[None, :] + 1

    sums = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    counts = (y1 - y0) * (x1 - x0)
    return sums / counts


def load_rgb_128(image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """Load an image path, PIL Image, or uint8/float array into a (128, 128, 3) float64 array in [0, 1]."""
    if isinstance(image_input, str):
        with Image.open(image_input) as img:
            img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float64) / 255.0
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float64) / 255.0
    else:
        arr = np.asarray(image_input, dtype=np.float64)
        if arr.shape[:2] != (IMG_SIZE, IMG_SIZE):
            pil_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
            pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
            arr = np.asarray(pil_img, dtype=np.float64) / 255.0
        elif arr.max() > 1.0:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def extract_features_and_diagnostics(
    image_input: Union[str, Image.Image, np.ndarray]
) -> Dict[str, object]:
    """
    Extract the 28-dimensional leaf pathology feature vector and diagnostic
    segmentation stats (leaf detection, lesion severity %, health grade).
    """
    rgb = load_rgb_128(image_input)
    h, w, _ = rgb.shape
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    v = max_c
    s = np.where(max_c > 1e-6, delta / np.maximum(max_c, 1e-6), 0.0)

    h_deg = np.zeros((h, w), dtype=np.float64)
    nonzero = delta > 1e-6
    d_safe = np.where(nonzero, delta, 1.0)

    is_r = nonzero & (max_c == r)
    is_g = nonzero & (~is_r) & (max_c == g)
    is_b = nonzero & (~is_r) & (~is_g)

    h_deg[is_r] = 60.0 * (((g[is_r] - b[is_r]) / d_safe[is_r]) % 6.0)
    h_deg[is_g] = 60.0 * (((b[is_g] - r[is_g]) / d_safe[is_g]) + 2.0)
    h_deg[is_b] = 60.0 * (((r[is_b] - g[is_b]) / d_safe[is_b]) + 4.0)

    exg = 2.0 * g - r - b
    exr = 1.4 * r - g
    gli = (2.0 * g - r - b) / (2.0 * g + r + b + 1e-4)
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Step 1: Leaf segmentation
    veg_seed = (gli > 0.015) | (
        (h_deg >= 30.0) & (h_deg <= 165.0) & (s > 0.13) & (v > 0.11)
    )
    veg_density = _box_blur(veg_seed.astype(np.float64), radius=5)

    is_sky_blue = (h_deg >= 175.0) & (h_deg <= 265.0) & (s > 0.15)
    is_outer_glare = (v > 0.96) & (s < 0.06) & (veg_density < 0.30)
    leaf_mask = (veg_seed | (veg_density >= 0.20)) & (~is_sky_blue) & (v >= 0.05) & (~is_outer_glare)

    used_fallback = False
    if int(leaf_mask.sum()) < 650:
        used_fallback = True
        yy, xx = np.mgrid[0:h, 0:w]
        ell = (((xx - 63.5) / 48.0) ** 2 + ((yy - 63.5) / 48.0) ** 2) <= 1.0
        leaf_mask = ell & (~is_sky_blue) & (v >= 0.06) & ~((v > 0.95) & (s < 0.07))

    n_leaf = int(leaf_mask.sum())
    if n_leaf < 60:
        return {
            "leaf_detected": False,
            "leaf_pixel_ratio": float(n_leaf) / float(h * w),
            "used_fallback_roi": used_fallback,
            "estimated_severity_pct": 0.0,
            "severity_grade": 0,
            "features": np.zeros(len(FEATURE_NAMES), dtype=np.float64),
            "feature_dict": {k: 0.0 for k in FEATURE_NAMES},
        }

    # Pixel pathology flags inside leaf
    is_healthy_px = (h_deg >= 68.0) & (h_deg <= 165.0) & (s >= 0.16) & (v >= 0.16) & (exg > 0.03)
    is_yellow_px = (h_deg >= 38.0) & (h_deg < 68.0) & (s >= 0.20) & (v >= 0.25)
    is_rust_px = ((h_deg >= 8.0) & (h_deg < 38.0) | (h_deg >= 345.0)) & (s >= 0.20) & (v >= 0.16) & (v <= 0.88)
    is_white_px = (v >= 0.72) & (s <= 0.25) & (((r + g + b) / 3.0) >= 0.65)
    is_dark_px = (v <= 0.28) & (exg < 0.05)
    is_tan_px = (h_deg >= 25.0) & (h_deg < 54.0) & (s >= 0.11) & (s <= 0.38) & (v >= 0.40) & (v <= 0.85) & (~is_healthy_px)

    r_l = r[leaf_mask]
    g_l = g[leaf_mask]
    b_l = b[leaf_mask]
    exg_l = exg[leaf_mask]
    exr_l = exr[leaf_mask]
    gli_l = gli[leaf_mask]
    s_l = s[leaf_mask]
    v_l = v[leaf_mask]

    r_mean = float(np.mean(r_l))
    g_mean = float(np.mean(g_l))
    b_mean = float(np.mean(b_l))
    r_std = float(np.std(r_l))
    g_std = float(np.std(g_l))
    b_std = float(np.std(b_l))
    exg_mean = float(np.mean(exg_l))
    exg_std = float(np.std(exg_l))
    exr_mean = float(np.mean(exr_l))
    gli_mean = float(np.mean(gli_l))
    s_mean = float(np.mean(s_l))
    v_mean = float(np.mean(v_l))

    healthy_green_frac = float((leaf_mask & is_healthy_px).sum()) / n_leaf
    yellow_chlorosis_frac = float((leaf_mask & is_yellow_px).sum()) / n_leaf
    rust_orange_brown_frac = float((leaf_mask & is_rust_px).sum()) / n_leaf
    white_pustule_frac = float((leaf_mask & is_white_px).sum()) / n_leaf
    dark_necrotic_frac = float((leaf_mask & is_dark_px).sum()) / n_leaf
    tan_straw_frac = float((leaf_mask & is_tan_px).sum()) / n_leaf

    # Circular Hue dispersion on saturated leaf pixels
    sat_mask = leaf_mask & (s >= 0.15)
    if int(sat_mask.sum()) > 10:
        rad = np.deg2rad(h_deg[sat_mask])
        c_mean = float(np.mean(np.cos(rad)))
        s_sin = float(np.mean(np.sin(rad)))
        hue_dispersion = 1.0 - math.sqrt(c_mean * c_mean + s_sin * s_sin)
    else:
        hue_dispersion = 0.0

    # Clamped central differences for gradient & structure tensor
    lum_pad = np.pad(lum, ((1, 1), (1, 1)), mode="edge")
    gx = 0.5 * (lum_pad[1:-1, 2:] - lum_pad[1:-1, :-2])
    gy = 0.5 * (lum_pad[2:, 1:-1] - lum_pad[:-2, 1:-1])

    grad_mag = 0.5 * (np.abs(gx) + np.abs(gy))
    edge_lesion_energy = float(np.mean(grad_mag[leaf_mask]))

    jxx = _box_blur(gx * gx, radius=3)
    jyy = _box_blur(gy * gy, radius=3)
    jxy = _box_blur(gx * gy, radius=3)
    trace = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2)) / (trace + 1e-6)
    valid_t = leaf_mask & (trace > 1e-4)
    if int(valid_t.sum()) > 0:
        weights = trace[valid_t]
        stripe_coherence = float(np.sum(coh[valid_t] * weights) / (np.sum(weights) + 1e-9))
    else:
        stripe_coherence = 0.0

    # High frequency spot deviation
    exg_blur5 = _box_blur(exg, radius=2)
    high_freq_spot_ratio = float((leaf_mask & (np.abs(exg - exg_blur5) > 0.07)).sum()) / n_leaf

    # Radial margin vs core lesion bias
    mask_density_r3 = _box_blur(leaf_mask.astype(np.float64), radius=3)
    is_lesion_px = leaf_mask & (~is_healthy_px)
    is_margin = leaf_mask & (mask_density_r3 < 0.88)
    is_core = leaf_mask & (mask_density_r3 >= 0.88)
    n_margin = int(is_margin.sum())
    n_core = int(is_core.sum())
    margin_frac = (float((is_margin & is_lesion_px).sum()) / n_margin) if n_margin > 0 else 0.0
    core_frac = (float((is_core & is_lesion_px).sum()) / n_core) if n_core > 0 else 0.0
    radial_margin_disease_bias = margin_frac - core_frac

    # 4x4 grid block heterogeneity of ExG
    block_means = []
    for by in range(4):
        for bx in range(4):
            y0, y1 = by * 32, (by + 1) * 32
            x0, x1 = bx * 32, (bx + 1) * 32
            bm = leaf_mask[y0:y1, x0:x1]
            if int(bm.sum()) >= 16:
                block_means.append(float(np.mean(exg[y0:y1, x0:x1][bm])))
    patch_heterogeneity = float(np.std(block_means)) if len(block_means) >= 2 else 0.0

    # PCA Elongation on (y, x) coordinates of leaf_mask
    ys_l, xs_l = np.where(leaf_mask)
    ys_f = ys_l.astype(np.float64)
    xs_f = xs_l.astype(np.float64)
    dx = xs_f - np.mean(xs_f)
    dy = ys_f - np.mean(ys_f)
    cxx = float(np.mean(dx * dx))
    cyy = float(np.mean(dy * dy))
    cxy = float(np.mean(dx * dy))
    tr_cov = cxx + cyy
    disc_cov = math.sqrt((cxx - cyy) ** 2 + 4.0 * cxy * cxy)
    lam1 = 0.5 * (tr_cov + disc_cov)
    lam2 = max(0.0, 0.5 * (tr_cov - disc_cov))
    leaf_elongation = 1.0 - (lam2 / (lam1 + 1e-6))

    # 4-connected perimeter / sqrt(area)
    up = np.zeros_like(leaf_mask)
    up[1:, :] = leaf_mask[:-1, :]
    dn = np.zeros_like(leaf_mask)
    dn[:-1, :] = leaf_mask[1:, :]
    lf = np.zeros_like(leaf_mask)
    lf[:, 1:] = leaf_mask[:, :-1]
    rt = np.zeros_like(leaf_mask)
    rt[:, :-1] = leaf_mask[:, 1:]
    interior_4 = leaf_mask & up & dn & lf & rt
    perimeter_px = int((leaf_mask & (~interior_4)).sum())
    perimeter_area_ratio = float(perimeter_px) / (math.sqrt(float(n_leaf)) + 1e-6)

    # Color spread R - G
    rg_diff_std = float(np.std(r_l - g_l))

    # Concentric ring DoG energy over lesion pixels
    lum_b1 = _box_blur(lum, radius=1)
    lum_b4 = _box_blur(lum, radius=4)
    dog = np.abs(lum_b1 - lum_b4)
    n_lesion = int(is_lesion_px.sum())
    dog_ring_energy = float(np.mean(dog[is_lesion_px])) if n_lesion > 12 else float(np.mean(dog[leaf_mask]))

    features = np.array(
        [
            r_mean,
            g_mean,
            b_mean,
            r_std,
            g_std,
            b_std,
            exg_mean,
            exg_std,
            exr_mean,
            gli_mean,
            s_mean,
            v_mean,
            healthy_green_frac,
            yellow_chlorosis_frac,
            rust_orange_brown_frac,
            white_pustule_frac,
            dark_necrotic_frac,
            tan_straw_frac,
            hue_dispersion,
            edge_lesion_energy,
            stripe_coherence,
            high_freq_spot_ratio,
            radial_margin_disease_bias,
            patch_heterogeneity,
            leaf_elongation,
            perimeter_area_ratio,
            rg_diff_std,
            dog_ring_energy,
        ],
        dtype=np.float64,
    )

    # Estimated severity % (infected tissue over foreground leaf)
    raw_lesion_pct = 100.0 * (1.0 - healthy_green_frac)
    est_severity = round(max(0.0, min(100.0, raw_lesion_pct)), 1)
    if est_severity <= 2.0:
        grade = 0
    elif est_severity <= 10.0:
        grade = 1
    elif est_severity <= 25.0:
        grade = 2
    elif est_severity <= 50.0:
        grade = 3
    elif est_severity <= 75.0:
        grade = 4
    else:
        grade = 5

    return {
        "leaf_detected": True,
        "leaf_pixel_ratio": float(n_leaf) / float(h * w),
        "used_fallback_roi": used_fallback,
        "estimated_severity_pct": est_severity,
        "severity_grade": grade,
        "features": features,
        "feature_dict": {k: float(v_i) for k, v_i in zip(FEATURE_NAMES, features)},
    }
