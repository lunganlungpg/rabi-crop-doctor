"""
generate_plots.py
-----------------
Generates publication-ready (300 DPI) diagnostic figures for the
Rabi Crop Disease Incidence & ML vs ANN Picture Classification project.
"""

from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    from .generate_rabi_dataset import CLASSES
except ImportError:
    from generate_rabi_dataset import CLASSES


SHORT_LABELS = {
    "wheat_healthy": "Wheat\nHealthy",
    "wheat_yellow_rust": "Wheat\nYell. Rust",
    "wheat_brown_rust": "Wheat\nBrown Rust",
    "mustard_healthy": "Mustard\nHealthy",
    "mustard_alternaria_blight": "Mustard\nAlternaria",
    "mustard_white_rust": "Mustard\nWhite Rust",
    "chickpea_healthy": "Chickpea\nHealthy",
    "chickpea_ascochyta_blight": "Chickpea\nAscochyta",
    "chickpea_fusarium_wilt": "Chickpea\nFus. Wilt",
    "potato_healthy": "Potato\nHealthy",
    "potato_early_blight": "Potato\nEarly Blight",
    "potato_late_blight": "Potato\nLate Blight",
}


def make_dataset_grid(root_dir: str, plots_dir: str) -> None:
    sample_dir = os.path.join(root_dir, "dataset_samples")
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.0), dpi=300)
    fig.patch.set_facecolor("#F8FAF8")

    for idx, cls_name in enumerate(CLASSES):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        img_path = os.path.join(sample_dir, f"{cls_name}_sample_1.png")
        if os.path.exists(img_path):
            ax.imshow(Image.open(img_path))
        title = SHORT_LABELS.get(cls_name, cls_name).replace("\n", " — ")
        is_h = cls_name.endswith("_healthy")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#1B5E20" if is_h else "#B71C1C", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#4CAF50" if is_h else "#EF9A9A")
            spine.set_linewidth(1.5)

    fig.suptitle(
        "RabiLeaf-1440 Representative Samples: Indian Rabi Crop Conditions (12 Classes)",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(plots_dir, "dataset_samples_grid.png")
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out}")


def make_model_comparison(metrics: dict, plots_dir: str) -> None:
    models_dict = metrics["models"]
    order = ["random_forest", "svm_rbf", "ann_mlp", "ann_deep", "hybrid_ensemble"]
    labels = ["Random Forest\n(ML)", "RBF SVM\n(ML)", "MLP ANN\n(64-32)", "Deep ANN\n(128-64-32)", "Hybrid\nEnsemble"]

    test_acc = [models_dict[k]["test_accuracy"] * 100 for k in order]
    macro_f1 = [models_dict[k]["test_f1_macro"] * 100 for k in order]
    crop_acc = [models_dict[k]["crop_filtered_accuracy"] * 100 for k in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300, gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.patch.set_facecolor("#FFFFFF")

    x = np.arange(len(order))
    w = 0.25
    b1 = ax1.bar(x - w, test_acc, w, label="12-Class Test Accuracy (%)", color="#1B5E20")
    b2 = ax1.bar(x, macro_f1, w, label="Macro F1-Score (%)", color="#0288D1")
    b3 = ax1.bar(x + w, crop_acc, w, label="Farmer Crop-Mode Acc (%)", color="#F59E0B")

    for bars in (b1, b2, b3):
        for bar in bars:
            h_val = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h_val + 0.6, f"{h_val:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 124)
    ax1.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax1.set_title("ML vs. ANN Classification Benchmark (Held-Out Test Set)", fontsize=11, fontweight="bold", pad=10)
    ax1.legend(loc="upper center", ncol=3, frameon=True, fontsize=7.8)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Loss curve for both ANN models
    for key, col, lbl in [
        ("ann_mlp", "#0288D1", "MLP ANN (64-32) Training Loss"),
        ("ann_deep", "#7B1FA2", "Deep ANN (128-64-32) Training Loss"),
    ]:
        curve = models_dict[key].get("loss_curve", [])
        if curve:
            ax2.plot(range(1, len(curve) + 1), curve, color=col, linewidth=2.2, label=lbl)

    ax2.set_xlabel("Training Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=10, fontweight="bold")
    ax2.set_title("ANN Optimization Convergence Curves", fontsize=11, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plots_dir, "ml_vs_ann_comparison.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def make_confusion_matrices(metrics: dict, plots_dir: str) -> None:
    classes = metrics["dataset"]["classes"]
    short = [SHORT_LABELS.get(c, c).replace("\n", " ") for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=300)
    pairs = [
        ("random_forest", "ML Model: Random Forest (12-Class Confusion)"),
        ("ann_deep", "ANN Model: Deep Regularized MLP (12-Class Confusion)"),
    ]

    for ax, (key, title) in zip(axes, pairs):
        cm = np.array(metrics["models"][key]["confusion_matrix"], dtype=np.float64)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(cm_norm, cmap="YlGn", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("Predicted Condition", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("True Field Condition", fontsize=9.5, fontweight="bold")
        acc = metrics["models"][key]["test_accuracy"] * 100
        ax.set_title(f"{title}\nTest Acc: {acc:.2f}%", fontsize=10.5, fontweight="bold", pad=8)

        for i in range(len(classes)):
            for j in range(len(classes)):
                val = int(cm[i, j])
                if val > 0:
                    txt_col = "white" if cm_norm[i, j] > 0.55 else "#0F172A"
                    ax.text(j, i, str(val), ha="center", va="center", color=txt_col, fontsize=7.5, fontweight="bold")

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.03, label="Class Recall Proportion")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.join(plots_dir, 'confusion_matrix_comparison.png')}")


def make_feature_and_pathology_plot(metrics: dict, df: pd.DataFrame, plots_dir: str) -> None:
    imp_dict = metrics["models"]["random_forest"].get("feature_importances", {})
    sorted_imp = sorted(imp_dict.items(), key=lambda kv: kv[1], reverse=True)[:14][::-1]
    f_names = [kv[0] for kv in sorted_imp]
    f_vals = [kv[1] for kv in sorted_imp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.8), dpi=300)
    colors = []
    for n in f_names:
        if any(k in n for k in ["coherence", "edge", "spot", "dog", "patch", "bias"]):
            colors.append("#0288D1")
        elif any(k in n for k in ["elongation", "perimeter"]):
            colors.append("#7B1FA2")
        else:
            colors.append("#2E7D32")

    ax1.barh(f_names, f_vals, color=colors, edgecolor="none")
    ax1.set_xlabel("Random Forest Gini Importance", fontsize=10, fontweight="bold")
    ax1.set_title("Top 14 Plant Pathology Image Features", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Domain discrimination: Linear Stripe Coherence (Yellow vs Brown Rust) & Margin Bias (Late vs Early Blight)
    pairs = [
        ("wheat_yellow_rust", "Wheat Yellow Rust\n(Stripe Coherence)", "stripe_coherence", "#F59E0B"),
        ("wheat_brown_rust", "Wheat Brown Rust\n(Stripe Coherence)", "stripe_coherence", "#B45309"),
        ("potato_late_blight", "Potato Late Blight\n(Margin Bias)", "radial_margin_disease_bias", "#1E293B"),
        ("potato_early_blight", "Potato Early Blight\n(Margin Bias)", "radial_margin_disease_bias", "#64748B"),
    ]
    data_lists = [df.loc[df["label"] == lbl, col].values for lbl, _, col, _ in pairs]
    bp = ax2.boxplot(data_lists, patch_artist=True, tick_labels=[t for _, t, _, _ in pairs], widths=0.52)
    for box, (_, _, _, col) in zip(bp["boxes"], pairs):
        box.set_facecolor(col)
        box.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("white")
        med.set_linewidth(2)

    ax2.axhline(0.0, color="#94A3B8", linestyle=":", linewidth=1.2)
    ax2.set_ylabel("Feature Signal Value", fontsize=10, fontweight="bold")
    ax2.set_title("Pathology Geometry Separation (Texture & Margin Physics)", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plots_dir, "feature_importance_and_pathology.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def make_rabi_incidence_plot(root_dir: str, plots_dir: str) -> None:
    inc_df = pd.read_csv(os.path.join(root_dir, "data", "rabi_field_incidence_india.csv"))
    inc_df["tag"] = inc_df["state"] + " — " + inc_df["crop"] + " (" + inc_df["disease"] + ")"
    inc_df = inc_df.sort_values("disease_incidence_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    y = np.arange(len(inc_df))
    h = 0.38

    ax.barh(y + h / 2, inc_df["disease_incidence_pct"], height=h, color="#D97706", label="Disease Incidence (%)")
    ax.barh(y - h / 2, inc_df["mean_pdi_pct"], height=h, color="#15803D", label="Wheeler PDI Severity (%)")
    ax.axvline(10.0, color="#DC2626", linestyle="--", linewidth=1.3, label="General ETL Alert Line (10%)")

    ax.set_yticks(y)
    ax.set_yticklabels(inc_df["tag"], fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Percentage (%)", fontsize=10, fontweight="bold")
    ax.set_title("Indian Rabi Belt Surveillance: Regional Disease Incidence & Severity (PDI)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plots_dir, "india_rabi_incidence_heatmap.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    plots_dir = os.path.join(root_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(root_dir, "models", "evaluation_metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    df = pd.read_csv(os.path.join(root_dir, "data", "crop_features_dataset.csv"))

    make_dataset_grid(root_dir, plots_dir)
    make_model_comparison(metrics, plots_dir)
    make_confusion_matrices(metrics, plots_dir)
    make_feature_and_pathology_plot(metrics, df, plots_dir)
    make_rabi_incidence_plot(root_dir, plots_dir)
    print("All 300 DPI diagnostic figures generated successfully.")


if __name__ == "__main__":
    main()
