"""
generate_interactive_plots.py
-----------------------------
Builds standalone interactive Plotly HTML dashboards for exploring:
1) ML vs ANN crop disease classification metrics, ANN loss convergence, and feature importance
2) Regional Indian Rabi crop disease incidence vs agro-climatic temperature/humidity triggers
"""

from __future__ import annotations

import json
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_model_dashboard(metrics: dict, out_path: str) -> None:
    models = metrics["models"]
    keys = ["random_forest", "svm_rbf", "ann_mlp", "ann_deep", "hybrid_ensemble"]
    names = [models[k]["display_name"] for k in keys]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "12-Class Accuracy vs Farmer Crop-Mode Accuracy (%)",
            "ANN Training Loss Curve (MLP vs Deep ANN)",
            "Hybrid Ensemble Confusion Matrix (Held-Out Test)",
            "Top 12 Pathological Features (Random Forest Importance)",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.14,
    )

    fig.add_trace(
        go.Bar(
            name="12-Class Test Acc (%)",
            x=names,
            y=[models[k]["test_accuracy"] * 100 for k in keys],
            marker_color="#1B5E20",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Crop-Mode Farmer Acc (%)",
            x=names,
            y=[models[k]["crop_filtered_accuracy"] * 100 for k in keys],
            marker_color="#F59E0B",
        ),
        row=1,
        col=1,
    )

    for k, col in [("ann_mlp", "#0288D1"), ("ann_deep", "#7B1FA2")]:
        curve = models[k].get("loss_curve", [])
        if curve:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(curve) + 1)),
                    y=curve,
                    mode="lines",
                    name=models[k]["display_name"],
                    line=dict(color=col, width=2.5),
                ),
                row=1,
                col=2,
            )

    classes = metrics["dataset"]["classes"]
    cm = models["hybrid_ensemble"]["confusion_matrix"]
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Greens",
            showscale=False,
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    imp = sorted(models["random_forest"]["feature_importances"].items(), key=lambda kv: kv[1], reverse=True)[:12][::-1]
    fig.add_trace(
        go.Bar(
            x=[v for _, v in imp],
            y=[k for k, _ in imp],
            orientation="h",
            name="Feature Importance",
            marker_color="#00897B",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Rabi Crop Doctor: Interactive ML vs. ANN Diagnostic Explorer",
        height=820,
        template="plotly_white",
        barmode="group",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved {out_path}")


def build_incidence_dashboard(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Regional Rabi Disease Incidence (%) & PDI Severity (%)",
            "Agro-Climatic Trigger Space (Temp vs Humidity, Bubble = Incidence %)",
        ),
        horizontal_spacing=0.12,
    )

    labels = df["state"] + " | " + df["crop"] + " (" + df["disease"] + ")"
    fig.add_trace(
        go.Bar(x=labels, y=df["disease_incidence_pct"], name="Incidence (%)", marker_color="#D97706"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=df["mean_pdi_pct"], name="PDI Severity (%)", marker_color="#15803D"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["mean_temp_c"],
            y=df["mean_rh_pct"],
            mode="markers+text",
            text=df["crop"] + " (" + df["state"].str[:2] + ")",
            textposition="top center",
            marker=dict(
                size=df["disease_incidence_pct"] * 1.15,
                color=df["mean_pdi_pct"],
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="PDI %"),
            ),
            customdata=df[["state", "crop", "disease", "peak_month", "disease_incidence_pct", "etl_status"]],
            hovertemplate=(
                "<b>%{customdata[0]} — %{customdata[1]}</b><br>"
                "Disease: %{customdata[2]} (Peak: %{customdata[3]})<br>"
                "Incidence: %{customdata[4]}% | Status: %{customdata[5]}<extra></extra>"
            ),
            name="Hotspots",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Relative Humidity (%)", row=1, col=2)
    fig.update_layout(
        title="India Rabi Crop Disease Incidence & Agro-Climatic Risk Explorer",
        height=600,
        template="plotly_white",
        barmode="group",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved {out_path}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    plots_dir = os.path.join(root_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(root_dir, "models", "evaluation_metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)

    build_model_dashboard(metrics, os.path.join(plots_dir, "interactive_model_explorer.html"))
    build_incidence_dashboard(
        os.path.join(root_dir, "data", "rabi_field_incidence_india.csv"),
        os.path.join(plots_dir, "interactive_rabi_incidence.html"),
    )


if __name__ == "__main__":
    main()
