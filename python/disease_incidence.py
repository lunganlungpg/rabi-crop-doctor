"""
disease_incidence.py
--------------------
Epidemiological calculator for Indian Rabi Crop Disease Incidence (%)
and Wheeler's Percent Disease Index (PDI) with agro-climatic ETL alerts.
"""

from __future__ import annotations

import os
from typing import Dict, List
import pandas as pd


def severity_to_grade(severity_pct: float) -> int:
    """Convert foliar lesion area percentage (0-100) into standard 0-5 pathology scale."""
    if severity_pct <= 2.0:
        return 0
    if severity_pct <= 10.0:
        return 1
    if severity_pct <= 25.0:
        return 2
    if severity_pct <= 50.0:
        return 3
    if severity_pct <= 75.0:
        return 4
    return 5


def compute_field_survey_metrics(observations: List[Dict[str, object]], max_scale: int = 5) -> Dict[str, object]:
    """
    Calculate Plot Disease Incidence (%) and Wheeler (1969) Percent Disease Index (PDI).

    Parameters
    ----------
    observations : list of dict
        Each dict contains keys: 'is_healthy' (bool), 'severity_pct' (float), 'disease' (str).
    max_scale : int
        Maximum rating scale (default = 5).
    """
    n_total = len(observations)
    if n_total == 0:
        return {
            "total_inspected": 0,
            "infected_count": 0,
            "disease_incidence_pct": 0.0,
            "pdi_pct": 0.0,
            "mean_severity_pct": 0.0,
            "risk_level": "No Data",
            "disease_counts": {},
        }

    infected = 0
    rating_sum = 0
    severity_sum = 0.0
    disease_counts: Dict[str, int] = {}

    for obs in observations:
        is_healthy = bool(obs.get("is_healthy", False))
        sev = float(obs.get("severity_pct", 0.0))
        disease = str(obs.get("disease", "Unknown"))
        disease_counts[disease] = disease_counts.get(disease, 0) + 1

        if not is_healthy:
            infected += 1
            grade = severity_to_grade(max(sev, 2.5))
        else:
            grade = 0
            sev = 0.0

        rating_sum += grade
        severity_sum += sev

    incidence_pct = (infected / n_total) * 100.0
    pdi_pct = (rating_sum / (n_total * max_scale)) * 100.0
    mean_sev = severity_sum / n_total

    if incidence_pct == 0.0:
        risk_level = "Healthy / Safe"
    elif incidence_pct < 10.0 and pdi_pct < 8.0:
        risk_level = "Low Watch (Monitor Borders)"
    elif incidence_pct < 25.0:
        risk_level = "Moderate / Near ETL — Verify with KVK"
    else:
        risk_level = "High Alert / Exceeds ETL — Prompt Action Needed"

    return {
        "total_inspected": n_total,
        "infected_count": infected,
        "disease_incidence_pct": round(incidence_pct, 2),
        "pdi_pct": round(pdi_pct, 2),
        "mean_severity_pct": round(mean_sev, 2),
        "risk_level": risk_level,
        "disease_counts": disease_counts,
    }


def summarize_india_rabi_incidence(csv_path: str | None = None) -> pd.DataFrame:
    """Load regional Indian Rabi disease incidence survey data and flag critical hotspots."""
    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "..", "data", "rabi_field_incidence_india.csv")
    df = pd.read_csv(csv_path)
    df["risk_index"] = (df["disease_incidence_pct"] * 0.6 + df["mean_pdi_pct"] * 0.4).round(2)
    return df.sort_values("risk_index", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    summary = summarize_india_rabi_incidence()
    print("=== India Rabi Disease Incidence Hotspot Summary ===")
    print(summary[["state", "crop", "disease", "peak_month", "disease_incidence_pct", "mean_pdi_pct", "etl_status"]].to_string(index=False))

    sample_plot = [
        {"is_healthy": True, "severity_pct": 1.2, "disease": "wheat_healthy"},
        {"is_healthy": True, "severity_pct": 0.8, "disease": "wheat_healthy"},
        {"is_healthy": False, "severity_pct": 18.5, "disease": "wheat_yellow_rust"},
        {"is_healthy": False, "severity_pct": 32.0, "disease": "wheat_yellow_rust"},
        {"is_healthy": True, "severity_pct": 1.5, "disease": "wheat_healthy"},
    ]
    print("\n=== Sample 5-Leaf Field Audit ===")
    print(compute_field_survey_metrics(sample_plot))
