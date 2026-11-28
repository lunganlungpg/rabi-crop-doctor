"""
rabi_timeline.py
----------------
Resolves which Indian Rabi crops and disease scouting priorities are active
for a given date, roadmap quarter, or month based on:
  - ROADMAP.md (Nov 2026 – Nov 2027 quarterly milestones Q1–Q4)
  - data/rabi_crops_knowledge.json (crop sowing/harvest windows & disease peaks)
  - data/rabi_field_incidence_india.csv (regional monthly incidence alerts)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

try:
    from ._paths import resolve_asset_path
except ImportError:
    from _paths import resolve_asset_path


MONTH_ALIASES: dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

MONTH_NAMES: dict[int, str] = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

CORE_RABI_CROPS: tuple[str, ...] = ("wheat", "mustard", "chickpea", "potato")
PLANNED_Q3_RABI_CROPS: tuple[str, ...] = ("barley", "lentil", "field_pea")
ALL_RABI_CROPS: tuple[str, ...] = CORE_RABI_CROPS + PLANNED_Q3_RABI_CROPS

ROADMAP_QUARTER_SPECS: dict[str, dict[str, Any]] = {
    "Q1": {
        "months_label": "Dec 2026 – Feb 2027 (incl. Nov 2026 sowing kickoff)",
        "seasonal_focus": "Active Rabi Field Season",
        "milestone": "v1.1 — Field Feedback & Dataset Growth",
        "crops": list(CORE_RABI_CROPS),
    },
    "Q2": {
        "months_label": "Mar – May 2027",
        "seasonal_focus": "Harvest & Post-Harvest Review",
        "milestone": "v1.2 — Real-Photo Model Fine-Tuning",
        "crops": list(CORE_RABI_CROPS),
    },
    "Q3": {
        "months_label": "Jun – Aug 2027",
        "seasonal_focus": "Monsoon Build & Localization",
        "milestone": "v1.3 — Multi-Language & Expanded Rabi Crops",
        "crops": list(ALL_RABI_CROPS),
    },
    "Q4": {
        "months_label": "Sep – Nov 2027",
        "seasonal_focus": "Pre-Sowing Pilot & Open Science",
        "milestone": "v2.0 — Next-Season Production Release",
        "crops": list(ALL_RABI_CROPS),
    },
}


def _parse_month_range(spec: str) -> set[int]:
    """Expand a month or month-range expression like 'Dec–Feb' or 'Jan–Mar' into 1..12."""
    cleaned = spec.replace("—", "-").replace("–", "-").strip()
    months: set[int] = set()
    for part in cleaned.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = [x.strip().lower() for x in token.split("-", 1)]
            start_m = MONTH_ALIASES.get(start_raw)
            end_m = MONTH_ALIASES.get(end_raw)
            if start_m is None or end_m is None:
                continue
            cur = start_m
            while True:
                months.add(cur)
                if cur == end_m:
                    break
                cur = (cur % 12) + 1
        else:
            m = MONTH_ALIASES.get(token.lower())
            if m is not None:
                months.add(m)
    return months


def _parse_crop_season_months(season_str: str) -> set[int]:
    """Extract active sowing-to-harvest month span from e.g. 'Sowing: Nov | Harvest: Mar–Apr'."""
    sowing_months: list[int] = []
    harvest_months: list[int] = []
    for segment in season_str.split("|"):
        seg = segment.strip()
        if ":" in seg:
            key, val = seg.split(":", 1)
            parsed = sorted(_parse_month_range(val))
            if "sowing" in key.lower():
                sowing_months.extend(parsed)
            elif "harvest" in key.lower():
                harvest_months.extend(parsed)
    if sowing_months and harvest_months:
        active: set[int] = set()
        cur = sowing_months[0]
        target = harvest_months[-1]
        for _ in range(12):
            active.add(cur)
            if cur == target:
                break
            cur = (cur % 12) + 1
        return active
    return set(sowing_months) | set(harvest_months)


def _quarter_for_date(target_date: dt.date) -> str:
    """Map a calendar date to the project's Nov 2026 – Nov 2027 ROADMAP.md quarter."""
    m = target_date.month
    if m in (12, 1, 2) or (target_date.year <= 2026 and m == 11):
        return "Q1"
    if m in (3, 4, 5):
        return "Q2"
    if m in (6, 7, 8):
        return "Q3"
    return "Q4"


def _load_knowledge_and_incidence() -> tuple[dict[str, Any], list[dict[str, str]]]:
    kb_path = Path(resolve_asset_path("data", "rabi_crops_knowledge.json"))
    inc_path = Path(resolve_asset_path("data", "rabi_field_incidence_india.csv"))
    kb = json.loads(kb_path.read_text(encoding="utf-8"))
    with inc_path.open("r", encoding="utf-8", newline="") as handle:
        incidence_rows = list(csv.DictReader(handle))
    return kb, incidence_rows


def resolve_rabi_timeline(
    as_of: str | dt.date | None = None,
    mode: str = "roadmap",
    quarter: str | None = None,
    month: int | None = None,
) -> dict[str, Any]:
    """
    Resolve active Rabi crops and disease priorities from the project timeline.

    Parameters
    ----------
    as_of : str | date | None
        Reference ISO date (`YYYY-MM-DD`). Defaults to today's date.
    mode : str
        - ``"roadmap"``: crops from ROADMAP.md milestones (Q1/Q2 -> core 4, Q3/Q4 -> all 7).
        - ``"peak-month"``: crops with active disease peaks in the target month
          (falling back to the quarter's crops outside Nov–Mar).
        - ``"all-rabi"``: all 7 Rabi crops listed across the full roadmap.
    quarter : str | None
        Optional explicit roadmap quarter (``Q1``..``Q4`` or ``auto``).
    month : int | None
        Optional explicit month (1..12) override.
    """
    if as_of is None:
        ref_date = dt.date.today()
    elif isinstance(as_of, dt.date):
        ref_date = as_of
    else:
        ref_date = dt.date.fromisoformat(str(as_of).strip())

    if month is not None:
        if not 1 <= month <= 12:
            raise ValueError(f"month must be between 1 and 12, got {month}")
        ref_month = int(month)
        ref_date = ref_date.replace(month=ref_month, day=1)
    else:
        ref_month = ref_date.month

    if quarter and quarter.lower() != "auto":
        q_key = quarter.upper().strip()
        if q_key not in ROADMAP_QUARTER_SPECS:
            raise ValueError(f"Unknown roadmap quarter {quarter!r}; expected Q1..Q4 or auto")
    else:
        q_key = _quarter_for_date(ref_date)

    q_spec = ROADMAP_QUARTER_SPECS[q_key]
    kb, incidence_rows = _load_knowledge_and_incidence()

    crop_details: dict[str, dict[str, Any]] = {}
    for entry in kb.get("crops", []):
        cid = str(entry["id"]).lower()
        crop_details[cid] = {
            "id": cid,
            "name_en": entry.get("name_en", cid.title()),
            "scientific": entry.get("scientific", ""),
            "season": entry.get("season", ""),
            "active_months": sorted(_parse_crop_season_months(entry.get("season", ""))),
            "major_states": entry.get("major_states", []),
        }
    for extra_id, label in [
        ("barley", "Barley"),
        ("lentil", "Lentil (Masoor)"),
        ("field_pea", "Field Pea (Matar)"),
    ]:
        crop_details.setdefault(
            extra_id,
            {
                "id": extra_id,
                "name_en": label,
                "scientific": "",
                "season": "Planned in ROADMAP.md Q3 (v1.3)",
                "active_months": [11, 12, 1, 2, 3],
                "major_states": [],
            },
        )

    active_diseases: list[dict[str, Any]] = []
    peak_crops_set: set[str] = set()
    for cls_entry in kb.get("classes", []):
        if cls_entry.get("is_healthy", False):
            continue
        peak_set = _parse_month_range(cls_entry.get("peak_months", ""))
        if ref_month in peak_set:
            crop_id = str(cls_entry["crop"]).lower()
            peak_crops_set.add(crop_id)
            active_diseases.append({
                "id": cls_entry["id"],
                "crop": crop_id,
                "name_en": cls_entry.get("name_en", cls_entry["id"]),
                "peak_months": cls_entry.get("peak_months", ""),
                "etl_pct": cls_entry.get("etl_pct", 0),
            })

    month_abbr = MONTH_NAMES[ref_month]
    incidence_hotspots: list[dict[str, Any]] = []
    for row in incidence_rows:
        if row.get("peak_month", "").strip().lower() == month_abbr.lower():
            crop_id = row.get("crop", "").strip().lower()
            peak_crops_set.add(crop_id)
            incidence_hotspots.append({
                "state": row["state"],
                "crop": crop_id,
                "disease": row["disease"],
                "disease_incidence_pct": float(row["disease_incidence_pct"]),
                "mean_pdi_pct": float(row["mean_pdi_pct"]),
                "etl_status": row["etl_status"],
            })

    mode_norm = mode.strip().lower()
    if mode_norm == "roadmap":
        selected_crops = list(q_spec["crops"])
    elif mode_norm == "peak-month":
        ordered = [c for c in ALL_RABI_CROPS if c in peak_crops_set]
        selected_crops = ordered if ordered else list(q_spec["crops"])
    elif mode_norm == "all-rabi":
        selected_crops = list(ALL_RABI_CROPS)
    else:
        raise ValueError(f"Unsupported timeline mode {mode!r}; expected roadmap, peak-month, or all-rabi")

    return {
        "as_of": ref_date.isoformat(),
        "month": ref_month,
        "month_name": month_abbr,
        "quarter": q_key,
        "months_label": q_spec["months_label"],
        "seasonal_focus": q_spec["seasonal_focus"],
        "milestone": q_spec["milestone"],
        "mode": mode_norm,
        "crops": selected_crops,
        "core_rabi_crops": list(CORE_RABI_CROPS),
        "planned_q3_rabi_crops": list(PLANNED_Q3_RABI_CROPS),
        "active_diseases_in_month": active_diseases,
        "incidence_hotspots_in_month": incidence_hotspots,
        "crop_details": {c: crop_details[c] for c in selected_crops if c in crop_details},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve active Rabi crops from the Rabi Crop Doctor roadmap and seasonal timeline."
    )
    parser.add_argument(
        "--mode",
        choices=["roadmap", "peak-month", "all-rabi"],
        default="roadmap",
        help="Timeline selection mode (default: roadmap).",
    )
    parser.add_argument(
        "--quarter",
        default="auto",
        help="Roadmap quarter override: Q1, Q2, Q3, Q4, or auto (default: auto).",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Reference date YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Optional month override (1-12).",
    )
    parser.add_argument(
        "--crops-only",
        action="store_true",
        help="Print space-separated selected crop identifiers only.",
    )
    args = parser.parse_args()

    resolved = resolve_rabi_timeline(
        as_of=args.as_of,
        mode=args.mode,
        quarter=args.quarter,
        month=args.month,
    )
    if args.crops_only:
        print(" ".join(resolved["crops"]))
    else:
        print(json.dumps(resolved, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
