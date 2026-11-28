"""
ai_vision.py
------------
Google Gemini Vision multi-model leaf verification, severity grading, and
bilingual (English / Hindi) diagnostic engine for Rabi Crop Doctor CLI.

Uses Python standard library HTTP (`urllib.request`) + `concurrent.futures`
with zero external package requirements. API keys are supplied by the user via
`GEMINI_API_KEY` environment variable or `--api-key`.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_AI_MODELS: tuple[str, ...] = ("gemini-3.5-flash-lite",)
KNOWN_DEFAULT_MODELS: tuple[dict[str, str], ...] = (
    {"id": "gemini-3.5-flash-lite", "label": "Gemini 3.5 Flash-Lite (Fast Free-Tier Default)"},
    {"id": "gemini-flash-lite-latest", "label": "Gemini Flash-Lite Latest"},
    {"id": "gemini-3.8-flash", "label": "Gemini 3.8 Flash"},
    {"id": "gemini-flash-latest", "label": "Gemini Flash Latest"},
    {"id": "gemini-pro-latest", "label": "Gemini Pro Latest"},
    {"id": "gemini-4-argon", "label": "Gemini 4 Argon (Frontier)"},
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "is_plant_leaf": {"type": "BOOLEAN"},
        "image_quality_note": {"type": "STRING"},
        "detected_crop": {"type": "STRING"},
        "canonical_class_id": {"type": "STRING"},
        "is_healthy": {"type": "BOOLEAN"},
        "diagnosis_en": {"type": "STRING"},
        "diagnosis_hi": {"type": "STRING"},
        "pathogen": {"type": "STRING"},
        "confidence_pct": {"type": "NUMBER"},
        "estimated_severity_pct": {"type": "NUMBER"},
        "severity_grade": {"type": "INTEGER"},
        "symptoms_en": {"type": "STRING"},
        "symptoms_hi": {"type": "STRING"},
        "reasoning_en": {"type": "STRING"},
        "action_en": {"type": "ARRAY", "items": {"type": "STRING"}},
        "action_hi": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": [
        "is_plant_leaf",
        "detected_crop",
        "canonical_class_id",
        "is_healthy",
        "diagnosis_en",
        "diagnosis_hi",
        "pathogen",
        "confidence_pct",
        "estimated_severity_pct",
        "severity_grade",
        "symptoms_en",
        "symptoms_hi",
        "action_en",
        "action_hi",
    ],
}


def normalize_model_id(model_id: str) -> str:
    return str(model_id or "").strip().removeprefix("models/")


def severity_to_grade(severity_pct: float) -> int:
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


def build_prompt(crop_hint: str | None, known_classes: list[str] | None = None) -> str:
    hint = (
        f'The user indicated the crop is likely "{crop_hint}". Confirm from the image; if clearly another crop, report the true crop.'
        if crop_hint and crop_hint.lower().strip() != "all"
        else "Auto-detect the crop species from the image."
    )
    base_classes = list(known_classes) if known_classes else [
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
    extra_classes = [
        "wheat_black_rust",
        "wheat_leaf_blight",
        "wheat_powdery_mildew",
        "wheat_scab",
        "wheat_flag_smut",
        "wheat_aphid",
        "wheat_mite",
        "wheat_stem_fly",
    ]
    all_ids = ", ".join(dict.fromkeys([*base_classes, *extra_classes]))
    return "\n".join([
        "You are an expert Indian ICAR / Krishi Vigyan Kendra (KVK) plant pathologist.",
        "Inspect the attached image carefully and respond ONLY with valid JSON matching the requested schema.",
        hint,
        "Diagnose the true disease, pest, nutrient stress, or healthy state strictly from the visual evidence in the photo—never force-fit an unrelated condition into a rust or blight label.",
        "Distinguish carefully between foliar fungal symptoms (linear yellow rust stripes vs scattered brown rust pustules vs black stem rust vs white powdery mildew vs tan leaf blight), head/spike diseases (wheat scab / Fusarium head blight showing premature bleached/straw spikelets vs dark smutted ears), and insect/mite pests (aphids, mites, stem fly).",
        f'For canonical_class_id, output a lowercase snake_case "<crop>_<condition>" id (matching [{all_ids}] when applicable, or "<crop>_<condition>" for any other condition).',
        "Rules:",
        "- Close-up/macro photos of leaf surfaces, pustules, stems, or grain ears ARE valid crop photos (keep is_plant_leaf=true even if cropped or slightly low-res). Set is_plant_leaf=false ONLY for completely non-plant subjects (e.g., blank wall, face, vehicle, text document).",
        "- severity_grade must follow the 0-5 botanical scale: 0 (<=2% healthy/trace), 1 (<=10%), 2 (<=25%), 3 (<=50%), 4 (<=75%), 5 (>75%). If healthy, set estimated_severity_pct=0 and severity_grade=0.",
        "- Provide concise, accurate symptoms and actionable ICAR/KVK-aligned IPM steps in BOTH English (*_en) and Hindi (*_hi).",
        "- Advise confirming chemical spray dosages with the nearest KVK or Kisan Call Centre (1800-180-1551) when chemical control is mentioned.",
    ])


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty response from Gemini API.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fenced:
            return json.loads(fenced.group(1).strip())
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            return json.loads(text[first_brace : last_brace + 1])
        raise ValueError("Model output was not valid JSON.")


def normalize_model_result(raw_obj: dict[str, Any], model_id: str, latency_ms: int) -> dict[str, Any]:
    is_leaf = raw_obj.get("is_plant_leaf", True) is not False
    is_healthy = bool(raw_obj.get("is_healthy", False))
    try:
        raw_sev = float(raw_obj.get("estimated_severity_pct", 15.0))
    except (TypeError, ValueError):
        raw_sev = 15.0
    sev_pct = 0.0 if (is_healthy or not is_leaf) else round(max(0.0, min(100.0, raw_sev)), 1)

    try:
        raw_grade = int(raw_obj.get("severity_grade", severity_to_grade(sev_pct)))
    except (TypeError, ValueError):
        raw_grade = severity_to_grade(sev_pct)
    grade = 0 if (is_healthy or not is_leaf) else max(0, min(5, raw_grade))

    try:
        raw_conf = float(raw_obj.get("confidence_pct", 75.0))
    except (TypeError, ValueError):
        raw_conf = 75.0
    conf = round(max(0.0, min(100.0, raw_conf)), 1)

    canon = re.sub(r"[^a-z0-9]+", "_", str(raw_obj.get("canonical_class_id", "unknown_condition")).lower()).strip("_")
    canon = canon or "unknown_condition"

    action_en = raw_obj.get("action_en")
    if not isinstance(action_en, list) or not action_en:
        action_en = ["Verify symptoms with your local KVK or Kisan Call Centre (1800-180-1551)."]
    action_hi = raw_obj.get("action_hi")
    if not isinstance(action_hi, list) or not action_hi:
        action_hi = ["सलाह की पुष्टि के लिए नजदीकी KVK या किसान कॉल सेंटर (1800-180-1551) से संपर्क करें।"]

    return {
        "model_id": normalize_model_id(model_id),
        "ok": True,
        "latency_ms": latency_ms,
        "is_plant_leaf": is_leaf,
        "image_quality_note": str(raw_obj.get("image_quality_note", "")),
        "detected_crop": str(raw_obj.get("detected_crop", "unknown")).lower(),
        "canonical_class_id": canon,
        "is_healthy": is_healthy,
        "diagnosis_en": str(raw_obj.get("diagnosis_en", canon)),
        "diagnosis_hi": str(raw_obj.get("diagnosis_hi", raw_obj.get("diagnosis_en", canon))),
        "pathogen": str(raw_obj.get("pathogen", "None" if is_healthy else "Consult local KVK for lab confirmation")),
        "confidence_pct": conf,
        "estimated_severity_pct": sev_pct,
        "severity_grade": grade,
        "symptoms_en": str(raw_obj.get("symptoms_en", "")),
        "symptoms_hi": str(raw_obj.get("symptoms_hi", raw_obj.get("symptoms_en", ""))),
        "reasoning_en": str(raw_obj.get("reasoning_en", "")),
        "action_en": [str(x) for x in action_en],
        "action_hi": [str(x) for x in action_hi],
    }


def build_consensus(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    successes = [r for r in results if r.get("ok")]
    if not successes:
        return None

    buckets: dict[str, list[dict[str, Any]]] = {}
    for res in successes:
        key = res["canonical_class_id"] or res["diagnosis_en"].lower()
        buckets.setdefault(key, []).append(res)

    ranked = sorted(
        buckets.values(),
        key=lambda items: (len(items), sum(max(1.0, x["confidence_pct"]) for x in items)),
        reverse=True,
    )
    winner_items = ranked[0]
    primary = max(winner_items, key=lambda x: x["confidence_pct"])
    avg_conf = round(sum(x["confidence_pct"] for x in winner_items) / len(winner_items), 1)
    avg_sev = round(sum(x["estimated_severity_pct"] for x in winner_items) / len(winner_items), 1)

    merged = dict(primary)
    merged["confidence_pct"] = avg_conf
    merged["estimated_severity_pct"] = 0.0 if (primary["is_healthy"] or not primary["is_plant_leaf"]) else avg_sev
    merged["severity_grade"] = 0 if (primary["is_healthy"] or not primary["is_plant_leaf"]) else severity_to_grade(avg_sev)
    merged["agreement_count"] = len(winner_items)
    merged["total_models_succeeded"] = len(successes)
    merged["total_models_requested"] = len(results)
    merged["all_model_results"] = results
    return merged


def _http_json(
    url: str,
    api_key: str,
    body: dict[str, Any] | None = None,
    timeout: int = 45,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"x-goog-api-key": api_key}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": {"message": raw or f"HTTP {err.code}"}}
        return err.code, parsed


def list_available_models(api_key: str, http_fn: Any = None) -> list[dict[str, str]]:
    clean_key = (api_key or "").strip()
    if not clean_key:
        raise ValueError("Gemini API key is required.")
    caller = http_fn or _http_json
    status, payload = caller(f"{GEMINI_BASE}/models?pageSize=100", clean_key, None)
    if status >= 400:
        raise RuntimeError(payload.get("error", {}).get("message", f"HTTP {status}"))
    models: list[dict[str, str]] = []
    for entry in payload.get("models", []):
        name = str(entry.get("name", ""))
        methods = entry.get("supportedGenerationMethods", [])
        if (
            "gemini" in name
            and "generateContent" in methods
            and "embedding" not in name
            and "tts" not in name
            and "-image" not in name
            and "nano-banana" not in name
            and "transcribe" not in name
            and "robotics" not in name
            and "computer-use" not in name
            and "customtools" not in name
            and not name.startswith("models/gemini-2.")
        ):
            mid = normalize_model_id(name)
            display = entry.get("displayName")
            models.append({
                "id": mid,
                "label": f"{display} ({mid})" if display else mid,
            })
    return models


def analyze_with_model(
    *,
    api_key: str,
    model_id: str,
    image_path: str,
    crop_hint: str | None = None,
    known_classes: list[str] | None = None,
    http_fn: Any = None,
) -> dict[str, Any]:
    clean_model = normalize_model_id(model_id)
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    prompt = build_prompt(crop_hint, known_classes)
    caller = http_fn or _http_json
    url = f"{GEMINI_BASE}/models/{urllib.parse.quote(clean_model, safe='')}:generateContent"

    def _payload(use_schema: bool) -> dict[str, Any]:
        gen_cfg: dict[str, Any] = {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        }
        if use_schema:
            gen_cfg["responseSchema"] = RESPONSE_SCHEMA
        return {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": b64}},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": gen_cfg,
        }

    t0 = time.perf_counter()
    status, resp = caller(url, api_key, _payload(True))
    if status == 400:
        status, resp = caller(url, api_key, _payload(False))
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if status >= 400:
        return {
            "model_id": clean_model,
            "ok": False,
            "error": resp.get("error", {}).get("message", f"HTTP {status}"),
            "latency_ms": elapsed_ms,
        }

    candidates = resp.get("candidates") or []
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    text_out = "\n".join(str(p.get("text", "")) for p in parts).strip()
    parsed = extract_json_payload(text_out)
    return normalize_model_result(parsed, clean_model, elapsed_ms)


def analyze_leaf_multi_model(
    *,
    image_path: str,
    api_key: str | None = None,
    models: list[str] | None = None,
    crop_hint: str | None = None,
    known_classes: list[str] | None = None,
    http_fn: Any = None,
) -> dict[str, Any]:
    clean_key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not clean_key:
        raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY or pass --api-key.")
    raw_list = [normalize_model_id(m) for m in (models or DEFAULT_AI_MODELS) if normalize_model_id(m)]
    model_list = list(dict.fromkeys(raw_list)) or list(DEFAULT_AI_MODELS)

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(model_list))) as pool:
        future_map = {
            pool.submit(
                analyze_with_model,
                api_key=clean_key,
                model_id=m,
                image_path=image_path,
                crop_hint=crop_hint,
                known_classes=known_classes,
                http_fn=http_fn,
            ): m
            for m in model_list
        }
        # Preserve input model ordering in final output
        ordered: dict[str, dict[str, Any]] = {}
        for fut, m in future_map.items():
            try:
                ordered[m] = fut.result()
            except Exception as exc:
                ordered[m] = {"model_id": m, "ok": False, "error": str(exc), "latency_ms": 0}
        results = [ordered[m] for m in model_list]

    return {
        "consensus": build_consensus(results),
        "results": results,
    }
