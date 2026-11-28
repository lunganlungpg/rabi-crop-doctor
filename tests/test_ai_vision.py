"""
test_ai_vision.py
-----------------
Unit tests for Python (`python/ai_vision.py`, `python/predict_image.py`) and
JavaScript (`web/js/ai_vision.js`) multi-model Gemini Vision integration,
consensus aggregation, schema-retry fallback, live model filtering, and
offline ML+ANN fallback. Uses mocked HTTP/fetch handlers (no live API key needed).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "python"))

from ai_vision import analyze_leaf_multi_model, list_available_models
from predict_image import predict_leaf_image


def test_python_ai_multi_model_and_fallback() -> None:
    sample_img = str(Path(ROOT_DIR) / "kaggle_samples" / "wheat_yellow_rust.jpg")

    def fake_http(url: str, api_key: str, body: dict | None = None):
        assert api_key == "test-key"
        if url.endswith("/models?pageSize=100"):
            return 200, {
                "models": [
                    {
                        "name": "models/gemini-2.5-flash",
                        "displayName": "Gemini 2.5 Flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemini-2.5-pro",
                        "displayName": "Gemini 2.5 Pro",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/text-embedding-004",
                        "displayName": "Embedding",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                ]
            }

        # Simulate schema rejection (400) on gemini-2.5-pro first call, succeeding without schema
        if "gemini-2.5-pro" in url and "responseSchema" in body["generationConfig"]:
            return 400, {"error": {"message": "Schema unsupported"}}

        if "gemini-3-flash-preview" in url:
            return 429, {"error": {"message": "Rate limit exceeded"}}

        payload = {
            "is_plant_leaf": True,
            "image_quality_note": "Clear daylight wheat leaf",
            "detected_crop": "wheat",
            "canonical_class_id": "wheat_yellow_rust",
            "is_healthy": False,
            "diagnosis_en": "Wheat Yellow (Stripe) Rust",
            "diagnosis_hi": "गेहूँ का पीला रतुआ",
            "pathogen": "Puccinia striiformis",
            "confidence_pct": 94.0 if "pro" in url else 90.0,
            "estimated_severity_pct": 42.0,
            "severity_grade": 3,
            "symptoms_en": "Yellow-orange powdery stripes along veins.",
            "symptoms_hi": "नसों के समानांतर पीली धारियां।",
            "reasoning_en": "Parallel linear yellow pustule rows match Puccinia striiformis.",
            "action_en": ["Consult nearest KVK for focal spot spray."],
            "action_hi": ["नजदीकी KVK से सलाह लें।"],
        }
        return 200, {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}

    listed = list_available_models("test-key", http_fn=fake_http)
    assert [m["id"] for m in listed] == ["gemini-2.5-flash", "gemini-2.5-pro"]

    outcome = analyze_leaf_multi_model(
        image_path=sample_img,
        api_key="test-key",
        models=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview"],
        crop_hint="wheat",
        http_fn=fake_http,
    )
    cons = outcome["consensus"]
    assert cons is not None
    assert cons["canonical_class_id"] == "wheat_yellow_rust"
    assert cons["agreement_count"] == 2
    assert cons["total_models_succeeded"] == 2
    assert cons["total_models_requested"] == 3
    assert cons["confidence_pct"] == 92.0
    assert outcome["results"][2]["ok"] is False

    # Verify automatic offline fallback when AI fails / no API key is present
    prev_env = os.environ.pop("GEMINI_API_KEY", None)
    try:
        fallback = predict_leaf_image(sample_img, crop_filter="wheat", use_ai=True, api_key="")
        assert fallback["source"] == "offline_ml_ann"
        assert fallback["predicted_class"] == "wheat_yellow_rust"
        assert "ai_fallback_reason" in fallback
    finally:
        if prev_env is not None:
            os.environ["GEMINI_API_KEY"] = prev_env
    print("1. Python AI multi-model + offline fallback tests PASSED")


def test_js_ai_vision_module() -> None:
    node_script = r"""
const ai = require(process.argv[1]);

const storageMap = new Map();
const fakeStorage = {
  getItem: (k) => storageMap.get(k) || null,
  setItem: (k, v) => storageMap.set(k, v),
};

ai.saveConfig({
  enabled: true,
  apiKey: "AIzaFakeKey123",
  rememberKey: true,
  selectedModels: ["gemini-2.5-flash", "gemini-2.5-pro"],
  customModels: ["models/gemini-3.8-flash"],
}, fakeStorage);

const loaded = ai.loadConfig(fakeStorage);
if (loaded.apiKey !== "AIzaFakeKey123" || loaded.selectedModels.length !== 2) {
  throw new Error("Config roundtrip failed");
}
const cleared = ai.clearConfigKey(fakeStorage);
if (cleared.apiKey !== "") {
  throw new Error("clearConfigKey failed");
}

async function run() {
  const fakeFetch = async (url, opts) => {
    if (url.includes("/models?")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          models: [
            { name: "models/gemini-2.5-flash", displayName: "Gemini 2.5 Flash", supportedGenerationMethods: ["generateContent"] },
            { name: "models/gemini-2.5-pro", displayName: "Gemini 2.5 Pro", supportedGenerationMethods: ["generateContent"] },
            { name: "models/gemini-2.5-flash-tts", displayName: "TTS", supportedGenerationMethods: ["generateContent"] },
          ],
        }),
      };
    }
    const body = JSON.parse(opts.body);
    if (url.includes("gemini-2.5-pro") && body.generationConfig.responseSchema) {
      return { ok: false, status: 400, json: async () => ({ error: { message: "No schema" } }) };
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        candidates: [{
          content: {
            parts: [{
              text: "```json\n" + JSON.stringify({
                is_plant_leaf: true,
                detected_crop: "wheat",
                canonical_class_id: "wheat_yellow_rust",
                is_healthy: false,
                diagnosis_en: "Wheat Yellow (Stripe) Rust",
                diagnosis_hi: "गेहूँ का पीला रतुआ",
                pathogen: "Puccinia striiformis",
                confidence_pct: url.includes("pro") ? 95 : 89,
                estimated_severity_pct: 35,
                severity_grade: 3,
                symptoms_en: "Yellow linear stripes",
                symptoms_hi: "पीली धारियां",
                reasoning_en: "Stripe pustules along leaf blade",
                action_en: ["Contact KVK"],
                action_hi: ["KVK से संपर्क करें"],
              }) + "\n```"
            }]
          }
        }]
      }),
    };
  };

  const listed = await ai.listAvailableModels("AIzaFakeKey123", fakeFetch);
  if (listed.length !== 2) throw new Error("Expected 2 filtered Gemini models, got " + listed.length);

  const res = await ai.analyzeLeafMultiModel({
    apiKey: "AIzaFakeKey123",
    models: ["gemini-2.5-flash", "gemini-2.5-pro"],
    dataUrl: "data:image/jpeg;base64,AAAA",
    cropHint: "wheat",
    knownClasses: ["wheat_healthy", "wheat_yellow_rust"],
    fetchFn: fakeFetch,
  });
  if (!res.consensus || res.consensus.agreementCount !== 2 || res.consensus.confidencePct !== 92) {
    throw new Error("Unexpected JS consensus: " + JSON.stringify(res));
  }
  console.log("2. JS ai_vision.js multi-model + schema retry tests PASSED");
}

run().catch((e) => { console.error(e); process.exit(1); });
"""
    js_path = os.path.join(ROOT_DIR, "web", "js", "ai_vision.js")
    subprocess.run(["node", "-e", node_script, js_path], check=True)


if __name__ == "__main__":
    test_python_ai_multi_model_and_fallback()
    test_js_ai_vision_module()
    print("\nALL GEMINI MULTI-MODEL AI VISION TESTS PASSED!")
