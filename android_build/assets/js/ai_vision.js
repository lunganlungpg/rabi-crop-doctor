/**
 * ai_vision.js
 * ------------
 * Browser & Android WebView client for Google Gemini Vision multi-model leaf
 * disease verification, severity estimation, and bilingual (EN/HI) advisory.
 *
 * Security: API keys are supplied by the user on their own device and stored
 * only in browser/WebView localStorage. Nothing is hardcoded or transmitted
 * anywhere except Google's official Generative Language API endpoint.
 */
(function (root) {
  "use strict";

  const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta";
  const STORAGE_CFG_KEY = "rabi_crop_doctor_ai_config_v1";

  const DEFAULT_MODELS = [
    {
      id: "gemini-3.5-flash-lite",
      label: "Gemini 3.5 Flash-Lite (Fast Free-Tier Default)",
      tier: "lite",
    },
    {
      id: "gemini-flash-lite-latest",
      label: "Gemini Flash-Lite Latest",
      tier: "lite",
    },
    {
      id: "gemini-3.8-flash",
      label: "Gemini 3.8 Flash (Standard)",
      tier: "flash",
    },
    {
      id: "gemini-flash-latest",
      label: "Gemini Flash Latest",
      tier: "flash",
    },
    {
      id: "gemini-pro-latest",
      label: "Gemini Pro Latest (Deep Reasoning)",
      tier: "pro",
    },
    {
      id: "gemini-4-argon",
      label: "Gemini 4 Argon (Frontier / Paid Tier)",
      tier: "pro",
    },
  ];

  const RESPONSE_SCHEMA = {
    type: "OBJECT",
    properties: {
      is_plant_leaf: { type: "BOOLEAN" },
      image_quality_note: { type: "STRING" },
      detected_crop: { type: "STRING" },
      canonical_class_id: { type: "STRING" },
      is_healthy: { type: "BOOLEAN" },
      diagnosis_en: { type: "STRING" },
      diagnosis_hi: { type: "STRING" },
      pathogen: { type: "STRING" },
      confidence_pct: { type: "NUMBER" },
      estimated_severity_pct: { type: "NUMBER" },
      severity_grade: { type: "INTEGER" },
      symptoms_en: { type: "STRING" },
      symptoms_hi: { type: "STRING" },
      reasoning_en: { type: "STRING" },
      action_en: { type: "ARRAY", items: { type: "STRING" } },
      action_hi: { type: "ARRAY", items: { type: "STRING" } },
    },
    required: [
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
  };

  function defaultConfig() {
    return {
      enabled: true,
      apiKey: "",
      rememberKey: true,
      selectedModels: ["gemini-3.5-flash-lite"],
      customModels: [],
      discoveredModels: [],
    };
  }

  function loadConfig(storage) {
    const store = storage || (typeof localStorage !== "undefined" ? localStorage : null);
    const base = defaultConfig();
    if (!store) return base;
    try {
      const raw = store.getItem(STORAGE_CFG_KEY);
      if (!raw) return base;
      const parsed = JSON.parse(raw);
      const rawSelected =
        Array.isArray(parsed.selectedModels) && parsed.selectedModels.length > 0
          ? parsed.selectedModels.map(String)
          : base.selectedModels;
      // Migrate retired gemini-2.5-flash default cleanly
      const selected = rawSelected.map((m) => (m === "gemini-2.5-flash" ? "gemini-3.5-flash-lite" : m));
      return {
        enabled: parsed.enabled !== false,
        apiKey: typeof parsed.apiKey === "string" ? parsed.apiKey : "",
        rememberKey: parsed.rememberKey !== false,
        selectedModels: Array.from(new Set(selected)),
        customModels: Array.isArray(parsed.customModels) ? parsed.customModels.map(String) : [],
        discoveredModels: Array.isArray(parsed.discoveredModels) ? parsed.discoveredModels : [],
      };
    } catch {
      return base;
    }
  }

  function saveConfig(cfg, storage) {
    const store = storage || (typeof localStorage !== "undefined" ? localStorage : null);
    if (!store) return;
    try {
      const toPersist = {
        enabled: cfg.enabled !== false,
        apiKey: cfg.rememberKey !== false ? String(cfg.apiKey || "").trim() : "",
        rememberKey: cfg.rememberKey !== false,
        selectedModels: Array.isArray(cfg.selectedModels) ? cfg.selectedModels : ["gemini-3.5-flash-lite"],
        customModels: Array.isArray(cfg.customModels) ? cfg.customModels : [],
        discoveredModels: Array.isArray(cfg.discoveredModels) ? cfg.discoveredModels : [],
      };
      store.setItem(STORAGE_CFG_KEY, JSON.stringify(toPersist));
    } catch {}
  }

  function clearConfigKey(storage) {
    const cfg = loadConfig(storage);
    cfg.apiKey = "";
    saveConfig(cfg, storage);
    return cfg;
  }

  function normalizeModelId(modelId) {
    return String(modelId || "").trim().replace(/^models\//, "");
  }

  function buildPrompt(cropHint, knownClasses) {
    const hintText =
      cropHint && cropHint !== "all"
        ? `The user indicated the crop is likely "${cropHint}". Confirm from the image; if clearly another crop, report the true crop.`
        : "Auto-detect the crop species from the image.";
    const extraClasses = [
      "wheat_black_rust",
      "wheat_leaf_blight",
      "wheat_powdery_mildew",
      "wheat_scab",
      "wheat_flag_smut",
      "wheat_aphid",
      "wheat_mite",
      "wheat_stem_fly",
    ];
    const baseIds = Array.isArray(knownClasses)
      ? knownClasses.map((c) => c.id || c)
      : [
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
        ];
    const allIds = Array.from(new Set([...baseIds, ...extraClasses])).join(", ");

    return [
      "You are an expert Indian ICAR / Krishi Vigyan Kendra (KVK) plant pathologist.",
      "Inspect the attached image carefully and respond ONLY with valid JSON matching the requested schema.",
      hintText,
      "Diagnose the true disease, pest, nutrient stress, or healthy state strictly from the visual evidence in the photo—never force-fit an unrelated condition into a rust or blight label.",
      "Distinguish carefully between foliar fungal symptoms (linear yellow rust stripes vs scattered brown rust pustules vs black stem rust vs white powdery mildew vs tan leaf blight), head/spike diseases (wheat scab / Fusarium head blight showing premature bleached/straw spikelets vs dark smutted ears), and insect/mite pests (aphids, mites, stem fly).",
      `For canonical_class_id, output a lowercase snake_case "<crop>_<condition>" id (matching [${allIds}] when applicable, or "<crop>_<condition>" for any other condition).`,
      "Rules:",
      "- Close-up/macro photos of leaf surfaces, pustules, stems, or grain ears ARE valid crop photos (keep is_plant_leaf=true even if cropped or slightly low-res). Set is_plant_leaf=false ONLY for completely non-plant subjects (e.g., blank wall, face, vehicle, text document).",
      "- severity_grade must follow the 0-5 botanical scale: 0 (<=2% healthy/trace), 1 (<=10%), 2 (<=25%), 3 (<=50%), 4 (<=75%), 5 (>75%). If healthy, set estimated_severity_pct=0 and severity_grade=0.",
      "- Provide concise, accurate symptoms and actionable ICAR/KVK-aligned IPM steps in BOTH English (*_en) and Hindi (*_hi).",
      "- Advise confirming chemical spray dosages with the nearest KVK or Kisan Call Centre (1800-180-1551) when chemical control is mentioned.",
    ].join("\n");
  }

  function parseDataUrl(dataUrl) {
    const match = /^data:([^;]+);base64,(.+)$/i.exec(String(dataUrl || ""));
    if (!match) {
      throw new Error("Invalid image data URL for AI analysis.");
    }
    return { mimeType: match[1], base64Data: match[2] };
  }

  function extractJsonPayload(rawText) {
    const text = String(rawText || "").trim();
    if (!text) throw new Error("Empty response from Gemini API.");
    try {
      return JSON.parse(text);
    } catch {
      const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
      if (fenced) return JSON.parse(fenced[1].trim());
      const firstBrace = text.indexOf("{");
      const lastBrace = text.lastIndexOf("}");
      if (firstBrace >= 0 && lastBrace > firstBrace) {
        return JSON.parse(text.slice(firstBrace, lastBrace + 1));
      }
      throw new Error("Model output was not valid JSON.");
    }
  }

  function clamp(num, min, max, fallback) {
    const n = Number(num);
    if (!Number.isFinite(n)) return fallback;
    return Math.max(min, Math.min(max, n));
  }

  function severityToGrade(pct) {
    if (pct <= 2.0) return 0;
    if (pct <= 10.0) return 1;
    if (pct <= 25.0) return 2;
    if (pct <= 50.0) return 3;
    if (pct <= 75.0) return 4;
    return 5;
  }

  function normalizeModelResult(rawObj, modelId, latencyMs) {
    const isLeaf = rawObj.is_plant_leaf !== false;
    const isHealthy = Boolean(rawObj.is_healthy);
    const sevPct = isHealthy || !isLeaf ? 0.0 : Math.round(clamp(rawObj.estimated_severity_pct, 0, 100, 15) * 10) / 10;
    const grade = isHealthy || !isLeaf
      ? 0
      : Math.round(clamp(rawObj.severity_grade, 0, 5, severityToGrade(sevPct)));
    const conf = Math.round(clamp(rawObj.confidence_pct, 0, 100, 75) * 10) / 10;
    const canonId = String(rawObj.canonical_class_id || "unknown_condition")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "") || "unknown_condition";

    const toList = (val, fallback) =>
      Array.isArray(val) && val.length > 0 ? val.map(String) : [fallback];

    return {
      modelId: normalizeModelId(modelId),
      ok: true,
      latencyMs: Math.round(latencyMs || 0),
      isPlantLeaf: isLeaf,
      imageQualityNote: String(rawObj.image_quality_note || ""),
      detectedCrop: String(rawObj.detected_crop || "unknown").toLowerCase(),
      canonicalClassId: canonId,
      isHealthy,
      diagnosisEn: String(rawObj.diagnosis_en || canonId),
      diagnosisHi: String(rawObj.diagnosis_hi || rawObj.diagnosis_en || canonId),
      pathogen: String(rawObj.pathogen || (isHealthy ? "None" : "Consult local KVK for lab confirmation")),
      confidencePct: conf,
      estimatedSeverityPct: sevPct,
      severityGrade: grade,
      symptomsEn: String(rawObj.symptoms_en || ""),
      symptomsHi: String(rawObj.symptoms_hi || rawObj.symptoms_en || ""),
      reasoningEn: String(rawObj.reasoning_en || ""),
      actionEn: toList(rawObj.action_en, "Verify symptoms with your local KVK or Kisan Call Centre (1800-180-1551)."),
      actionHi: toList(rawObj.action_hi, "सलाह की पुष्टि के लिए नजदीकी KVK या किसान कॉल सेंटर (1800-180-1551) से संपर्क करें।"),
    };
  }

  function buildConsensus(results) {
    const successes = results.filter((r) => r && r.ok);
    if (successes.length === 0) {
      return null;
    }

    // Group votes by canonicalClassId (or normalized diagnosis_en if unknown)
    const buckets = new Map();
    successes.forEach((r) => {
      const key = r.canonicalClassId || r.diagnosisEn.toLowerCase();
      if (!buckets.has(key)) {
        buckets.set(key, { key, count: 0, weight: 0, items: [] });
      }
      const b = buckets.get(key);
      b.count += 1;
      b.weight += Math.max(1, r.confidencePct);
      b.items.push(r);
    });

    const ranked = Array.from(buckets.values()).sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return b.weight - a.weight;
    });

    const winnerBucket = ranked[0];
    // Pick highest-confidence representative from winning bucket
    const primary = winnerBucket.items.slice().sort((a, b) => b.confidencePct - a.confidencePct)[0];
    const avgConf =
      Math.round(
        (winnerBucket.items.reduce((acc, x) => acc + x.confidencePct, 0) / winnerBucket.items.length) * 10
      ) / 10;
    const avgSev =
      Math.round(
        (winnerBucket.items.reduce((acc, x) => acc + x.estimatedSeverityPct, 0) / winnerBucket.items.length) * 10
      ) / 10;

    return {
      ...primary,
      confidencePct: avgConf,
      estimatedSeverityPct: primary.isHealthy || !primary.isPlantLeaf ? 0 : avgSev,
      severityGrade: primary.isHealthy || !primary.isPlantLeaf ? 0 : severityToGrade(avgSev),
      agreementCount: winnerBucket.count,
      totalModelsSucceeded: successes.length,
      totalModelsRequested: results.length,
      allModelResults: results,
    };
  }

  async function listAvailableModels(apiKey, fetchFn) {
    const key = String(apiKey || "").trim();
    if (!key) throw new Error("Enter a Gemini API key first.");
    const doFetch = fetchFn || fetch;
    const url = `${GEMINI_BASE}/models?key=${encodeURIComponent(key)}&pageSize=100`;
    const resp = await doFetch(url, { method: "GET" });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const msg = data?.error?.message || `HTTP ${resp.status}`;
      throw new Error(msg);
    }
    const rawModels = Array.isArray(data.models) ? data.models : [];
    return rawModels
      .filter((m) => {
        const name = String(m.name || "");
        const methods = Array.isArray(m.supportedGenerationMethods) ? m.supportedGenerationMethods : [];
        return (
          name.includes("gemini") &&
          methods.includes("generateContent") &&
          !name.includes("embedding") &&
          !name.includes("tts") &&
          !name.includes("-image") &&
          !name.includes("nano-banana") &&
          !name.includes("transcribe") &&
          !name.includes("robotics") &&
          !name.includes("computer-use") &&
          !name.includes("customtools") &&
          !name.startsWith("models/gemini-2.")
        );
      })
      .map((m) => ({
        id: normalizeModelId(m.name),
        label: m.displayName ? `${m.displayName} (${normalizeModelId(m.name)})` : normalizeModelId(m.name),
      }));
  }

  async function analyzeWithModel({ apiKey, modelId, dataUrl, cropHint, knownClasses, fetchFn }) {
    const cleanModel = normalizeModelId(modelId);
    const { mimeType, base64Data } = parseDataUrl(dataUrl);
    const prompt = buildPrompt(cropHint, knownClasses);
    const doFetch = fetchFn || fetch;
    const url = `${GEMINI_BASE}/models/${encodeURIComponent(cleanModel)}:generateContent?key=${encodeURIComponent(apiKey)}`;

    const makeBody = (useSchema) => ({
      contents: [
        {
          parts: [
            { inline_data: { mime_type: mimeType, data: base64Data } },
            { text: prompt },
          ],
        },
      ],
      generationConfig: useSchema
        ? { temperature: 0.1, responseMimeType: "application/json", responseSchema: RESPONSE_SCHEMA }
        : { temperature: 0.1, responseMimeType: "application/json" },
    });

    const t0 = Date.now();
    let resp = await doFetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(makeBody(true)),
    });
    let payload = await resp.json().catch(() => ({}));

    // Retry without strict responseSchema if a specific preview/custom model rejects schema fields
    if (!resp.ok && resp.status === 400) {
      resp = await doFetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(makeBody(false)),
      });
      payload = await resp.json().catch(() => ({}));
    }

    const elapsed = Date.now() - t0;
    if (!resp.ok) {
      const msg = payload?.error?.message || `HTTP ${resp.status}`;
      return { modelId: cleanModel, ok: false, error: msg, latencyMs: elapsed };
    }

    const parts = payload?.candidates?.[0]?.content?.parts || [];
    const textOut = parts.map((p) => p.text || "").join("\n").trim();
    const parsed = extractJsonPayload(textOut);
    return normalizeModelResult(parsed, cleanModel, elapsed);
  }

  async function analyzeLeafMultiModel({ apiKey, models, dataUrl, cropHint, knownClasses, fetchFn }) {
    const cleanKey = String(apiKey || "").trim();
    if (!cleanKey) {
      throw new Error("Gemini API key is missing.");
    }
    const rawModels = Array.isArray(models) && models.length > 0 ? models.map(normalizeModelId).filter(Boolean) : ["gemini-3.5-flash-lite"];
    const modelList = Array.from(new Set(rawModels));
    const settled = await Promise.all(
      modelList.map((m) =>
        analyzeWithModel({ apiKey: cleanKey, modelId: m, dataUrl, cropHint, knownClasses, fetchFn }).catch((err) => ({
          modelId: m,
          ok: false,
          error: err && err.message ? err.message : String(err),
          latencyMs: 0,
        }))
      )
    );
    const consensus = buildConsensus(settled);
    return {
      consensus,
      results: settled,
    };
  }

  const api = {
    GEMINI_BASE,
    STORAGE_CFG_KEY,
    DEFAULT_MODELS,
    RESPONSE_SCHEMA,
    loadConfig,
    saveConfig,
    clearConfigKey,
    normalizeModelId,
    buildPrompt,
    parseDataUrl,
    extractJsonPayload,
    normalizeModelResult,
    buildConsensus,
    listAvailableModels,
    analyzeWithModel,
    analyzeLeafMultiModel,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.RabiAIVision = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
