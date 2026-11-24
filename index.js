/**
 * index.js
 * --------
 * CommonJS entrypoint for the @lunganlungpg/rabi-crop-doctor npm package.
 * Exposes the exact 28-feature leaf pathology extractor and all 5 offline
 * classification engines (Deep ANN, MLP ANN, Random Forest, RBF-SVM, Hybrid)
 * without external runtime dependencies.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const featureExtractor = require("./web/js/feature_extractor.js");
const inferenceEngine = require("./web/js/inference_engine.js");

let cachedModels = null;
let cachedKnowledge = null;

/**
 * Lazy-load exported offline JSON model weights (Scaler, MLP-ANN, Deep-ANN, RF, SVM).
 * @returns {object} Serialized model bundle
 */
function loadModelBundle() {
  if (!cachedModels) {
    const p = path.join(__dirname, "models", "android_models.json");
    cachedModels = JSON.parse(fs.readFileSync(p, "utf8"));
  }
  return cachedModels;
}

/**
 * Lazy-load bilingual (EN/HI) ICAR/KVK Rabi crop disease knowledge base.
 * @returns {object} Knowledge metadata JSON
 */
function loadKnowledgeBase() {
  if (!cachedKnowledge) {
    const p = path.join(__dirname, "data", "rabi_crops_knowledge.json");
    cachedKnowledge = JSON.parse(fs.readFileSync(p, "utf8"));
  }
  return cachedKnowledge;
}

/**
 * Classify a 128x128 RGBA leaf image buffer using all offline ML & ANN engines.
 *
 * @param {Uint8ClampedArray|Uint8Array|number[]} rgbaData Flat 128*128*4 RGBA bytes
 * @param {string} [cropFilter="all"] Optional crop ('wheat'|'mustard'|'chickpea'|'potato'|'all')
 * @returns {object} Diagnostics, severity %, and model-by-model probability predictions
 */
function classifyLeafRGBA(rgbaData, cropFilter = "all") {
  const diag = featureExtractor.extractFeaturesFromRGBA(
    rgbaData,
    featureExtractor.IMG_SIZE,
    featureExtractor.IMG_SIZE
  );
  if (!diag.leafDetected) {
    return {
      leafDetected: false,
      message: "Leaf not clearly detected. Center leaf in good daylight.",
    };
  }
  const models = loadModelBundle();
  const inference = inferenceEngine.predictAllModels(diag.features, models, cropFilter);
  const winner = inference.predictions.hybrid_ensemble;
  const kb = loadKnowledgeBase();
  const classMeta = kb.classes.find((c) => c.id === winner.className) || {};

  return {
    leafDetected: true,
    predictedClass: winner.className,
    nameEn: classMeta.name_en || winner.className,
    nameHi: classMeta.name_hi || "",
    confidencePct: Math.round(winner.confidence * 10000) / 100,
    estimatedSeverityPct: classMeta.is_healthy ? 0.0 : diag.estimatedSeverityPct,
    severityGrade: classMeta.is_healthy ? 0 : diag.severityGrade,
    predictions: inference.predictions,
    features: diag.features,
  };
}

module.exports = {
  IMG_SIZE: featureExtractor.IMG_SIZE,
  FEATURE_NAMES: featureExtractor.FEATURE_NAMES,
  extractFeaturesFromRGBA: featureExtractor.extractFeaturesFromRGBA,
  predictAllModels: inferenceEngine.predictAllModels,
  loadModelBundle,
  loadKnowledgeBase,
  getClasses: () => loadModelBundle().classes,
  classifyLeafRGBA,
};
