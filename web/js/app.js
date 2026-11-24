/**
 * app.js
 * ------
 * Interactive bilingual (English / Hindi) Rabi Crop Doctor controller.
 * Manages leaf scanning, native Android camera/gallery bridges, offline ML & ANN inference,
 * plot-level Disease Incidence / Wheeler PDI survey persistence, and India Rabi surveillance guide.
 */
(function () {
  "use strict";

  const STORAGE_KEY = "rabi_crop_doctor_survey_v1";
  let currentLang = "en";
  let selectedCrop = "all";
  let currentScan = null;
  let surveyItems = [];

  const UI_TEXT = {
    en: {
      appTitle: "Rabi Crop Doctor",
      appSub: "Free Offline ML + ANN Crop Advisory",
      langBtn: "🇮🇳 हिन्दी",
      step1: "1. Select Your Rabi Crop (Recommended)",
      step2: "2. Take or Choose Leaf Photo",
      btnCam: "📷 Take Photo",
      btnGal: "🖼️ Gallery",
      samplesLbl: "Or tap a sample Rabi leaf to test offline:",
      noLeafTitle: "Leaf Not Clearly Detected",
      noLeafDesc:
        "Please hold one leaf near the center of the frame in bright daylight and try again.",
      capOrig: "Original Leaf",
      capHeat: "AI Lesion Map",
      modelAgree: "Hybrid ML + ANN Consensus Confidence",
      advisoryTitle: "Field Action & IPM Advisory",
      addSurvey: "➕ Add to Field Survey",
      viewLab: "🔬 View ML vs ANN",
      surveyTitle: "Plot Disease Incidence & PDI Calculator",
      surveyDesc:
        "Inspect 5–20 leaves in a W-pattern across your plot to compute Disease Incidence (%) and Wheeler Percent Disease Index (PDI).",
      stTotal: "Leaves Inspected",
      stInc: "Disease Incidence",
      stPdi: "Wheeler PDI",
      labLive: "Live On-Device Model Comparison (Current Leaf)",
      labSub:
        "Every scan executes 2 Artificial Neural Networks (ANN) and 2 Machine Learning models offline inside your phone.",
      labFeat: "Extracted Pathology Feature Signals",
      labBench: "RabiLeaf-1440 Offline Model Benchmark",
      guideHot: "India Rabi Disease Incidence Surveillance",
      nav_scan: "Scan Leaf",
      nav_survey: "Survey",
      nav_lab: "ML vs ANN",
      nav_guide: "Rabi Guide",
      crop_all: "Auto Detect",
      crop_wheat: "Wheat",
      crop_mustard: "Mustard",
      crop_chickpea: "Chickpea",
      crop_potato: "Potato",
      leg_healthy: "Healthy",
      leg_yellow: "Chlorosis / Yellow Rust",
      leg_blight: "Brown Rust / Necrosis",
      leg_white: "White Blisters",
    },
    hi: {
      appTitle: "फसल मित्र (रबी डॉक्टर)",
      appSub: "निःशुल्क ऑफ़लाइन AI फसल रोग जांच",
      langBtn: "🌐 English",
      step1: "1. अपनी रबी फसल चुनें (सटीकता के लिए)",
      step2: "2. पत्ती की फोटो लें या चुनें",
      btnCam: "📷 कैमरा खोलें",
      btnGal: "🖼️ गैलरी चुनें",
      samplesLbl: "या जांचने के लिए नीचे दिए नमूने पर टैप करें:",
      noLeafTitle: "पत्ती स्पष्ट नहीं दिखी",
      noLeafDesc:
        "कृपया दिन की साफ रोशनी में एक पत्ती को कैमरे के बीच में रखकर दोबारा फोटो लें।",
      capOrig: "असली पत्ती",
      capHeat: "AI रोग मानचित्र",
      modelAgree: "ML + ANN संयुक्त विश्वसनीयता",
      advisoryTitle: "खेत के लिए बचाव व उपचार सलाह",
      addSurvey: "➕ खेत सर्वेक्षण में जोड़ें",
      viewLab: "🔬 ML बनाम ANN देखें",
      surveyTitle: "खेत रोग प्रकोप (%) एवं PDI कैलकुलेटर",
      surveyDesc:
        "अपने खेत में W-आकार में 5–20 पत्तियों की जांच जोड़कर कुल रोग प्रकोप और तीव्रता सूचकांक (PDI) निकालें।",
      stTotal: "जांची गई पत्तियां",
      stInc: "रोग प्रकोप (%)",
      stPdi: "PDI सूचकांक",
      labLive: "फ़ोन पर ML और ANN मॉडल तुलना (वर्तमान पत्ती)",
      labSub:
        "हर जांच आपके फ़ोन के अंदर बिना इंटरनेट 2 न्यूरल नेटवर्क (ANN) और 2 मशीन लर्निंग मॉडल चलाती है।",
      labFeat: "पत्ती से निकाले गए मुख्य वैज्ञानिक लक्षण",
      labBench: "रबी-लीफ 1440 ऑफ़लाइन मॉडल प्रदर्शन",
      guideHot: "भारत के प्रमुख रबी रोग प्रकोप क्षेत्र",
      nav_scan: "जांच करें",
      nav_survey: "सर्वेक्षण",
      nav_lab: "ML व ANN",
      nav_guide: "रोग गाइड",
      crop_all: "स्वतः पहचान",
      crop_wheat: "गेहूँ",
      crop_mustard: "सरसों",
      crop_chickpea: "चना",
      crop_potato: "आलू",
      leg_healthy: "स्वस्थ भाग",
      leg_yellow: "पीलापन / पीला रतुआ",
      leg_blight: "भूरा रतुआ / झुलसा",
      leg_white: "सफेद फफोले",
    },
  };

  function getBundle() {
    return window.RABI_BUNDLE;
  }

  function findClassInfo(clsId) {
    const classes = getBundle().knowledge.classes;
    return classes.find((c) => c.id === clsId) || classes[0];
  }

  function showToast(msg) {
    if (window.AndroidBridge && window.AndroidBridge.showToast) {
      window.AndroidBridge.showToast(msg);
    } else {
      console.log("[Toast]", msg);
    }
  }

  function applyLanguage() {
    const t = UI_TEXT[currentLang];
    document.getElementById("app-title").textContent = t.appTitle;
    document.getElementById("app-subtitle").textContent = t.appSub;
    document.getElementById("btn-lang").textContent = t.langBtn;
    document.getElementById("lbl-step1").textContent = t.step1;
    document.getElementById("lbl-step2").textContent = t.step2;
    document.getElementById("btn-camera").textContent = t.btnCam;
    document.getElementById("btn-gallery").textContent = t.btnGal;
    document.getElementById("lbl-samples").textContent = t.samplesLbl;
    document.getElementById("lbl-noleaf-title").textContent = t.noLeafTitle;
    document.getElementById("lbl-noleaf-desc").textContent = t.noLeafDesc;
    document.getElementById("cap-orig").textContent = t.capOrig;
    document.getElementById("cap-heat").textContent = t.capHeat;
    document.getElementById("lbl-model-agree").textContent = t.modelAgree;
    document.getElementById("lbl-advisory").textContent = t.advisoryTitle;
    document.getElementById("btn-add-survey").textContent = t.addSurvey;
    document.getElementById("btn-goto-lab").textContent = t.viewLab;
    document.getElementById("lbl-survey-title").textContent = t.surveyTitle;
    document.getElementById("lbl-survey-desc").textContent = t.surveyDesc;
    document.getElementById("lbl-st-total").textContent = t.stTotal;
    document.getElementById("lbl-st-inc").textContent = t.stInc;
    document.getElementById("lbl-st-pdi").textContent = t.stPdi;
    document.getElementById("lbl-lab-live").textContent = t.labLive;
    document.getElementById("lbl-lab-sub").textContent = t.labSub;
    document.getElementById("lbl-lab-feat").textContent = t.labFeat;
    document.getElementById("lbl-lab-bench").textContent = t.labBench;
    document.getElementById("lbl-guide-hot").textContent = t.guideHot;

    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const k = el.getAttribute("data-i18n");
      if (t[k]) el.textContent = t[k];
    });

    renderSampleStrip();
    if (currentScan) renderScanResult(currentScan);
    renderSurvey();
    renderGuide();
  }

  function renderSampleStrip() {
    const strip = document.getElementById("sample-strip");
    strip.innerHTML = "";
    const samples = getBundle().samples;
    samples.forEach((smp) => {
      if (selectedCrop !== "all" && smp.crop !== selectedCrop) return;
      const clsInfo = findClassInfo(smp.id);
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "sample-pill";
      const label = currentLang === "hi" ? clsInfo.name_hi : clsInfo.name_en;
      pill.innerHTML = `<img src="${smp.data_url}" alt="${label}" /><div>${label}</div>`;
      pill.addEventListener("click", () => analyzeImageUrl(smp.data_url));
      strip.appendChild(pill);
    });
  }

  function analyzeImageUrl(dataUrl) {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, 128, 128);
      const imageData = ctx.getImageData(0, 0, 128, 128);

      const diag = window.RabiFeatureExtractor.extractFeaturesFromRGBA(
        imageData.data,
        128,
        128
      );

      if (!diag.leafDetected) {
        document.getElementById("card-no-leaf").classList.remove("hidden");
        document.getElementById("card-result").classList.add("hidden");
        return;
      }

      document.getElementById("card-no-leaf").classList.add("hidden");
      document.getElementById("card-result").classList.remove("hidden");

      const inference = window.RabiInferenceEngine.predictAllModels(
        diag.features,
        getBundle().models,
        selectedCrop
      );

      currentScan = {
        dataUrl,
        diag,
        inference,
        cropFilterAtScan: selectedCrop,
        timestamp: new Date().toISOString(),
      };

      renderScanResult(currentScan);
    };
    img.src = dataUrl;
  }

  function renderScanResult(scan) {
    const previewImg = document.getElementById("img-input-preview");
    previewImg.src = scan.dataUrl;

    const heatCanvas = document.getElementById("canvas-heatmap");
    const hCtx = heatCanvas.getContext("2d");
    const hImgData = new ImageData(scan.diag.overlayRGBA, 128, 128);
    hCtx.putImageData(hImgData, 0, 0);

    // Always evaluate with current crop selection so user can switch crop filter live!
    const inference = window.RabiInferenceEngine.predictAllModels(
      scan.diag.features,
      getBundle().models,
      selectedCrop
    );
    scan.inference = inference;

    const ensWinner = inference.predictions.hybrid_ensemble;
    const clsInfo = findClassInfo(ensWinner.className);
    const confPct = Math.round(ensWinner.confidence * 1000) / 10;
    const sevPct = clsInfo.is_healthy ? 0.0 : scan.diag.estimatedSeverityPct;
    const grade = clsInfo.is_healthy ? 0 : scan.diag.severityGrade;

    const banner = document.getElementById("diag-banner");
    banner.className = "diag-banner";
    if (!clsInfo.is_healthy) {
      banner.classList.add(grade >= 2 ? "diseased" : "watch");
    }

    document.getElementById("diag-name").textContent =
      currentLang === "hi" ? clsInfo.name_hi : clsInfo.name_en;
    document.getElementById("diag-pathogen").textContent = clsInfo.pathogen;
    document.getElementById("badge-conf").textContent =
      (currentLang === "hi" ? "सटीकता: " : "Confidence: ") + `${confPct}%`;
    document.getElementById("badge-sev").textContent =
      (currentLang === "hi" ? "संक्रमण: " : "Lesion Area: ") + `${sevPct}%`;
    document.getElementById("badge-grade").textContent =
      (currentLang === "hi" ? "ग्रेड: " : "Scale: ") + `${grade} / 5`;

    document.getElementById("diag-symptoms").textContent =
      currentLang === "hi" ? clsInfo.symptoms_hi : clsInfo.symptoms_en;

    document.getElementById("lbl-conf-pct").textContent = `${confPct}%`;
    const fill = document.getElementById("bar-conf-fill");
    fill.style.width = `${Math.min(100, confPct)}%`;
    fill.className = "bar-fill" + (clsInfo.is_healthy ? "" : " warn");

    const advList = document.getElementById("advisory-list");
    advList.innerHTML = "";
    const actions = currentLang === "hi" ? clsInfo.action_hi : clsInfo.action_en;
    actions.forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      advList.appendChild(li);
    });

    const helpline = getBundle().knowledge.helpline;
    document.getElementById("box-helpline").innerHTML =
      `<strong>📞 Kisan Call Centre: ${helpline.kisan_call_centre}</strong><br/>` +
      (currentLang === "hi" ? helpline.note_hi : helpline.note_en);

    renderLiveLab(scan);
  }

  function renderLiveLab(scan) {
    const container = document.getElementById("live-model-bars");
    if (!scan) return;

    const modelOrder = [
      ["hybrid_ensemble", "Hybrid Ensemble (ANN + SVM + RF)", ""],
      ["ann_deep", "Deep Regularized ANN (128→64→32→12)", "ann"],
      ["ann_mlp", "MLP Neural Network (64→32→12)", "ann"],
      ["random_forest", "Random Forest (45 Trees)", ""],
      ["svm_rbf", "RBF Kernel SVM (12-Class OvO)", ""],
    ];

    container.innerHTML = "";
    modelOrder.forEach(([key, label, barCls]) => {
      const pred = scan.inference.predictions[key];
      const info = findClassInfo(pred.className);
      const pct = Math.round(pred.confidence * 1000) / 10;
      const clsLabel = currentLang === "hi" ? info.name_hi : info.name_en;

      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <div class="bar-label">
          <span>${label}</span>
          <span>${clsLabel} (${pct}%)</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill ${barCls}" style="width: ${Math.min(100, pct)}%"></div>
        </div>
      `;
      container.appendChild(row);
    });

    // Display key pathological telemetry features
    const f = scan.diag.features;
    const fNames = window.RabiFeatureExtractor.FEATURE_NAMES;
    const highlightIndices = [12, 13, 14, 15, 16, 17, 20, 22, 6, 24, 25, 27];
    let tbl = "<table><thead><tr><th>Feature Signal</th><th>Extracted Value</th></tr></thead><tbody>";
    highlightIndices.forEach((idx) => {
      tbl += `<tr><td><code>${fNames[idx]}</code></td><td><strong>${f[idx].toFixed(4)}</strong></td></tr>`;
    });
    tbl += "</tbody></table>";
    document.getElementById("feature-telemetry-grid").innerHTML = tbl;
  }

  function renderBenchmarkTable() {
    const m = getBundle().metrics.models;
    const order = ["random_forest", "svm_rbf", "ann_mlp", "ann_deep", "hybrid_ensemble"];
    const tbody = document.getElementById("benchmark-tbody");
    tbody.innerHTML = "";
    order.forEach((k) => {
      const row = m[k];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${row.display_name}</strong></td>
        <td>${(row.test_accuracy * 100).toFixed(2)}%</td>
        <td><strong>${(row.crop_filtered_accuracy * 100).toFixed(2)}%</strong></td>
        <td>${row.test_f1_macro.toFixed(4)}</td>
        <td>${row.macro_roc_auc.toFixed(4)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function loadSurveyFromStorage() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      surveyItems = raw ? JSON.parse(raw) : [];
    } catch {
      surveyItems = [];
    }
  }

  function saveSurveyToStorage() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(surveyItems));
    } catch {}
  }

  function computeSurveyStats() {
    const n = surveyItems.length;
    if (n === 0) {
      return { total: 0, infected: 0, incidence: 0, pdi: 0, statusClass: "", statusText: currentLang === "hi" ? "अभी कोई पत्ती दर्ज नहीं है।" : "No scans recorded yet." };
    }
    let infected = 0;
    let ratingSum = 0;
    surveyItems.forEach((it) => {
      if (!it.isHealthy) infected++;
      ratingSum += it.grade;
    });
    const incidence = (infected / n) * 100;
    const pdi = (ratingSum / (n * 5)) * 100;

    let statusClass = "";
    let statusText = "";
    if (incidence === 0) {
      statusText = currentLang === "hi" ? "खेत स्वस्थ है — नियमित निगरानी रखें।" : "Healthy Plot — Continue routine weekly scouting.";
    } else if (incidence < 10) {
      statusClass = "watch";
      statusText = currentLang === "hi" ? "कम प्रकोप (<10%) — मेड़ों की निगरानी करें।" : "Low Watch (<10% Incidence) — Monitor field edges.";
    } else if (incidence < 25) {
      statusClass = "watch";
      statusText = currentLang === "hi" ? "मध्यम खतरा (10–25%) — KVK विशेषज्ञ से परामर्श लें।" : "Moderate Risk (10–25% Incidence) — Verify with nearest KVK.";
    } else {
      statusClass = "diseased";
      statusText = currentLang === "hi" ? "उच्च प्रकोप (>25% ETL) — तुरंत नियंत्रण आवश्यक!" : "Critical Alert (>25% Exceeds ETL) — Urgent intervention recommended!";
    }
    return {
      total: n,
      infected,
      incidence: incidence.toFixed(1),
      pdi: pdi.toFixed(1),
      statusClass,
      statusText,
    };
  }

  function renderSurvey() {
    const stats = computeSurveyStats();
    document.getElementById("stat-total").textContent = stats.total;
    document.getElementById("stat-incidence").textContent = `${stats.incidence}%`;
    document.getElementById("stat-pdi").textContent = `${stats.pdi}%`;

    const riskBox = document.getElementById("survey-risk-banner");
    riskBox.className = "diag-banner " + stats.statusClass;
    document.getElementById("survey-risk-text").textContent = stats.statusText;

    const tbody = document.getElementById("survey-tbody");
    tbody.innerHTML = "";
    surveyItems.forEach((item, idx) => {
      const info = findClassInfo(item.classId);
      const name = currentLang === "hi" ? info.name_hi : info.name_en;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td><strong>${name}</strong></td>
        <td>${item.severityPct}%</td>
        <td>${item.grade}/5</td>
        <td><button data-del="${idx}" style="border:none;background:#fee2e2;color:#b91c1c;border-radius:6px;padding:4px 8px;font-weight:800;cursor:pointer;">×</button></td>
      `;
      tr.querySelector("button").addEventListener("click", () => {
        surveyItems.splice(idx, 1);
        saveSurveyToStorage();
        renderSurvey();
      });
      tbody.appendChild(tr);
    });
  }

  function addCurrentScanToSurvey() {
    if (!currentScan) return;
    const win = currentScan.inference.predictions.hybrid_ensemble;
    const info = findClassInfo(win.className);
    const isH = Boolean(info.is_healthy);
    const sev = isH ? 0 : currentScan.diag.estimatedSeverityPct;
    const gr = isH ? 0 : Math.max(1, currentScan.diag.severityGrade);
    surveyItems.push({
      classId: win.className,
      isHealthy: isH,
      severityPct: sev,
      grade: gr,
      confidencePct: Math.round(win.confidence * 1000) / 10,
      timestamp: currentScan.timestamp,
    });
    saveSurveyToStorage();
    renderSurvey();
    showToast(currentLang === "hi" ? "खेत सर्वेक्षण में जोड़ा गया!" : "Added to Field Incidence Survey!");
  }

  function renderGuide() {
    const filter = document.getElementById("guide-crop-filter").value;
    const tbody = document.getElementById("incidence-tbody");
    tbody.innerHTML = "";
    getBundle().incidence_india.forEach((row) => {
      if (filter !== "all" && row.crop.toLowerCase() !== filter) return;
      const pillCls =
        row.disease_incidence_pct >= 25
          ? "pill-critical"
          : row.disease_incidence_pct >= 15
          ? "pill-watch"
          : "pill-safe";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${row.state}</strong></td>
        <td>${row.crop} — ${row.disease}</td>
        <td>${row.peak_month}</td>
        <td>${row.mean_temp_c}°C / ${row.mean_rh_pct}% RH</td>
        <td><span class="pill-tag ${pillCls}">${row.disease_incidence_pct}%</span></td>
        <td>${row.mean_pdi_pct}%</td>
      `;
      tbody.appendChild(tr);
    });

    const cardsDiv = document.getElementById("encyclopedia-cards");
    cardsDiv.innerHTML = "";
    getBundle().knowledge.classes.forEach((c) => {
      if (filter !== "all" && c.crop !== filter) return;
      if (c.is_healthy) return;
      const card = document.createElement("div");
      card.className = "card";
      const title = currentLang === "hi" ? `${c.name_hi} (${c.name_en})` : c.name_en;
      const symp = currentLang === "hi" ? c.symptoms_hi : c.symptoms_en;
      const trig = currentLang === "hi" ? c.weather_trigger_hi : c.weather_trigger_en;
      const acts = currentLang === "hi" ? c.action_hi : c.action_en;
      card.innerHTML = `
        <h3 class="section-title" style="color:#991b1b;">🦠 ${title}</h3>
        <div class="badge-row">
          <span class="badge">Peak: ${c.peak_months}</span>
          <span class="badge">ETL: ${c.etl_pct}%</span>
        </div>
        <p style="font-size:0.86rem; margin:6px 0;"><strong>Weather Trigger:</strong> ${trig}</p>
        <p style="font-size:0.86rem; margin:6px 0;"><strong>Symptoms:</strong> ${symp}</p>
        <ul class="advisory-list">${acts.map((a) => `<li>${a}</li>`).join("")}</ul>
      `;
      cardsDiv.appendChild(card);
    });
  }

  function handleFileInput(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      analyzeImageUrl(e.target.result);
    };
    reader.readAsDataURL(file);
  }

  function buildCsvContent() {
    const stats = computeSurveyStats();
    let csv = "index,class_id,disease_name,healthy,severity_pct,grade_0_5,confidence_pct,timestamp\n";
    surveyItems.forEach((it, idx) => {
      const info = findClassInfo(it.classId);
      csv += `${idx + 1},${it.classId},"${info.name_en}",${it.isHealthy},${it.severityPct},${it.grade},${it.confidencePct},${it.timestamp}\n`;
    });
    csv += `\nSUMMARY,Total=${stats.total},Incidence=${stats.incidence}%,PDI=${stats.pdi}%,,,,\n`;
    return csv;
  }

  // Native Android Callback Hook invoked by MainActivity.java after camera/gallery resize
  window.onNativeImageSelected = function (dataUrl) {
    if (dataUrl && typeof dataUrl === "string") {
      analyzeImageUrl(dataUrl);
    }
  };

  function init() {
    loadSurveyFromStorage();
    renderBenchmarkTable();

    // Navigation Tabs
    document.querySelectorAll(".nav-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.getAttribute("data-tab");
        document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(target).classList.add("active");
        window.scrollTo(0, 0);
      });
    });

    // Language switch
    document.getElementById("btn-lang").addEventListener("click", () => {
      currentLang = currentLang === "en" ? "hi" : "en";
      applyLanguage();
    });

    // Crop selector
    document.querySelectorAll(".crop-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        document.querySelectorAll(".crop-chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        selectedCrop = chip.getAttribute("data-crop");
        renderSampleStrip();
        if (currentScan) renderScanResult(currentScan);
      });
    });

    // Camera / Gallery buttons (Native Android Bridge with web fallback)
    document.getElementById("btn-camera").addEventListener("click", () => {
      if (window.AndroidBridge && window.AndroidBridge.capturePhoto) {
        window.AndroidBridge.capturePhoto();
      } else {
        document.getElementById("file-camera").click();
      }
    });

    document.getElementById("btn-gallery").addEventListener("click", () => {
      if (window.AndroidBridge && window.AndroidBridge.pickGalleryPhoto) {
        window.AndroidBridge.pickGalleryPhoto();
      } else {
        document.getElementById("file-gallery").click();
      }
    });

    document.getElementById("file-camera").addEventListener("change", (e) => {
      handleFileInput(e.target.files[0]);
      e.target.value = "";
    });
    document.getElementById("file-gallery").addEventListener("change", (e) => {
      handleFileInput(e.target.files[0]);
      e.target.value = "";
    });

    document.getElementById("btn-add-survey").addEventListener("click", addCurrentScanToSurvey);
    document.getElementById("btn-goto-lab").addEventListener("click", () => {
      document.querySelector('.nav-btn[data-tab="tab-lab"]').click();
    });

    document.getElementById("btn-demo-survey").addEventListener("click", () => {
      surveyItems = [
        { classId: "wheat_healthy", isHealthy: true, severityPct: 0, grade: 0, confidencePct: 97.4, timestamp: new Date().toISOString() },
        { classId: "wheat_healthy", isHealthy: true, severityPct: 0, grade: 0, confidencePct: 96.8, timestamp: new Date().toISOString() },
        { classId: "wheat_yellow_rust", isHealthy: false, severityPct: 18.2, grade: 2, confidencePct: 95.1, timestamp: new Date().toISOString() },
        { classId: "wheat_yellow_rust", isHealthy: false, severityPct: 31.4, grade: 3, confidencePct: 96.3, timestamp: new Date().toISOString() },
        { classId: "wheat_healthy", isHealthy: true, severityPct: 0, grade: 0, confidencePct: 98.0, timestamp: new Date().toISOString() },
        { classId: "wheat_brown_rust", isHealthy: false, severityPct: 12.6, grade: 2, confidencePct: 91.2, timestamp: new Date().toISOString() },
        { classId: "wheat_healthy", isHealthy: true, severityPct: 0, grade: 0, confidencePct: 95.9, timestamp: new Date().toISOString() },
        { classId: "wheat_healthy", isHealthy: true, severityPct: 0, grade: 0, confidencePct: 97.1, timestamp: new Date().toISOString() },
      ];
      saveSurveyToStorage();
      renderSurvey();
    });

    document.getElementById("btn-clear-survey").addEventListener("click", () => {
      surveyItems = [];
      saveSurveyToStorage();
      renderSurvey();
    });

    document.getElementById("btn-export-csv").addEventListener("click", () => {
      const csv = buildCsvContent();
      const fname = `Rabi_Field_Incidence_${new Date().toISOString().slice(0, 10)}.csv`;
      if (window.AndroidBridge && window.AndroidBridge.saveFile) {
        window.AndroidBridge.saveFile(fname, csv, "text/csv");
      } else {
        const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = fname;
        a.click();
      }
    });

    document.getElementById("btn-share-report").addEventListener("click", () => {
      const stats = computeSurveyStats();
      const summary =
        `Rabi Crop Doctor — Field Survey Report\n` +
        `Leaves Inspected: ${stats.total}\n` +
        `Disease Incidence: ${stats.incidence}%\n` +
        `Wheeler PDI Severity: ${stats.pdi}%\n` +
        `Assessment: ${stats.statusText}\n` +
        `Kisan Call Centre Helpline: 1800-180-1551`;
      if (window.AndroidBridge && window.AndroidBridge.shareText) {
        window.AndroidBridge.shareText("Rabi Field Disease Report", summary);
      } else if (navigator.share) {
        navigator.share({ title: "Rabi Field Disease Report", text: summary }).catch(() => {});
      } else {
        navigator.clipboard?.writeText(summary);
        alert(summary);
      }
    });

    document.getElementById("btn-print-report").addEventListener("click", () => {
      if (window.AndroidBridge && window.AndroidBridge.printDocument) {
        window.AndroidBridge.printDocument();
      } else {
        window.print();
      }
    });

    document.getElementById("guide-crop-filter").addEventListener("change", renderGuide);

    if ("serviceWorker" in navigator && location.protocol.startsWith("http")) {
      navigator.serviceWorker.register("./sw.js").catch(() => {});
    }

    const apkDl = document.getElementById("btn-apk-dl");
    if (apkDl && (window.AndroidBridge || location.protocol === "file:")) {
      apkDl.classList.add("hidden");
    }

    applyLanguage();
    // Auto-load Wheat Yellow Rust sample so first open demonstrates capability immediately!
    const samples = getBundle().samples;
    const defaultSample = samples.find((s) => s.id === "wheat_yellow_rust") || samples[0];
    if (defaultSample) analyzeImageUrl(defaultSample.data_url);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
