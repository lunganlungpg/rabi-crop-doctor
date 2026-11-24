# 📦 Packages & Dependencies Guide (`rabi-crop-doctor`)

Everything inside **Rabi Crop Doctor** is structured so researchers, agronomists, developers, and farmers can access it in their preferred environment (**Python, R, Node.js/Web, Container, or Android APK**).

---

## 1. Installable Packages Provided by This Repository

| Ecosystem | Package Name | Source / Definition | Quick Install / Usage |
|:---|:---|:---|:---|
| **Python (`pip`)** | `rabi-crop-doctor` | [`pyproject.toml`](pyproject.toml), [`python/`](python/) | `pip install "git+https://github.com/lunganlungpg/rabi-crop-doctor.git"` |
| **R (Base R)** | `rabidoctor` | [`R-package/DESCRIPTION`](R-package/DESCRIPTION), [`R-package/R/`](R-package/R/) | `remotes::install_github("lunganlungpg/rabi-crop-doctor", subdir = "R-package")` |
| **Node.js / GitHub npm** | `@lunganlungpg/rabi-crop-doctor` | [`package.json`](package.json), [`index.js`](index.js) | `npm install @lunganlungpg/rabi-crop-doctor` (GitHub Packages) |
| **Container (OCI)** | `ghcr.io/lunganlungpg/rabi-crop-doctor` | [`Dockerfile`](Dockerfile), [`.github/workflows/packages.yml`](.github/workflows/packages.yml) | `docker pull ghcr.io/lunganlungpg/rabi-crop-doctor:latest` |
| **Android APK** | `in.iasri.rabidoctor` | [`RabiCropDoctor.apk`](RabiCropDoctor.apk), [`android_build/`](android_build/) | Download from [GitHub Releases](https://github.com/lunganlungpg/rabi-crop-doctor/releases/latest) |

---

## 2. Usage Examples by Language

### 🐍 Python Package (`rabi_crop_doctor`)
Works both locally (with scikit-learn pickles) and from any fresh `pip install` (automatically falling back to pure-NumPy inference over bundled JSON weights):

```python
from rabi_crop_doctor import (
    predict_leaf_image,
    compute_field_survey_metrics,
    summarize_india_rabi_incidence,
)

# 1. Classify any leaf photo across all 4 ML & ANN models + Hybrid Ensemble
result = predict_leaf_image("dataset_samples/wheat_yellow_rust_sample_1.png", crop_filter="wheat")
print(result["name_en"], result["confidence_pct"], result["severity_grade"])

# 2. Compute Plot Disease Incidence (%) and Wheeler (1969) PDI (%)
audit = compute_field_survey_metrics([
    {"is_healthy": True, "severity_pct": 0.5, "disease": "wheat_healthy"},
    {"is_healthy": False, "severity_pct": 18.0, "disease": "wheat_yellow_rust"},
])
print(audit)

# 3. Inspect regional Indian Rabi surveillance hotspots
print(summarize_india_rabi_incidence().head())
```

**CLI Tool (installed automatically with `pip install`):**
```bash
rabi-doctor dataset_samples/wheat_yellow_rust_sample_1.png --crop wheat
```

### 📊 R Companion Package (`rabidoctor`)
Requires **zero external CRAN packages** (`R >= 4.1.0`, `stats`, `utils` only):

```r
library(rabidoctor)

# Plot Disease Incidence (%) and Wheeler's PDI (%)
survey <- compute_field_incidence(
  is_healthy   = c(TRUE, FALSE, FALSE, TRUE),
  severity_pct = c(0.5, 8.0, 40.0, 1.0),
  disease      = c("wheat_healthy", "wheat_yellow_rust", "wheat_yellow_rust", "wheat_healthy")
)

# Regional hotspots & bilingual ICAR/KVK farmer actions
hotspots <- summarize_regional_incidence("Wheat")
advisory <- get_rabi_advisory("wheat_yellow_rust", lang = "hi")
```

### 🟨 Node.js / Browser Offline Inference (`@lunganlungpg/rabi-crop-doctor`)
Executes the exact same 28 plant pathology computer vision features + 5 models with zero native bindings:

```javascript
const rabiDoctor = require("@lunganlungpg/rabi-crop-doctor");
// Pass a flat 128x128x4 Uint8ClampedArray RGBA buffer:
// const result = rabiDoctor.classifyLeafRGBA(rgbaBytes, "wheat");
console.log("Supported classes:", rabiDoctor.getClasses());
```

---

## 3. Complete Dependency Inventory (`requirements.txt` & Runtime)

| Layer | Package / Dependency | Version / Role |
|:---|:---|:---|
| **Python Core** | `numpy` | `>=1.26.0` — vectorized leaf synthesis, integral box blurs, matrix inference |
| **Python Core** | `pandas` | `>=2.1.0` — feature tables (`crop_features_dataset.csv`) & incidence surveillance |
| **Python Core** | `scipy` | `>=1.11.0` — scientific routines underpinning `scikit-learn` |
| **Python ML/ANN** | `scikit-learn` | `>=1.4.0` — `RandomForestClassifier`, `SVC(kernel='rbf')`, `MLPClassifier` ANNs |
| **Python Vision** | `pillow` | `>=10.0.0` — image loading, RGB conversion, 128×128 bilinear resizing, icon rendering |
| **Python Plotting** | `matplotlib`, `plotly` | Optional (`[plots]`) — 300 DPI static publication charts & interactive HTML dashboards |
| **R Package** | `R`, `stats`, `utils` | `>=4.1.0` — zero-external-dependency epidemiological calculations |
| **Node / Web** | Vanilla ES6 + Canvas API | Zero runtime npm dependencies (`feature_extractor.js`, `inference_engine.js`) |
| **Android APK** | Android SDK 35 (`minSdk 21`), JDK 17 | Pure `android.jar` (`WebView`, `MediaStore`, `ExifInterface`, `ContentProvider`) |
