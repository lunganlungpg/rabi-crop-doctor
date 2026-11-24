# 🗓️ 1-Year Stress-Free Collaborative Roadmap (Nov 2026 – Nov 2027)

Welcome! **Rabi Crop Doctor (फसल मित्र)** is built as an open, community-driven agricultural AI tool. Because contributors include **farmers, agricultural statisticians, plant pathologists, translators, and software/ML developers** working across different schedules, our **12-month roadmap** is intentionally paced around real Indian crop seasons — **no crunch, no rushed deadlines, steady async progress**.

---

## 🌱 Guiding Principles for a Stress-Free Year
- **Season-Aligned Rhythm:** Gather field insights during Rabi winter peaks (**Dec–Mar**) and refine code, models, and regional languages comfortably during summer/monsoon months (**Apr–Sep**).
- **Small, Bite-Sized Pull Requests:** Any improvement — fixing a Hindi word, adding one farm observation row, improving a plot, or tuning an ANN hyperparameter — is valuable.
- **Offline Safety First:** Every major release must preserve zero-internet operation in `RabiCropDoctor.apk` and pass Python ↔ JavaScript inference parity tests.

---

## 📅 Quarterly Milestones (Nov 2026 – Nov 2027)

| Quarter | Months | Seasonal Focus | Relaxed Target Milestone | Core Deliverables & Activities |
|:---:|:---|:---|:---|:---|
| **Q1** | **Dec 2026 – Feb 2027** | 🌾 Active Rabi Field Season | **`v1.1` — Field Feedback & Dataset Growth** | • Capture real smartphone leaf photos (daylight, shade, overcast) across Wheat, Mustard, Chickpea & Potato.<br>• Field-test APK usability with farmers & KVK specialists.<br>• Collect feedback on Hindi UI clarity and camera framing. |
| **Q2** | **Mar – May 2027** | 🔍 Harvest & Post-Harvest Review | **`v1.2` — Real-Photo Model Fine-Tuning** | • Fine-tune `MLP ANN`, `Deep ANN`, `Random Forest`, and `RBF SVM` on real field imagery via `--image-dir`.<br>• Add optional Gradient Boosting / lightweight MobileNet-feature comparison track.<br>• Audit false-positive cases (dust, aphids, nutrient stress vs rust/blight). |
| **Q3** | **Jun – Aug 2027** | 🛠️ Monsoon Build & Localization | **`v1.3` — Multi-Language & Accessibility** | • Expand UI & crop advisories to **Punjabi (ਪੰਜਾਬੀ)**, **Marathi (मराठी)**, **Gujarati (ગੁજરાતી)** & **Bengali (বাংলা)**.<br>• Add Barley, Lentil (*Masoor*), and Field Pea (*Matar*) knowledge cards.<br>• Explore Android Text-to-Speech advisory readout for low-literacy accessibility.<br>• Add offline plot GPS/village metadata export (strictly opt-in). |
| **Q4** | **Sep – Nov 2027** | 🚀 Pre-Sowing Pilot & Open Science | **`v2.0` — Next-Season Production Release** | • Pre-sowing regression & APK compatibility testing on budget Android devices.<br>• Publish clean benchmark dataset summary & reproducible notebooks.<br>• Prepare joint open-access manuscript / extension bulletin crediting all co-contributors. |

---

## 🤝 Where Co-Contributors Can Jump In Anytime

Pick **one small area** that matches your background — zero pressure to know the whole stack:

### 1. 🌾 Plant Pathologists, Agronomists & KVK Scouts
- Review EN/HI symptom descriptions and IPM/cultural recommendations in [`data/rabi_crops_knowledge.json`](data/rabi_crops_knowledge.json).
- Contribute anonymized regional disease incidence observations (`state, crop, disease, temp, RH, incidence %, PDI %`) to [`data/rabi_field_incidence_india.csv`](data/rabi_field_incidence_india.csv).
- Donate labeled field photos (`<crop>_<disease>/*.jpg`) captured on standard Android phones.

### 2. 🗣️ Language & Farmer-UX Contributors
- Improve everyday Hindi (`हिन्दी`) readability in [`web/js/app.js`](web/js/app.js) — avoiding stiff textbook words where simpler village terminology is clearer.
- Contribute new language dictionaries (Punjabi, Marathi, Bengali, Gujarati, Bhojpuri/Awadhi audio prompts).

### 3. 🧠 Machine Learning & Artificial Neural Network Contributors
- Experiment with new pathology/texture features in [`python/feature_extractor.py`](python/feature_extractor.py) (keeping mathematical parity in [`web/js/feature_extractor.js`](web/js/feature_extractor.js)).
- Benchmark calibrated SVMs, deeper residual MLPs, compact CNN-feature extractors, or pruning/quantization in [`python/train_and_evaluate.py`](python/train_and_evaluate.py).

### 4. 📱 Web & Android APK Contributors
- Polish responsive UI cards, print layouts, and dark/sunlight contrast in [`web/styles/app.css`](web/styles/app.css).
- Enhance [`MainActivity.java`](android_build/src/in/iasri/rabidoctor/MainActivity.java) (e.g., pinch-to-zoom preview, voice readout, or Android 15 edge-to-edge padding).

---

## 🧭 Progress Check-In Cadence
- **Async by Default:** All planning happens through GitHub Issues and Pull Requests — no mandatory meetings.
- **Monthly Snapshot:** At the end of each month, maintainers tag verified updates (`v1.x.y`) and refresh **Contributors** credits in [`README.md`](README.md).
