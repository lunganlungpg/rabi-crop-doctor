# 🤝 Contributing to Rabi Crop Doctor (फसल मित्र)

Thank you for helping build a free, reliable, offline crop disease tool for Indian farmers! Whether you are an **agronomist, plant pathologist, farmer-scout, translator, data scientist, or Android/web developer**, you are warmly welcome here.

Our core philosophy is **calm, sustainable, long-term collaboration** over a 1-year horizon — check [`ROADMAP.md`](ROADMAP.md) to see where we are headed!

---

## 🌱 Ways to Contribute (Code & Non-Code)

| Contribution Type | What to Edit / Submit | Skills Needed |
|:---|:---|:---|
| **Agronomic Advisories & Hindi Review** | [`data/rabi_crops_knowledge.json`](data/rabi_crops_knowledge.json) | Plant pathology / Agronomy / Hindi |
| **Regional Incidence Data** | [`data/rabi_field_incidence_india.csv`](data/rabi_field_incidence_india.csv) | Agricultural statistics / KVK survey data |
| **New Translations** | `UI_TEXT` inside [`web/js/app.js`](web/js/app.js) | Regional Indian languages |
| **ML / ANN Feature & Model Upgrades** | [`python/feature_extractor.py`](python/feature_extractor.py), [`python/train_and_evaluate.py`](python/train_and_evaluate.py) | Python, `scikit-learn`, `numpy` |
| **Farmer Web UI & Android APK** | [`web/`](web/), [`android_build/`](android_build/) | HTML/CSS/JS or Android Java |

---

## 🔀 Simple Co-Contributor Git Workflow

1. **Fork or Create a Feature Branch** from `main`:
   ```bash
   git checkout -b feat/your-topic
   # Useful prefixes: feat/, fix/, data/, i18n/, docs/
   ```
2. **Make Small, Focused Changes.**
3. **Sync & Verify Locally Before Opening a PR:**
   - **Check Python ↔ JavaScript Numerical Parity & Epidemiology Math:**
     ```bash
     python tests/test_parity_and_pipeline.py
     ```
   - **If you updated models (`android_models.json`) or knowledge JSON (`data/`):**
     Rebuild the offline web bundle first:
     ```bash
     python python/build_web_bundle.py
     ```
   - **Check JS syntax:**
     ```bash
     node --check web/js/feature_extractor.js
     node --check web/js/inference_engine.js
     node --check web/js/app.js
     ```
   - *(Optional if rebuilding Android APK)*:
     ```bash
     python build_apk.py && python verify_apk.py
     ```
4. **Open a Pull Request** targeting `main` and fill out the brief PR checklist.

---

## 🏅 Pairing & Co-Authorship Credit

If multiple contributors pair on field surveys, dataset curation, translation, or code, please credit everyone directly in the commit message so GitHub shows all contributors automatically:

```bash
git commit -m "feat(i18n): add Punjabi crop symptom cards

Co-authored-by: Collaborator Name <email@example.com>"
```

*(Never rush — even small typo fixes, new sample leaf photos, or single advisory clarifications make the tool safer and clearer for farmers!)*
