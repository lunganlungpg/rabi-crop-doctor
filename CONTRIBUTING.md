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

## 📷 Q1 Field Photos & Hindi UX Feedback (Dec 2026–Feb 2027)

Farmers, KVK specialists, and scouts can help validate the app with real winter-season leaves from Wheat, Mustard, Chickpea, and Potato:

1. Capture the whole leaf in focus in daylight, shade, or overcast light. Include varied natural backgrounds; avoid people, faces, and identifying details.
2. Share only photos you took or have permission to contribute, and get the field owner's consent. Remove EXIF/location metadata before sharing; do not include names, phone numbers, or exact field coordinates.
3. Use a disease folder only when the diagnosis is confirmed by a qualified reviewer. Otherwise, describe the condition as uncertain in the feedback issue rather than adding it as a training label.
4. Submit accepted, consented photos in a pull request under `data/custom/<crop>_<disease>/` (for example, `data/custom/wheat_yellow_rust/`). Use `.jpg`, `.jpeg`, `.png`, or `.webp`; the image loader reads these class-named folders for `--image-dir` training. Include the capture lighting and broad state/region in the PR description, not in filenames or precise location metadata.

To share field-testing results or Hindi UI feedback, open a **Q1 Field Photo / Hindi UX Feedback** issue. Mention the screen or prompt, what was unclear, and a suggested everyday Hindi wording if available. For camera feedback, note whether framing, focus, lighting, or leaf placement was the problem. Do not post personal or precise location information. Feedback can be submitted without photos.

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
