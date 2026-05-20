---
title: CrCl Calculator
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Creatinine Clearance Calculator

Cockcroft-Gault estimation with nephrotoxic-drug alerts.
Educational demo — not for clinical use.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
python init_db.py
python app.py
```

Open the local URL printed in the terminal (usually http://127.0.0.1:7860).

## Test case

- Age 65, Weight 70 kg, Serum creatinine 1.8 mg/dL, Sex Female
- Expected CrCl ≈ 26 mL/min
- All five nephrotoxic drugs should be flagged

## Files

- `app.py` — Gradio app with Cockcroft-Gault calculator
- `init_db.py` — Initializes `drugs.db` with 5 nephrotoxic drugs
- `requirements.txt` — Python dependencies
- `.gitignore` — Excludes venv, secrets, and database
- `.env.example` — Template showing required environment variables

## Built by

CCI Session 9 — Local Development with VS Code, Git & Hugging Face
King Hussein Cancer Center AI Office
