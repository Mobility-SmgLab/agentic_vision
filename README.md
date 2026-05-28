# Visio · Manufacturing Intelligence (Streamlit)

## What this project does
This repo contains a Streamlit app for:
- **Gauge Reader**: agentic analog gauge reading using **Gemini Robotics ER 1.6** with code execution and structured JSON output.
- **PCB Inspection**: detects PCB defects and outputs a structured defect report.
- **Label & Packaging**: checks label/packaging quality (OCR, placement, barcode/QR, date fields) and outputs a structured defect report.

## Key files
- `qc_vision.py`: main multi-mode Streamlit app (mode selector + PCB/Label flows).
- `streamlit_er16_image_first.py`: embedded Gauge Reader app (rendered inside `qc_vision.py` under Gauge Reader).
- `read_gauges_er16.py`, `read_gauges_er16_zoom_codeexec.py`: gauge-reading prompts/helpers.

## Setup
Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file next to `qc_vision.py` with two API keys:

```env
GEMINI_API_KEY_1=...   # used for Gauge Reader
GEMINI_API_KEY_2=...   # used for PCB Inspection + Label & Packaging
```

## Run

```bash
streamlit run qc_vision.py
```

## Workflow / Routing
### Mode selection
In `qc_vision.py`, the user selects one of the inspection modes:
- **Gauge Reader**
- **PCB Inspection**
- **Label & Packaging**

### API keys and models (locked)
The app is intentionally locked so each mode uses a specific key + model:

- **Gauge Reader**
  - **API key**: `GEMINI_API_KEY_1`
  - **Model**: `gemini-robotics-er-1.6-preview`
  - **UI**: `qc_vision.py` embeds `streamlit_er16_image_first.render_app(...)`

- **PCB Inspection** + **Label & Packaging**
  - **API key**: `GEMINI_API_KEY_2`
  - **Model**: `gemini-3-flash-preview`
  - **UI**: handled directly in `qc_vision.py`

### Gauge Reader prompts
In `streamlit_er16_image_first.py`:
- If the user runs a **Scene Prompt**, the app will also run a second pass with the **Gauge JSON prompt** (if the scene output isn’t parseable JSON), so the UI can still render:
  - annotated overlay
  - gauge result cards

### Outputs
- **Gauge Reader**
  - Annotated output image (bbox + needle/center markers + labels)
  - Scene Description card (when Scene Prompt used)
  - Gauge Results cards + full JSON expander
- **PCB/Label**
  - Verdict + summary stats
  - Annotated output image
  - Defect report cards (with extra label fields like batch/lot/expiry/code when provided)# agentic_vision
