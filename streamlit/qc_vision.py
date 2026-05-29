import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFile
import json, os, io, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(
    page_title="Visio · Manufacturing Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
:root {
    --bg:#f4f5f7; --surface:#ffffff; --card:#ffffff; --card2:#f9fafb;
    --border:#e2e5ea; --border2:#d0d5de;
    --accent:#e85d26; --accent2:#f5a623;
    --blue:#2563eb; --danger:#dc2626; --warn:#d97706; --ok:#16a34a;
    --text:#111827; --sub:#4b5563; --muted:#9ca3af;
    --mono:'DM Mono',monospace; --sans:'DM Sans',sans-serif; --display:'Syne',sans-serif;
}
*{box-sizing:border-box;}
html,body,.stApp{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
.block-container{
  padding:0 1.25rem 2.25rem!important;
  max-width:100%!important;
}
header[data-testid="stHeader"]{display:none!important;}
.stDeployButton{display:none!important;}
footer{display:none!important;}

.topbar{background:var(--surface);border-bottom:2px solid var(--border);padding:0 48px;height:64px;
  display:flex;align-items:center;justify-content:space-between;
  margin:0 -2.5rem 2rem;position:sticky;top:0;z-index:100;
  box-shadow:0 1px 4px rgba(0,0,0,0.07);}
.topbar-brand{display:flex;align-items:baseline;gap:10px;}
.topbar-logo{font-family:var(--display);font-size:24px;font-weight:800;color:var(--accent);letter-spacing:-0.5px;}
.topbar-wordmark{font-family:var(--display);font-size:24px;font-weight:700;color:var(--text);letter-spacing:-0.5px;}
.topbar-sub{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-left:4px;}
.topbar-right{display:flex;align-items:center;gap:10px;}
.chip{font-family:var(--mono);font-size:10px;letter-spacing:0.5px;padding:4px 12px;border-radius:20px;font-weight:500;}
.chip-green{background:#dcfce7;color:#15803d;border:1px solid #bbf7d0;}
.chip-blue{background:#dbeafe;color:#1d4ed8;border:1px solid #bfdbfe;}
.chip-orange{background:#fff7ed;color:#c2410c;border:1px solid #fed7aa;}

.sec-head{font-family:var(--mono);font-size:10px;font-weight:500;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;
  display:flex;align-items:center;gap:10px;}
.sec-head::after{content:'';flex:1;height:1px;background:var(--border);}

.mode-card{background:var(--card);border:1.5px solid var(--border);border-radius:10px;
  padding:16px 14px;transition:all .18s ease;position:relative;overflow:hidden;}
.mode-card:hover{border-color:var(--accent);box-shadow:0 4px 16px rgba(232,93,38,.10);transform:translateY(-1px);}
.mode-card.active{border-color:var(--accent);background:#fff6f2;box-shadow:0 4px 16px rgba(232,93,38,.12);}
.mode-card.active::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:10px 10px 0 0;}
.mode-icon{font-size:22px;margin-bottom:8px;}
.mode-title{font-family:var(--sans);font-size:12px;font-weight:600;color:var(--text);margin-bottom:3px;}
.mode-sub{font-size:11px;color:var(--sub);line-height:1.4;}
.mode-tag{display:inline-block;margin-top:8px;font-family:var(--mono);font-size:9px;
  padding:2px 7px;border-radius:4px;background:var(--bg);color:var(--muted);border:1px solid var(--border);}

div[data-testid="stFileUploader"]{background:var(--card)!important;border:2px dashed var(--border2)!important;border-radius:10px!important;}

.stButton>button{background:var(--accent)!important;color:#fff!important;font-family:var(--sans)!important;
  font-weight:600!important;font-size:13px!important;border:none!important;border-radius:8px!important;
  padding:.6rem 1.4rem!important;width:100%!important;box-shadow:0 2px 8px rgba(232,93,38,.25)!important;transition:all .18s!important;}
.stButton>button:hover{background:#d14e1e!important;box-shadow:0 4px 14px rgba(232,93,38,.35)!important;transform:translateY(-1px)!important;}

.verdict{padding:18px 22px;border-radius:10px;margin-bottom:20px;display:flex;align-items:center;gap:16px;box-shadow:0 1px 4px rgba(0,0,0,.06);}
.verdict-pass{background:#f0fdf4;border:1.5px solid #86efac;}
.verdict-fail{background:#fef2f2;border:1.5px solid #fca5a5;}
.verdict-review{background:#fffbeb;border:1.5px solid #fcd34d;}
.verdict-icon{font-size:28px;}
.verdict-title{font-family:var(--display);font-size:18px;font-weight:700;letter-spacing:.5px;}
.verdict-pass .verdict-title{color:#15803d;}
.verdict-fail .verdict-title{color:#dc2626;}
.verdict-review .verdict-title{color:#b45309;}
.verdict-sub{font-size:12px;color:var(--sub);margin-top:2px;}
.score-ring-wrap{margin-left:auto;text-align:center;min-width:70px;}
.score-num{font-family:var(--display);font-size:28px;font-weight:800;line-height:1;}
.score-lbl{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px;}

.stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px;}
.stat-card{background:var(--card);border:1.5px solid var(--border);border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.04);}
.stat-num{font-family:var(--display);font-size:30px;font-weight:800;line-height:1.1;}
.stat-lbl{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-top:3px;}
.stat-red{color:var(--danger);}.stat-amber{color:var(--warn);}.stat-green{color:var(--ok);}.stat-blue{color:var(--blue);}

.defect-card{background:var(--card);border:1.5px solid var(--border);border-radius:10px;padding:14px 16px;
  margin-bottom:10px;border-left:4px solid var(--border2);box-shadow:0 1px 3px rgba(0,0,0,.04);transition:box-shadow .15s;}
.defect-card:hover{box-shadow:0 4px 12px rgba(0,0,0,.08);}
.defect-card.critical{border-left-color:var(--danger);}
.defect-card.warning{border-left-color:var(--warn);}
.defect-card.pass{border-left-color:var(--ok);}
.defect-id{font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:3px;}
.defect-name{font-weight:600;font-size:13px;color:var(--text);margin-bottom:4px;}
.defect-loc{font-size:12px;color:var(--sub);}
.defect-cause{font-size:11px;color:var(--muted);margin-top:4px;font-style:italic;}
.defect-meta{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;align-items:center;}
.badge{display:inline-flex;align-items:center;font-family:var(--mono);font-size:9px;font-weight:500;padding:3px 8px;border-radius:5px;letter-spacing:.3px;}
.badge-red{background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5;}
.badge-amber{background:#fef3c7;color:#92400e;border:1px solid #fcd34d;}
.badge-green{background:#dcfce7;color:#15803d;border:1px solid #86efac;}
.badge-blue{background:#dbeafe;color:#1e40af;border:1px solid #93c5fd;}
.badge-slate{background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;}
.badge-teal{background:#ccfbf1;color:#0f766e;border:1px solid #99f6e4;}
.badge-purple{background:#ede9fe;color:#6d28d9;border:1px solid #c4b5fd;}

.trace-wrap{background:var(--card2);border:1.5px solid var(--border);border-radius:10px;overflow:hidden;}
.trace-hdr{padding:10px 16px;background:var(--surface);font-family:var(--mono);font-size:10px;
  color:var(--sub);letter-spacing:1.5px;text-transform:uppercase;border-bottom:1px solid var(--border);}
.trace-step{display:flex;gap:14px;padding:12px 16px;border-bottom:1px solid var(--border);align-items:flex-start;}
.trace-step:last-child{border-bottom:none;}
.trace-num{font-family:var(--mono);font-size:11px;font-weight:500;color:var(--accent);min-width:26px;
  background:#fff6f2;border:1px solid #fed7aa;border-radius:4px;padding:1px 5px;text-align:center;margin-top:1px;}
.trace-body{flex:1;}
.trace-title{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;}
.trace-content{font-size:12px;color:var(--sub);line-height:1.6;}
.trace-tags{margin-top:6px;display:flex;gap:5px;flex-wrap:wrap;}
.trace-tag{font-family:var(--mono);font-size:9px;padding:2px 7px;border-radius:4px;
  background:var(--bg);color:var(--sub);border:1px solid var(--border2);}
.conf-row{margin-top:6px;display:flex;align-items:center;gap:8px;}
.conf-bar-bg{flex:1;height:4px;background:var(--border);border-radius:4px;overflow:hidden;}
.conf-bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent2));}
.conf-pct{font-family:var(--mono);font-size:9px;color:var(--muted);min-width:28px;}

.gauge-card{background:var(--card);border:1.5px solid var(--border);border-radius:10px;
  padding:14px 16px;margin-bottom:10px;border-left:4px solid #2563eb;
  box-shadow:0 1px 3px rgba(0,0,0,.04);transition:box-shadow .15s;}
.gauge-card:hover{box-shadow:0 4px 12px rgba(0,0,0,.08);}
.gauge-role{font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:3px;text-transform:uppercase;letter-spacing:1px;}
.gauge-reading{font-family:var(--display);font-size:22px;font-weight:800;color:var(--blue);line-height:1.1;margin-bottom:4px;}
.gauge-approx{font-size:11px;color:var(--muted);font-style:italic;}
.gauge-notes{font-size:11px;color:var(--sub);margin-top:6px;line-height:1.5;}
.gauge-meta{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;}

.scene-summary{background:#f0f9ff;border:1.5px solid #bae6fd;border-radius:10px;
  padding:12px 16px;margin-bottom:16px;font-size:13px;color:#0c4a6e;line-height:1.6;}

section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1.5px solid var(--border)!important;}
div[data-testid="stSelectbox"]>div{background:var(--card)!important;border:1.5px solid var(--border)!important;border-radius:8px!important;}
div[data-testid="stTextInput"] input{background:var(--card)!important;border:1.5px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important;}
div[data-testid="stExpander"]{background:var(--card)!important;border:1.5px solid var(--border)!important;border-radius:10px!important;}
</style>
""", unsafe_allow_html=True)


# ── Env loading ────────────────────────────────────────────────────────────────
if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
else:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        try:
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        except Exception:
            pass

ENV_FILE_PATH = Path(__file__).resolve().parent / ".env"


def _mask_key(k: str) -> str:
    k = (k or "").strip()
    if not k:
        return "—"
    if len(k) <= 8:
        return "********"
    return f"{k[:4]}…{k[-4:]}"


# ── Robust image directory finder ─────────────────────────────────────────────
def _find_images_dir(subdir: str) -> Optional[Path]:
    # Always resolve relative to THIS file (qc_vision.py)
    # so streamlit/static/ is found regardless of where
    # the process is launched from (Render, Docker, local).
    here = Path(__file__).resolve().parent   # → .../streamlit/
    p = here / subdir
    if p.is_dir():
        return p
    return None



# ── JSON / drawing helpers ─────────────────────────────────────────────────────
def _extract_first_json_object(text: str) -> Optional[Dict]:
    """Extract first {...} JSON block from raw model text."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if start is None:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def _get_pil_font(size: int) -> ImageFont.ImageFont:
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "Arial.ttf", "Helvetica.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _yx1000_to_xy_px(pt, w: int, h: int) -> Optional[Tuple[int, int]]:
    if not isinstance(pt, list) or len(pt) != 2:
        return None
    try:
        y, x = float(pt[0]), float(pt[1])
    except Exception:
        return None
    return (int(round(max(0., min(1000., x)) * (w - 1) / 1000.)),
            int(round(max(0., min(1000., y)) * (h - 1) / 1000.)))


def _bbox1000_to_px(b, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(b, list) or len(b) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in b]
    except Exception:
        return None
    x0 = int(round(max(0., min(1000., xmin)) * (w - 1) / 1000.))
    y0 = int(round(max(0., min(1000., ymin)) * (h - 1) / 1000.))
    x1 = int(round(max(0., min(1000., xmax)) * (w - 1) / 1000.))
    y1 = int(round(max(0., min(1000., ymax)) * (h - 1) / 1000.))
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def draw_gauge_annotations(image: Image.Image, data: Dict) -> Image.Image:
    """Render gauge bboxes, needle tips, dial centers, and point labels onto image."""
    im = image.copy().convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    font_label = _get_pil_font(max(14, int(min(w, h) * 0.022)))
    font_small = _get_pil_font(max(11, int(min(w, h) * 0.016)))

    gauge_color = (37, 99, 235)
    needle_color = (255, 215, 0)
    center_color = (0, 255, 255)
    point_color  = (34, 197, 94)

    for idx, g in enumerate(data.get("gauges") or [], start=1):
        bbox = _bbox1000_to_px(g.get("dial_bbox_2d"), w, h)
        if bbox:
            bw = max(2, int(min(w, h) * 0.004))
            draw.rectangle(bbox, outline=gauge_color, width=bw)
            tk = 12
            for cx, cy, dx, dy in [(bbox[0], bbox[1], 1, 1), (bbox[2], bbox[1], -1, 1),
                                    (bbox[0], bbox[3], 1, -1), (bbox[2], bbox[3], -1, -1)]:
                draw.line([(cx, cy), (cx + dx * tk, cy)], fill=gauge_color, width=bw + 1)
                draw.line([(cx, cy), (cx, cy + dy * tk)], fill=gauge_color, width=bw + 1)
            tx, ty = bbox[0], max(0, bbox[1] - 28)
        else:
            tx, ty = 8, 8 + idx * 30

        role = str(g.get("role") or "unknown").upper()
        rd = g.get("reading") or {}
        val = rd.get("value")
        unit = str(rd.get("unit") or "").strip()
        approx = bool(rd.get("approximate"))
        val_s = "?" if val is None else str(val)
        line = f"{role}  {val_s} {unit}".strip()
        if approx:
            line += " ~"
        tw = 8 * len(line) + 18
        draw.rectangle((tx, ty, tx + tw, ty + 22), fill=(0, 0, 0, 200))
        draw.text((tx + 6, ty + 3), line, fill=(255, 255, 255), font=font_label)

        for key, col in (("needle_tip_point", needle_color), ("dial_center_point", center_color)):
            xy = _yx1000_to_xy_px(g.get(key), w, h)
            if xy:
                r = max(4, int(min(w, h) * 0.007))
                draw.ellipse((xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r), outline=col, width=2)
                lbl = "needle" if key == "needle_tip_point" else "center"
                draw.text((xy[0] + r + 3, xy[1] - r), lbl, fill=col, font=font_small)

    for p in data.get("points") or []:
        xy = _yx1000_to_xy_px(p.get("point"), w, h)
        label = str(p.get("label") or "").strip()
        if not xy:
            continue
        r = max(4, int(min(w, h) * 0.006))
        draw.ellipse((xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r), outline=point_color, width=2)
        if label:
            tw = 7 * len(label) + 12
            draw.rectangle((xy[0] + r + 2, xy[1] - 2, xy[0] + r + 2 + tw, xy[1] + 18), fill=(0, 0, 0, 180))
            draw.text((xy[0] + r + 6, xy[1]), label, fill=(255, 255, 255), font=font_small)

    return im


def render_gauge_cards(gauges: List[Dict]):
    for idx, g in enumerate(gauges, start=1):
        role = str(g.get("role") or "unknown").upper()
        rd = g.get("reading") or {}
        val = rd.get("value")
        unit = str(rd.get("unit") or "").strip()
        approx = bool(rd.get("approximate"))
        notes = str(g.get("notes") or "").strip()

        val_display = "—" if val is None else str(val)
        approx_tag = ' <span class="badge badge-amber">~approx</span>' if approx else ""
        unit_tag = f'<span class="badge badge-teal">{unit}</span>' if unit else ""
        notes_html = f'<div class="gauge-notes">💬 {notes}</div>' if notes else ""

        st.markdown(f"""<div class="gauge-card">
        <div class="gauge-role">Gauge {idx:02d} · {role}</div>
        <div class="gauge-reading">{val_display} <span style="font-size:14px;font-weight:400;color:var(--sub)">{unit}</span></div>
        <div class="gauge-meta">{unit_tag}{approx_tag}</div>
        {notes_html}
        </div>""", unsafe_allow_html=True)


def render_gauge_stats(data: Dict):
    gauges = data.get("gauges") or []
    readable = sum(1 for g in gauges if g.get("reading", {}).get("value") is not None)
    approx = sum(1 for g in gauges if g.get("reading", {}).get("approximate"))
    pts = len(data.get("points") or [])
    st.markdown(f"""<div class="stats-row">
    <div class="stat-card"><div class="stat-num stat-blue">{len(gauges)}</div><div class="stat-lbl">Gauges Found</div></div>
    <div class="stat-card"><div class="stat-num stat-green">{readable}</div><div class="stat-lbl">Readable</div></div>
    <div class="stat-card"><div class="stat-num stat-amber">{approx}</div><div class="stat-lbl">Approximate</div></div>
    <div class="stat-card"><div class="stat-num stat-blue">{pts}</div><div class="stat-lbl">Labeled Points</div></div>
    </div>""", unsafe_allow_html=True)


# ── Prompts / modes ────────────────────────────────────────────────────────────
GAUGE_SYSTEM_INSTRUCTION = """You are the vision module on a Unitree Go2 quadruped doing facility inspection.
Your job: locate every analog gauge or dial in the scene, read its value precisely, and return structured JSON.
Use code execution to zoom/crop around each gauge face before reading — iterate until ticks and needle are legible.
Always report coordinates relative to the ORIGINAL full-frame image (0–1000 scale)."""

GAUGE_PROMPT = """You are the vision module on a Unitree Go2 quadruped doing facility inspection.

Task: read every analog gauge or dial (pressure, vacuum, temperature, level, etc.).

Step 1 — Use code execution (zoom/crop):
- Load the image, zoom/crop tightly around each gauge face so major ticks, minor ticks, printed numbers, and the needle tip are legible.
- Iterate: coarse localization → tight crop → small rotation if the dial is skewed.

Step 2 — After inspection, emit ONE JSON object only (no markdown fences, no commentary):
{
  "scene_summary": "<one sentence what the panel/device is>",
  "points": [
    {"point": [y, x], "label": "<short label including gauge name and approximate reading + unit>"}
  ],
  "gauges": [
    {
      "role": "<e.g. TANK, OUTLET, OIL, PRESSURE, TEMP, or unknown>",
      "reading": { "value": <number or null>, "unit": "<string or null>", "approximate": <true|false> },
      "needle_tip_point": [y, x],
      "dial_center_point": [y, x],
      "dial_bbox_2d": [ymin, xmin, ymax, xmax],
      "notes": "<optional: glare, occlusion, confidence>"
    }
  ]
}

Spatial rules:
- All coordinates are integers in [0, 1000], relative to the ORIGINAL full-frame image.
- Points use [y, x]; dial_bbox_2d is [ymin, xmin, ymax, xmax].
- needle_tip_point = needle tip; dial_center_point = pivot center.
- Map crop coordinates back to original image space when reporting."""

GAUGE_SCENE_PROMPT = """Describe this industrial scene in detail for robot inspection.
Focus on:
- visible equipment and panel components
- approximate dial/gauge positions and any readable values
- potential anomalies (glare, obstruction, blur, unusual states)
- where the robot should move next to get a better reading

Be concise but precise."""

MODES = {
    "Gauge Reader": {
        "icon": "🔌", "title": "Gauge Reader", "sub": "Analog gauges, dials & panels", "tag": "Robotics / ER1.6",
        "reasoning": GAUGE_SYSTEM_INSTRUCTION,
        "prompt": GAUGE_PROMPT,
    },
    "pcb": {
        "icon": "🔌", "title": "PCB Inspection", "sub": "Solder bridges, cold joints, missing parts", "tag": "IPC-A-610",
        "reasoning": """You are an IPC-A-610 certified PCB inspector:
1. BOARD OVERVIEW — Type, density, expected component count?
2. COMPONENT INVENTORY — Present, correct orientation, no tombstoning?
3. SOLDER QUALITY — Shiny concave (good) vs dull grainy blobby (bad)?
4. DEFECT MAPPING — Bridges, opens, misalignments, lifted pads?
5. IPC VERDICT — Class 2 or 3 compliance?
Return ONLY a JSON array: [{"title":"...","content":"...","tags":["..."],"confidence":0.9}]""",
        "prompt": """You are an expert PCB quality inspector (IPC-A-610).

Focus heavily on COMPONENT PLACEMENT, especially capacitors:
- Detect misaligned capacitors (offset, rotated, skewed, not centered on pads) and tombstoned passives.
- If you see a capacitor not square to pads or shifted relative to solder pads, report it as Misalignment with a tight bbox.

Return ONLY valid JSON:
{
  "verdict":"PASS|FAIL|REVIEW",
  "verdict_reason":"...",
  "overall_quality_score":0,
  "defects":[
    {
      "id":"D001",
      "type":"Solder Bridge|Cold Joint|Missing Component|Misalignment|Tombstoning|Excess Solder|Lifted Pad",
      "component":"<e.g. capacitor C21, electrolytic, 0603 resistor, unknown>",
      "location":"<human description, e.g. near C21/J1>",
      "bbox":[ymin,xmin,ymax,xmax],
      "severity":"CRITICAL|WARNING|PASS",
      "size_estimate":"...",
      "confidence":0.9,
      "root_cause":"...",
      "action":"Rework|Scrap|Re-inspect|Accept",
      "evidence":"<short: shifted off pads / rotated / tombstoned / etc>"
    }
  ],
  "summary":"..."
}"""
    },
    "label": {
        "icon": "🏷️", "title": "Label & Packaging", "sub": "OCR accuracy, placement, barcode integrity", "tag": "Packaging Line",
        "reasoning": """You are a packaging QC inspector:
1. LABEL DETECTION — Count labels. Expected vs actual?
2. TEXT EXTRACTION — Read all text. Misprints, smears, truncation?
3. BARCODE SCAN — Intact, unobstructed, correct symbology?
4. PLACEMENT CHECK — Square, centred, no bubbles/wrinkles/overlaps?
5. COMPLIANCE VERDICT — Meets GS1/FDA/brand label standards?
Return ONLY a JSON array: [{"title":"...","content":"...","tags":["..."],"confidence":0.9}]""",
        "prompt": """You are a packaging quality inspector with OCR and barcode expertise.

You MUST extract and return key text fields even if there are no defects:
- batch/lot number (required)
- expiry date (if present)
- MFG date (if present)
- product name / SKU (if present)

Return ONLY valid JSON:
{"verdict":"PASS|FAIL|REVIEW","verdict_reason":"...","overall_quality_score":0,
"defects":[{"id":"D001","type":"Misprint|Barcode Damage|Misplacement|Wrinkle|Missing Text|Wrong Label","location":"...","bbox":[ymin,xmin,ymax,xmax],"severity":"CRITICAL|WARNING|PASS","size_estimate":"...","confidence":0.9,"root_cause":"...","action":"Relabel|Reject|Monitor|Accept"}],"summary":"..."}"""
    },
}

QC_MODEL_ID = "gemini-3-flash-preview"

SEV_COLORS = {
    "CRITICAL": ("#dc2626", (220, 38, 38)),
    "WARNING":  ("#d97706", (217, 119, 6)),
    "PASS":     ("#16a34a", (22, 163, 74)),
}


# ── QC drawing / rendering helpers ────────────────────────────────────────────
def draw_qc_annotations(image, data):
    ann = image.copy()
    draw = ImageDraw.Draw(ann, "RGBA")
    W, H = ann.size
    try:
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        fn = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        fb = fn = ImageFont.load_default()
    used = []

    def px(bbox):
        ymin, xmin, ymax, xmax = bbox
        l = max(0, min(int(xmin / 1000 * W), W - 1))
        t = max(0, min(int(ymin / 1000 * H), H - 1))
        r = max(0, min(int(xmax / 1000 * W), W - 1))
        b = max(0, min(int(ymax / 1000 * H), H - 1))
        return min(l, r), min(t, b), max(l, r), max(t, b)

    def place(lx, lw, bt, bb, lh, m=4):
        for cy in [bt - lh - m, bb + m]:
            cy = max(0, min(cy, H - lh))
            r = (lx, cy, lx + lw, cy + lh)
            if not any(r[0] < u[2] and r[2] > u[0] and r[1] < u[3] and r[3] > u[1] for u in used):
                used.append(r)
                return cy
        last = max((u[3] for u in used), default=bb)
        cy = min(last + m, H - lh)
        used.append((lx, cy, lx + lw, cy + lh))
        return cy

    for d in data.get("defects", []):
        if "bbox" not in d:
            continue
        sev = d.get("severity", "WARNING")
        hx, rgb = SEV_COLORS.get(sev, SEV_COLORS["WARNING"])
        l, t, r, b = px(d["bbox"])
        if r - l < 4 or b - t < 4:
            continue
        draw.rectangle([l, t, r, b], fill=rgb + (40,), outline=rgb, width=3)
        tk = 10
        for cx, cy, dx, dy in [(l, t, 1, 1), (r, t, -1, 1), (l, b, 1, -1), (r, b, -1, -1)]:
            draw.line([(cx, cy), (cx + dx * tk, cy)], fill=rgb, width=3)
            draw.line([(cx, cy), (cx, cy + dy * tk)], fill=rgb, width=3)
        ln1 = f"{d.get('id', '')} · {d.get('type', '')}  {int(d.get('confidence', 0) * 100)}%"
        ln2 = f"{sev}  ·  {d.get('action', '')}"
        b1 = draw.textbbox((0, 0), ln1, font=fb)
        b2 = draw.textbbox((0, 0), ln2, font=fn)
        lw2 = max(b1[2] - b1[0], b2[2] - b2[0]) + 14
        lh2 = (b1[3] - b1[1]) + (b2[3] - b2[1]) + 10
        lx = max(0, min(l, W - lw2 - 2))
        ly = place(lx, lw2, t, b, lh2)
        draw.rectangle([lx, ly, lx + lw2, ly + lh2], fill=(255, 255, 255, 220), outline=rgb, width=1)
        draw.text((lx + 6, ly + 3), ln1, fill=hx, font=fb)
        draw.text((lx + 6, ly + 3 + (b1[3] - b1[1]) + 2), ln2, fill=(80, 80, 90), font=fn)
        draw.line([(lx + lw2 // 2, ly + lh2 // 2), ((l + r) // 2, (t + b) // 2)], fill=rgb + (100,), width=1)
    return ann


def render_verdict(v, reason, score):
    cls = {"PASS": "verdict-pass", "FAIL": "verdict-fail", "REVIEW": "verdict-review"}.get(v, "verdict-review")
    icon = {"PASS": "✅", "FAIL": "❌", "REVIEW": "⚠️"}.get(v, "⚠️")
    sc = {"PASS": "#15803d", "FAIL": "#dc2626", "REVIEW": "#b45309"}.get(v, "#b45309")
    st.markdown(f"""<div class="verdict {cls}"><div class="verdict-icon">{icon}</div>
    <div style="flex:1"><div class="verdict-title">{v} — Inspection Complete</div><div class="verdict-sub">{reason}</div></div>
    <div class="score-ring-wrap"><div class="score-num" style="color:{sc}">{score}</div><div class="score-lbl">QUALITY<br>SCORE</div></div>
    </div>""", unsafe_allow_html=True)


def render_stats(defects):
    crit = sum(1 for d in defects if d.get("severity") == "CRITICAL")
    warn = sum(1 for d in defects if d.get("severity") == "WARNING")
    types = len(set(d.get("type", "") for d in defects))
    st.markdown(f"""<div class="stats-row">
    <div class="stat-card"><div class="stat-num stat-blue">{len(defects)}</div><div class="stat-lbl">Total Findings</div></div>
    <div class="stat-card"><div class="stat-num stat-red">{crit}</div><div class="stat-lbl">Critical</div></div>
    <div class="stat-card"><div class="stat-num stat-amber">{warn}</div><div class="stat-lbl">Warnings</div></div>
    <div class="stat-card"><div class="stat-num stat-green">{types}</div><div class="stat-lbl">Defect Types</div></div>
    </div>""", unsafe_allow_html=True)


def render_cards(defects):
    sb = {"CRITICAL": "badge-red", "WARNING": "badge-amber", "PASS": "badge-green"}
    for d in defects:
        sev = d.get("severity", "WARNING")
        cls = {"CRITICAL": "critical", "WARNING": "warning", "PASS": "pass"}.get(sev, "warning")
        st.markdown(f"""<div class="defect-card {cls}">
        <div class="defect-id">{d.get('id', '')} · {d.get('type', '')}</div>
        <div class="defect-name">{d.get('type', '')} — {d.get('size_estimate', '')}</div>
        <div class="defect-loc">📍 {d.get('location', '')}</div>
        <div class="defect-cause">💡 {d.get('root_cause', '')}</div>
        <div class="defect-meta">
          <span class="badge {sb.get(sev, 'badge-amber')}">{sev}</span>
          <span class="badge badge-blue">{int(d.get('confidence', 0) * 100)}% conf</span>
          <span class="badge badge-slate">→ {d.get('action', '')}</span>
        </div></div>""", unsafe_allow_html=True)


def render_reasoning(steps):
    html = '<div class="trace-wrap"><div class="trace-hdr">🧠 &nbsp;Agent Reasoning Trace</div>'
    for i, s in enumerate(steps):
        body = s.get("content", "").replace("<", "&lt;").replace(">", "&gt;")
        tags = "".join(f'<span class="trace-tag">{t}</span>' for t in s.get("tags", []))
        conf = s.get("confidence", None)
        cb = ""
        if conf is not None:
            p = int(conf * 100)
            cb = (f'<div class="conf-row"><div class="conf-bar-bg">'
                  f'<div class="conf-bar-fill" style="width:{p}%"></div></div>'
                  f'<div class="conf-pct">{p}%</div></div>')
        html += (f'<div class="trace-step"><div class="trace-num">{i + 1:02d}</div>'
                 f'<div class="trace-body"><div class="trace-title">{s.get("title", f"Step {i + 1}")}</div>'
                 f'<div class="trace-content">{body}</div>'
                 f'<div class="trace-tags">{tags}</div>{cb}</div></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.caption("API keys are taken from `.env`/environment variables.")
    with st.expander("API key status", expanded=False):
        st.caption(f".env path: `{str(ENV_FILE_PATH)}`")
        st.caption(f".env exists: `{ENV_FILE_PATH.is_file()}`")
        st.caption(f"GEMINI_API_KEY_1: `{_mask_key(os.environ.get('GEMINI_API_KEY_1'))}`")
        st.caption(f"GEMINI_API_KEY_2: `{_mask_key(os.environ.get('GEMINI_API_KEY_2'))}`")
    if not (os.environ.get("GEMINI_API_KEY_1") or "").strip():
        st.warning("Missing `GEMINI_API_KEY_1` (used for Gauge Reader).")
    if not (os.environ.get("GEMINI_API_KEY_2") or "").strip():
        st.warning("Missing `GEMINI_API_KEY_2` (used for PCB + Label & Packaging).")
    st.caption("Locked models:")
    st.caption("- Gauge Reader: `gemini-robotics-er-1.6-preview`")
    st.caption("- PCB/Label: `gemini-3-flash-preview`")

    st.markdown("---")
    st.markdown("### 🔭 Gauge Reader Settings")
    use_thinking = st.checkbox("Enable thinking config (ER1.6)", value=True,
                               help="Passes thinking_budget to the model for deeper gauge analysis.")
    thinking_budget = st.slider("Thinking budget", 0, 24576, 8192, 256, disabled=not use_thinking)
    temperature_gauge = st.slider("Temperature (Gauge)", 0.0, 1.0, 0.3, 0.05)
    gauge_prompt_mode = st.radio("Gauge prompt mode",
                                 ["Gauge JSON (structured)", "Scene Description (freeform)"],
                                 help="JSON mode returns parseable gauge readings with bboxes. Scene mode returns a freeform inspection description.")

    st.markdown("---")
    if st.button("List Available Models"):
        key2 = (os.environ.get("GEMINI_API_KEY_2") or "").strip()
        if key2:
            try:
                c = genai.Client(api_key=key2)
                st.code("\n".join(m.name for m in c.models.list() if hasattr(m, "name")))
            except Exception as e:
                st.error(str(e))

    st.markdown("---")

    # Path debug — helpful when diagnosing Render deployment issues
    with st.expander("🗂 Path debug", expanded=False):
        st.caption(f"**`__file__`**: `{Path(__file__).resolve()}`")
        st.caption(f"**cwd**: `{Path.cwd()}`")
        for sd in ("static"):
            found = _find_images_dir(sd)
            st.caption(f"**{sd}/**: `{found or 'not found'}`")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("Visio uses Gemini agentic vision. Gauge Reader uses ER1.6 code-execution + thinking for precise dial readings with annotated bboxes, needle tips, and structured output.")


# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <span class="topbar-logo">VISIO</span>
    <span class="topbar-wordmark">· Manufacturing Intelligence</span>
    <span class="topbar-sub">Agentic Vision QC</span>
  </div>
</div>""", unsafe_allow_html=True)


# ── Mode selector ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Select Inspection Mode</div>', unsafe_allow_html=True)
if "mode" not in st.session_state:
    st.session_state["mode"] = "Gauge Reader"
mode_keys = list(MODES.keys())
for row in range((len(mode_keys) + 3) // 4):
    cols = st.columns(4)
    for ci, key in enumerate(mode_keys[row * 4:(row + 1) * 4]):
        m = MODES[key]
        active = "active" if st.session_state["mode"] == key else ""
        with cols[ci]:
            st.markdown(f"""<div class="mode-card {active}">
            <div class="mode-icon">{m['icon']}</div>
            <div class="mode-title">{m['title']}</div>
            <div class="mode-sub">{m['sub']}</div>
            <span class="mode-tag">{m['tag']}</span></div>""", unsafe_allow_html=True)
            if st.button("Select", key=f"btn_{key}"):
                st.session_state["mode"] = key
                st.rerun()

mode_key = st.session_state["mode"]
mode_cfg = MODES[mode_key]
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Gauge Reader — delegate to ER1.6 sub-app ──────────────────────────────────
if mode_key == "Gauge Reader":
    from streamlit_er16_image_first import render_app as _render_er16_image_first

    key1 = (os.environ.get("GEMINI_API_KEY_1") or "").strip()
    if not key1:
        st.error("⚠️ Missing `GEMINI_API_KEY_1` for Gauge Reader.")
        st.stop()
    _render_er16_image_first(embedded=True, api_key=key1)
    st.stop()


# ── Image input section (PCB / Label modes) ────────────────────────────────────
st.markdown(f'<div class="sec-head">{mode_cfg["icon"]} {mode_cfg["title"]} — Image Input</div>',
            unsafe_allow_html=True)

left_col, right_col = st.columns([2, 2])

# Collect sample images from the streamlit/static folder only
img_root = Path(__file__).resolve().parent
def _collect_images_from_static(root: Path) -> List[str]:
    imgs: List[str] = []
    static_dir = root / "static"
    if static_dir.exists():
        for f in sorted(static_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                imgs.append(str(f.resolve()))
    return imgs

sample_images = _collect_images_from_static(img_root)

with left_col:
    # ── Sample image selector (static-only) ─────────────────────────────────
    selection: Optional[str] = None
    if sample_images:
        display_names = [Path(p).name for p in sample_images]
        chosen_idx = st.selectbox(
            "Select a sample image",
            range(len(sample_images)),
            format_func=lambda i: display_names[i],
        )
        selection = sample_images[chosen_idx]
    else:
        st.info("No sample images found in `static/`. Add image files to static/ to use this app.")

    # ── Resolve final image bytes ──────────────────────────────────────────────
    img_bytes: Optional[bytes] = None
    uploaded_name: Optional[str] = None
    image: Optional[Image.Image] = None

    if selection:
        # selection is an absolute path (collected from streamlit folder roots)
        sel_path = Path(selection)

        # Fallback: if somehow selection is not a file, try common locations
        if not sel_path.is_file():
            alt = img_root / Path(selection).name
            if alt.is_file():
                sel_path = alt
            else:
                alt2 = Path.cwd() / Path(selection).name
                if alt2.is_file():
                    sel_path = alt2

        if not sel_path.is_file():
            st.warning(f"Sample image not found on disk: `{selection}`")
        else:
            try:
                img_bytes = sel_path.read_bytes()
                uploaded_name = sel_path.name
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                st.image(image,
                         caption=f"{uploaded_name} — {image.size[0]}×{image.size[1]}px",
                         width=700)
            except Exception as e:
                st.error(f"Could not open {sel_path}")
                st.exception(e)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run = st.button("▶ Run Inspection", key="run_btn")


st.markdown("---")


# ── Execute ────────────────────────────────────────────────────────────────────
if run and image:
    key2 = (os.environ.get("GEMINI_API_KEY_2") or "").strip()
    if not key2:
        st.error("⚠️ Missing `GEMINI_API_KEY_2` for PCB/Label inspections.")
        st.stop()

    client = genai.Client(api_key=key2)

    if img_bytes is None:
        st.error("Please upload or select an image first.")
        st.stop()

    mime = "image/jpeg" if (uploaded_name or "").lower().endswith(("jpg", "jpeg")) else "image/png"

    # ── GAUGE READER (ER1.6 path) ──────────────────────────────────────────────
    if mode_key == "Gauge Reader":
        active_prompt = GAUGE_SCENE_PROMPT if "Scene" in gauge_prompt_mode else GAUGE_PROMPT

        with st.spinner("🔭 Gauge Reader — running ER1.6 agentic inspection with code execution…"):
            cfg_kwargs: Dict[str, Any] = {
                "temperature": temperature_gauge,
                "system_instruction": GAUGE_SYSTEM_INSTRUCTION,
                "tools": [types.Tool(code_execution=types.ToolCodeExecution())],
            }
            if use_thinking and "Scene" not in gauge_prompt_mode:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
            cfg = types.GenerateContentConfig(**cfg_kwargs)

            try:
                resp = client.models.generate_content(
                    model=QC_MODEL_ID,
                    contents=[types.Part.from_bytes(data=img_bytes, mime_type=mime), active_prompt],
                    config=cfg,
                )
                raw_text = (resp.text or "").strip()
            except Exception as e:
                st.error(f"Gemini API error: {e}")
                st.stop()

        if "Scene" in gauge_prompt_mode:
            st.markdown('<div class="sec-head">🔭 Scene Description</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="scene-summary">🤖 {raw_text}</div>', unsafe_allow_html=True)
            with st.expander("Raw model output", expanded=False):
                st.code(raw_text)
        else:
            parsed = _extract_first_json_object(raw_text)

            if parsed is None:
                st.warning("Could not parse JSON from model response. Showing raw output.")
                st.code(raw_text)
            else:
                summary = parsed.get("scene_summary", "")
                if summary:
                    st.markdown(f'<div class="scene-summary">🤖 {summary}</div>', unsafe_allow_html=True)

                render_gauge_stats(parsed)

                ic, cc = st.columns([3, 2])
                with ic:
                    st.markdown('<div class="sec-head">Annotated Output</div>', unsafe_allow_html=True)
                    annotated = draw_gauge_annotations(image, parsed)
                    st.image(annotated, width=700)

                    buf = io.BytesIO()
                    annotated.save(buf, format="PNG")
                    st.download_button("⬇ Download Annotated PNG", data=buf.getvalue(),
                                       file_name="gauge_annotated.png", mime="image/png")

                with cc:
                    st.markdown('<div class="sec-head">Gauge Readings</div>', unsafe_allow_html=True)
                    gauges = parsed.get("gauges") or []
                    if gauges:
                        render_gauge_cards(gauges)
                    else:
                        st.info("No gauges detected in the response.")

                    with st.expander("📋 Full JSON output", expanded=False):
                        st.code(json.dumps(parsed, indent=2, ensure_ascii=False), language="json")

                with st.expander("🧠 Raw model text (includes code execution trace)", expanded=False):
                    st.text_area("Raw response", value=raw_text, height=300)

    # ── Standard QC path (PCB, Label, etc.) ───────────────────────────────────
    else:
        with st.spinner(f"🧠 Phase 1 — {mode_cfg['title']} reasoning trace…"):
            r1 = client.models.generate_content(
                model=QC_MODEL_ID,
                contents=[types.Part.from_bytes(data=img_bytes, mime_type=mime), mode_cfg["reasoning"]],
                config=types.GenerateContentConfig(temperature=0.3))
        steps = []
        try:
            rt = r1.text.strip()
            if rt.startswith("```"):
                rt = rt.split("```")[1]
                rt = rt[4:] if rt.startswith("json") else rt
            steps = json.loads(rt.strip())
        except Exception:
            steps = [{"title": "RAW ANALYSIS", "content": r1.text, "tags": []}]

        with st.spinner("🔍 Phase 2 — Defect detection & bounding boxes…"):
            r2 = client.models.generate_content(
                model=QC_MODEL_ID,
                contents=[types.Part.from_bytes(data=img_bytes, mime_type=mime), mode_cfg["prompt"]],
                config=types.GenerateContentConfig(temperature=0.1))
        try:
            raw = r2.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                raw = raw[4:] if raw.startswith("json") else raw
            data = json.loads(raw.strip())

            try:
                with right_col:
                    st.markdown('<div class="sec-head">Annotated Output</div>', unsafe_allow_html=True)
                    st.image(draw_qc_annotations(image, data), width=700)
                    if "summary" in data:
                        st.markdown(f"""<div style="background:#f9fafb;border:1.5px solid #e2e5ea;border-radius:10px;
                        padding:12px 16px;margin-top:10px;font-size:12px;color:#4b5563;line-height:1.6;">
                        📋 {data['summary']}</div>""", unsafe_allow_html=True)

                    render_verdict(data.get("verdict", "REVIEW"), data.get("verdict_reason", "Done."),
                                   data.get("overall_quality_score", 0))
                    render_stats(data.get("defects", []))

                    st.markdown('<div class="sec-head">Defect Report</div>', unsafe_allow_html=True)
                    defects = data.get("defects", [])
                    if defects:
                        render_cards(defects)
                    else:
                        st.markdown("""<div style="background:#f0fdf4;border:1.5px solid #86efac;border-radius:10px;
                            padding:16px;text-align:center;color:#15803d;font-weight:600;">✅ No defects detected</div>""",
                                    unsafe_allow_html=True)

                    st.markdown('<div class="sec-head" style="margin-top:20px">Agent Reasoning</div>',
                                unsafe_allow_html=True)
                    with st.expander("🧠 View full reasoning trace", expanded=False):
                        render_reasoning(steps)

            except Exception:
                ic, cc = st.columns([3, 2])
                with ic:
                    st.markdown('<div class="sec-head">Annotated Output</div>', unsafe_allow_html=True)
                    st.image(draw_qc_annotations(image, data), width=700)
                    if "summary" in data:
                        st.markdown(f"""<div style="background:#f9fafb;border:1.5px solid #e2e5ea;border-radius:10px;
                        padding:12px 16px;margin-top:10px;font-size:12px;color:#4b5563;line-height:1.6;">
                        📋 {data['summary']}</div>""", unsafe_allow_html=True)

                    render_verdict(data.get("verdict", "REVIEW"), data.get("verdict_reason", "Done."),
                                   data.get("overall_quality_score", 0))
                    render_stats(data.get("defects", []))
                with cc:
                    st.markdown('<div class="sec-head">Defect Report</div>', unsafe_allow_html=True)
                    defects = data.get("defects", [])
                    if defects:
                        render_cards(defects)
                    else:
                        st.markdown("""<div style="background:#f0fdf4;border:1.5px solid #86efac;border-radius:10px;
                            padding:16px;text-align:center;color:#15803d;font-weight:600;">✅ No defects detected</div>""",
                                    unsafe_allow_html=True)
                st.markdown('<div class="sec-head" style="margin-top:20px">Agent Reasoning</div>',
                            unsafe_allow_html=True)
                with st.expander("🧠 View full reasoning trace", expanded=False):
                    render_reasoning(steps)

        except json.JSONDecodeError as e:
            st.error(f"JSON parse error: {e}")
            st.code(r2.text)
        except Exception as e:
            st.error(f"Render error: {e}")
            st.code(r2.text)

elif run and not image:
    st.warning("Please upload or select an image first.")