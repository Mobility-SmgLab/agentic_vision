#!/usr/bin/env python3
"""
Jetson demo app: Gemini Robotics-ER 1.6 — image-first layout.

Same behavior as streamlit_er16_new.py, but the main column leads with source controls
and a full-width image; system/user prompts live in a collapsed expander by default.

Run: streamlit run streamlit_er16_image_first.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from read_gauges_er16 import MODEL_ID, _extract_first_json_object
from read_gauges_er16_zoom_codeexec import SYSTEM_INSTRUCTION, ZOOM_GAUGE_PROMPT

# Match streamlit_demo_gauge_er16.py / read_gauges_zoom_codeexec (zoom + crop before JSON).
DEFAULT_SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION
DEFAULT_PROMPT_GAUGE_JSON = ZOOM_GAUGE_PROMPT

DEFAULT_PROMPT_SCENE = """Describe this industrial scene in detail for robot inspection.
Focus on:
- visible equipment and panel components
- approximate dial/gauge positions and any readable values
- potential anomalies (glare, obstruction, blur, unusual states)
- where the robot should move next to get a better reading

Be concise but precise."""

def _fmt_xy(pt: Any) -> str:
    if not isinstance(pt, list) or len(pt) != 2:
        return "—"
    return f"[y={int(pt[0])}, x={int(pt[1])}]"

def _fmt_bbox(b: Any) -> str:
    if not isinstance(b, list) or len(b) != 4:
        return "—"
    try:
        ymin, xmin, ymax, xmax = [int(float(v)) for v in b]
    except Exception:
        return "—"
    return f"[ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax}]"

def render_gauge_results_cards(parsed: Dict[str, Any]) -> None:
    gauges = parsed.get("gauges") or []
    points = parsed.get("points") or []
    if not gauges:
        st.info("No gauges found in the parsed JSON.")
        return

    def _closest_point_label(target_yx: Optional[List[Any]]) -> str:
        if not isinstance(target_yx, list) or len(target_yx) != 2:
            return ""
        try:
            ty, tx = float(target_yx[0]), float(target_yx[1])
        except Exception:
            return ""
        best = ("", 1e18)
        for p in points:
            pt = p.get("point")
            lbl = str(p.get("label") or "").strip()
            if not lbl or not isinstance(pt, list) or len(pt) != 2:
                continue
            try:
                py, px = float(pt[0]), float(pt[1])
            except Exception:
                continue
            d2 = (py - ty) ** 2 + (px - tx) ** 2
            if d2 < best[1]:
                best = (lbl, d2)
        return best[0]

    for idx, g in enumerate(gauges, start=1):
        gid = f"G{idx:03d}"
        role = str(g.get("role") or "UNKNOWN").strip().upper()
        rd = g.get("reading") or {}
        val = rd.get("value")
        unit = str(rd.get("unit") or "").strip()
        approx = bool(rd.get("approximate"))
        val_s = "—" if val is None else str(val)
        reading_s = f"{val_s} {unit}".strip()
        if approx and reading_s != "—":
            reading_s += " ~"
        notes = str(g.get("notes") or "").strip()

        loc_bbox = _fmt_bbox(g.get("dial_bbox_2d"))
        needle = _fmt_xy(g.get("needle_tip_point"))
        center = _fmt_xy(g.get("dial_center_point"))

        # Use the nearest points[].label for UNKNOWN roles (user requested "label value").
        label_value = ""
        if role in ("UNKNOWN", ""):
            label_value = _closest_point_label(g.get("needle_tip_point")) or _closest_point_label(g.get("dial_center_point"))
        title_text = (label_value or f"{role} — {reading_s}").strip()

        approx_badge = '<span class="gr-badge gr-badge-warn">APPROX</span>' if approx else '<span class="gr-badge gr-badge-ok">PRECISE</span>'
        unit_badge = f'<span class="gr-badge gr-badge-info">{unit}</span>' if unit else ""
        role_badge = f'<span class="gr-badge">{role}</span>'

        note_row = f'<div class="gr-row"><span style="min-width:16px">💡</span><span>{notes}</span></div>' if notes else ""

        st.markdown(
            f"""
<div class="gr-card">
  <div class="gr-id">{gid} · Gauge Reading</div>
  <div class="gr-title">{title_text}</div>
  <div class="gr-sub">Analog gauge reading extracted from inspection image.</div>
  <div class="gr-row"><span style="min-width:16px">📍</span><span><b>Dial bbox</b>: {loc_bbox}</span></div>
  <div class="gr-row"><span style="min-width:16px">🎯</span><span><b>Needle</b>: {needle} &nbsp;&nbsp; <b>Center</b>: {center}</span></div>
  {note_row}
  <div class="gr-badges">{approx_badge}{unit_badge}{role_badge}</div>
</div>
""",
            unsafe_allow_html=True,
        )

def render_scene_description_card(scene_text: str, parsed: Optional[Dict[str, Any]]) -> None:
    scene_text = (scene_text or "").strip()
    if not scene_text:
        return

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _strip_json_fences(text: str) -> str:
        t = (text or "").strip()
        if t.startswith("```"):
            t = t.strip("`")
            # If it starts with "json", drop that token and any leading newlines.
            if t.lower().lstrip().startswith("json"):
                t = t.lstrip()[4:].lstrip()
        return t.strip()

    def _maybe_parse_scene_json(text: str) -> Optional[Dict[str, Any]]:
        cleaned = _strip_json_fences(text)
        if not (cleaned.startswith("{") and cleaned.endswith("}")):
            return None
        try:
            return json.loads(cleaned)
        except Exception:
            return None

    def _label_for_key(key: str) -> str:
        # Derive headings from JSON keys (no hardcoded section titles).
        k = (key or "").strip().replace("-", " ").replace("_", " ")
        return " ".join(w.capitalize() if w else "" for w in k.split()) or "Field"

    def _render_json_value(v: Any, *, depth: int = 0) -> str:
        # Generic JSON → readable HTML (bullets / paragraphs).
        if depth > 3:
            return "<div style='opacity:.85'>…</div>"
        if v is None:
            return "<div>—</div>"
        if isinstance(v, (str, int, float, bool)):
            return f"<div>{esc(str(v))}</div>"
        if isinstance(v, list):
            if not v:
                return "<div>—</div>"
            items = []
            for item in v[:12]:
                items.append(f"<li>{_render_json_value(item, depth=depth+1)}</li>")
            more = "<li style='opacity:.75'>…</li>" if len(v) > 12 else ""
            return f"<ul style='margin:6px 0 0 18px'>{''.join(items)}{more}</ul>"
        if isinstance(v, dict):
            if not v:
                return "<div>—</div>"
            parts = []
            for k, vv in list(v.items())[:12]:
                parts.append(
                    f"<div style='margin-top:6px'><b>{esc(_label_for_key(str(k)))}</b>{_render_json_value(vv, depth=depth+1)}</div>"
                )
            if len(v) > 12:
                parts.append("<div style='opacity:.75;margin-top:6px'>…</div>")
            return "".join(parts)
        # Fallback for unknown types
        return f"<div>{esc(str(v))}</div>"

    summary = ""
    gauges: List[Dict[str, Any]] = []
    points: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        summary = str(parsed.get("scene_summary") or "").strip()
        gauges = parsed.get("gauges") or []
        points = parsed.get("points") or []

    # If the scene output itself is JSON, render it as a readable report.
    scene_payload = _maybe_parse_scene_json(scene_text)
    if isinstance(scene_payload, dict):
        title = esc(summary or _label_for_key("scene_description"))
        sections = []
        for k, v in scene_payload.items():
            sections.append(
                f"<div style='margin-bottom:10px'><b>{esc(_label_for_key(str(k)))}</b>{_render_json_value(v, depth=0)}</div>"
            )
        body_html = "".join(sections) or "<div>—</div>"
        st.markdown(
            f"""
<div class="gr-card">
  <div class="gr-id">SCN-001 · Scene Description</div>
  <div class="gr-title">{title}</div>
  <div class="gr-sub">
    {body_html}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    # Otherwise, treat it as plain text + add structured highlights from parsed JSON.
    readable = 0
    approx_n = 0
    gauge_lines: List[str] = []
    for i, g in enumerate(gauges, start=1):
        rd = g.get("reading") or {}
        val = rd.get("value")
        unit = str(rd.get("unit") or "").strip()
        approx = bool(rd.get("approximate"))
        if val is not None:
            readable += 1
        if approx:
            approx_n += 1
        role = str(g.get("role") or "unknown").strip().upper()
        val_s = "—" if val is None else str(val)
        line = f"{i:02d}. {role}: {val_s} {unit}".strip()
        if approx and val is not None:
            line += " ~"
        gauge_lines.append(line)

    point_lines: List[str] = []
    for p in points[:8]:
        lbl = str(p.get("label") or "").strip()
        if lbl:
            point_lines.append(lbl)

    title = summary or "Scene Description"
    scene_html = "<br/>".join(esc(scene_text).splitlines())
    gauges_html = "<br/>".join(esc(x) for x in gauge_lines[:10]) if gauge_lines else "—"
    points_html = "<br/>".join(esc(x) for x in point_lines) if point_lines else "—"

    st.markdown(
        f"""
<div class="gr-card">
  <div class="gr-id">SCN-001 · Scene Description</div>
  <div class="gr-title">{title}</div>
  <div class="gr-sub">{scene_html}</div>
  <div class="gr-row"><span style="min-width:16px">🧾</span><span><b>Gauges</b>: {len(gauges)} found · {readable} readable · {approx_n} approx</span></div>
  <div class="gr-row"><span style="min-width:16px">📌</span><span><b>Readings</b>:<br/>{gauges_html}</span></div>
  <div class="gr-row"><span style="min-width:16px">🏷️</span><span><b>Labels</b>:<br/>{points_html}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


def _get_pil_font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "Helvetica.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _yx1000_to_xy_px(pt: List[Any], w: int, h: int) -> Optional[Tuple[int, int]]:
    if not isinstance(pt, list) or len(pt) != 2:
        return None
    try:
        y = float(pt[0])
        x = float(pt[1])
    except Exception:
        return None
    x_px = int(round(max(0.0, min(1000.0, x)) * (w - 1) / 1000.0))
    y_px = int(round(max(0.0, min(1000.0, y)) * (h - 1) / 1000.0))
    return (x_px, y_px)


def _bbox1000_to_px(b: List[Any], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(b, list) or len(b) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in b]
    except Exception:
        return None
    x0 = int(round(max(0.0, min(1000.0, xmin)) * (w - 1) / 1000.0))
    y0 = int(round(max(0.0, min(1000.0, ymin)) * (h - 1) / 1000.0))
    x1 = int(round(max(0.0, min(1000.0, xmax)) * (w - 1) / 1000.0))
    y1 = int(round(max(0.0, min(1000.0, ymax)) * (h - 1) / 1000.0))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def render_labeled_image_with_role(image_bytes: bytes, data: Dict[str, Any], *, scale: float = 1.0) -> bytes:
    im = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = im.size
    draw = ImageDraw.Draw(im, "RGBA")
    scale = float(scale or 1.0)
    base = float(min(w, h))
    font_label = _get_pil_font(max(20, int(base * 0.02 * scale)))
    font_small = _get_pil_font(max(20, int(base * 0.016 * scale)))
    stroke_w = max(2, int(base * 0.004 * scale))
    pt_r = max(3, int(base * 0.006 * scale))
    pad_x = max(10, int(10 * scale))
    pad_y = max(7, int(7 * scale))
    pill_h = max(28, int(28 * scale))
    pill_r = max(10, int(10 * scale))

    def _text_wh(text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        b = draw.textbbox((0, 0), text, font=font)
        return b[2] - b[0], b[3] - b[1]

    def _pill(x: int, y: int, text: str, *, font: ImageFont.ImageFont, fg, bg) -> None:
        tw, th = _text_wh(text, font)
        w0 = tw + pad_x * 2
        h0 = max(pill_h, th + pad_y * 2)
        draw.rounded_rectangle(
            (x, y, x + w0, y + h0),
            radius=pill_r,
            fill=bg,
            outline=(148, 163, 184, 140),  # slate gray
            width=1,
        )
        draw.text((x + pad_x, y + (h0 - th) // 2), text, fill=fg, font=font)

    gauges = data.get("gauges") or []

    

    for idx, g in enumerate(gauges, start=1):
        color = (255, 80, 80)
        bbox = _bbox1000_to_px(g.get("dial_bbox_2d"), w, h)
        if bbox:
            draw.rectangle(bbox, outline=color + (255,), width=stroke_w)
            tx, ty = bbox[0], max(0, bbox[1] - int(42 * scale))
        else:
            tx, ty = 8, 8 + idx * int(26 * scale)

        role = str(g.get("role") or "unknown")
        rd = g.get("reading") or {}
        val = rd.get("value")
        unit = str(rd.get("unit") or "").strip()
        approx = bool(rd.get("approximate"))

        val_s = "?" if val is None else str(val)
        line = ("%s  %s %s" % (role, val_s, unit)).strip()
        if approx:
            line += " ~"
        _pill(tx, ty, line, font=font_label, fg=(255, 255, 255, 255), bg=(2, 6, 23, 210))

        for key, pcol in (("needle_tip_point", (255, 215, 0)), ("dial_center_point", (0, 255, 255))):
            xy = _yx1000_to_xy_px(g.get(key), w, h)
            if xy:
                draw.ellipse(
                    (xy[0] - pt_r, xy[1] - pt_r, xy[0] + pt_r, xy[1] + pt_r),
                    outline=pcol + (255,),
                    width=max(2, int(2 * scale)),
                )
                draw.text((xy[0] + pt_r + 2, xy[1] - pt_r), key.replace("_", " "), fill=pcol + (255,), font=font_small)

    for p in data.get("points") or []:
        xy = _yx1000_to_xy_px(p.get("point"), w, h)
        label = str(p.get("label") or "").strip()
        if not xy:
            continue
        draw.ellipse(
            (xy[0] - pt_r, xy[1] - pt_r, xy[0] + pt_r, xy[1] + pt_r),
            outline=(0, 255, 0, 255),
            width=max(2, int(2 * scale)),
        )
        if label:
            _pill(
                xy[0] + pt_r + max(8, int(8 * scale)),
                max(2, xy[1] - int(12 * scale)),
                label,
                font=font_small,
                fg=(255, 255, 255, 255),
                bg=(2, 6, 23, 210),
            )

    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def infer_with_custom_prompt(
    image_bytes: bytes,
    mime: str,
    *,
    api_key: Optional[str] = None,
    prompt: str,
    system_instruction: str,
    model_id: str,
    thinking_budget: Optional[int],
    temperature: float,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    cfg_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "system_instruction": system_instruction,
        "tools": [types.Tool(code_execution=types.ToolCodeExecution())],
    }
    if thinking_budget is not None:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(budget=thinking_budget)
    cfg = types.GenerateContentConfig(**cfg_kwargs)

    resp = client.models.generate_content(
        model=model_id,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime), prompt],
        config=cfg,
    )
    raw = (resp.text or "").strip()

    parsed = None
    if raw:
        try:
            parsed = _extract_first_json_object(raw)
        except Exception:
            parsed = None
    return raw, parsed


def _load_gallery_images(images_dir: Path) -> List[Path]:
    if not images_dir.is_dir():
        return []
    pats = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP")
    out: List[Path] = []
    for pat in pats:
        out.extend(images_dir.glob(pat))
    return sorted(set(out))


def _fetch_live_frame(api_base: str, timeout_sec: float) -> bytes:
    url = api_base.rstrip("/") + "/frame/latest.jpg"
    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.content


def _inject_app_styles() -> None:
    st.markdown(
        """
<style>
/* Fixed hero: keep z-index below sidebar/header so Model ID & menu stay usable */
:root {
  --er16-streamlit-header: 3.75rem;
  --er16-hero-h: 4.35rem;
  --er16-hero-top-gap: 0.55rem;
  --er16-chrome-z: 1000010;
  --er16-hero-z: 999960;
}
header[data-testid="stHeader"] {
  z-index: var(--er16-chrome-z) !important;
}
section[data-testid="stSidebar"] {
  z-index: calc(var(--er16-chrome-z) - 1) !important;
}
[data-testid="collapsedControl"] {
  z-index: calc(var(--er16-chrome-z) - 1) !important;
}
.hero-sticky-wrap {
  position: flexible;
  top: calc(var(--er16-streamlit-header) + var(--er16-hero-top-gap));
  left: 0.75rem;
  right: 0.75rem;
  width: auto;
  z-index: var(--er16-hero-z);
  box-sizing: border-box;
  min-height: var(--er16-hero-h);
  margin: 0;
  padding: 0.95rem 1.25rem 1rem 1.25rem;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 14px;
  border: 1px black;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.32);
  background: light blue;
}

.hero-sticky-wrap h1 {
  font-size: clamp(1.05rem, 2.1vw, 1.55rem);
  font-weight: 650; 
  margin: 0;
  width: 100%;
  text-align: center;
  letter-spacing: -0.02em;
  line-height: 1.28;
  background: black;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Space for fixed bar + gap under Streamlit header */
.hero-fixed-spacer {
  height: calc(var(--er16-hero-h) + var(--er16-hero-top-gap) + 0.2rem);
  margin: 0 0 0.85rem 0;
}
/* Section headings */
.section-label {
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #64748b;
  margin: 0.35rem 0 0.55rem 0;
  padding-left: 0.55rem;
  border-left: 3px solid rgba(99, 102, 241, 0.55);
}
[data-theme="light"] .section-label {
  color: #475569;
  border-left-color: rgba(79, 70, 229, 0.45);
}
/* Main column polish */
[data-testid="stAppViewContainer"] .main .block-container {
  padding-top: 0.75rem;
  padding-bottom: 2.5rem;
}
[data-testid="stAppViewContainer"] .main img {
  border-radius: 12px;
  box-shadow: 0 4px 24px rgba(15, 23, 42, 0.12);
}
[data-theme="dark"] [data-testid="stAppViewContainer"] .main img {
  box-shadow: 0 4px 28px rgba(0, 0, 0, 0.45);
}
.stTextArea textarea {
  border-radius: 12px !important;
  border: 1px solid rgba(148, 163, 184, 0.45) !important;
  padding: 0.65rem 0.75rem !important;
}
[data-theme="dark"] .stTextArea textarea {
  border-color: rgba(71, 85, 105, 0.85) !important;
}
.stButton > button {
  border-radius: 12px !important;
  font-weight: 600 !important;
}
div[data-testid="stRadio"] > div {
  gap: 0.35rem 1rem;
  flex-wrap: wrap;
  padding: 0.15rem 0 0.05rem 0;
}
[data-testid="stAppViewContainer"] .main [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stRadio"] {
  padding-bottom: 0.15rem;
}
hr {
  margin: 1.15rem 0 !important;
  border: none !important;
  border-top: 1px solid rgba(148, 163, 184, 0.35) !important;
}
[data-theme="dark"] hr {
  border-top-color: rgba(71, 85, 105, 0.55) !important;
}
/* Bordered section panels (st.container(border=True)) */
[data-testid="stAppViewContainer"] .main [data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 16px !important;
  padding: 0.4rem 0.65rem 0.85rem 0.65rem !important;
  border-color: rgba(99, 102, 241, 0.22) !important;
  background: linear-gradient(165deg, rgba(248, 250, 252, 0.95) 0%, rgba(255, 255, 255, 0.98) 55%, rgba(241, 245, 249, 0.55) 100%);
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.8) inset, 0 10px 28px rgba(15, 23, 42, 0.06);
}
[data-theme="dark"] [data-testid="stAppViewContainer"] .main [data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(165deg, rgba(30, 41, 59, 0.55) 0%, rgba(15, 23, 42, 0.72) 100%);
  border-color: rgba(129, 140, 248, 0.28) !important;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}
[data-testid="stAppViewContainer"] .main .stTextArea label p,
[data-testid="stAppViewContainer"] .main .stSelectbox label p,
[data-testid="stAppViewContainer"] .main .stRadio label p {
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.01em;
  color: #334155 !important;
}
[data-theme="dark"] [data-testid="stAppViewContainer"] .main .stTextArea label p,
[data-theme="dark"] [data-testid="stAppViewContainer"] .main .stSelectbox label p,
[data-theme="dark"] [data-testid="stAppViewContainer"] .main .stRadio label p {
  color: #e2e8f0 !important;
}
[data-testid="stAppViewContainer"] .main .stTextArea textarea {
  background: rgba(255, 255, 255, 0.92) !important;
  line-height: 1.45 !important;
  font-size: 0.92rem !important;
}
[data-theme="dark"] [data-testid="stAppViewContainer"] .main .stTextArea textarea {
  background: rgba(15, 23, 42, 0.55) !important;
}
[data-testid="stAppViewContainer"] .main div[data-baseweb="select"] > div {
  border-radius: 12px !important;
  border-color: rgba(148, 163, 184, 0.45) !important;
}
/* Primary actions: teal/cyan — scope to app view (not Deploy header); avoid .main (layout varies by Streamlit version) */
[data-testid="stAppViewContainer"] div[data-testid="stButton"] > button[kind="primary"],
[data-testid="stAppViewContainer"] .stButton > button[kind="primary"],
[data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"] {
  background-color: #0d9488 !important;
  background-image: linear-gradient(90deg, #0f766e 0%, #0d9488 48%, #0891b2 100%) !important;
  color: #f8fafc !important;
  border-color: transparent !important;
  border: none !important;
  box-shadow: 0 6px 20px rgba(13, 148, 136, 0.38) !important;
  transition: box-shadow 0.18s ease, filter 0.18s ease;
}
[data-testid="stAppViewContainer"] div[data-testid="stButton"] > button[kind="primary"] p,
[data-testid="stAppViewContainer"] .stButton > button[kind="primary"] p,
[data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"] p {
  color: #f8fafc !important;
}
[data-testid="stAppViewContainer"] div[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stAppViewContainer"] .stButton > button[kind="primary"]:hover,
[data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"]:hover {
  background-color: #0f766e !important;
  background-image: linear-gradient(90deg, #115e59 0%, #0f766e 45%, #0d9488 100%) !important;
  color: #ffffff !important;
  box-shadow: 0 8px 26px rgba(8, 145, 178, 0.48) !important;
  filter: brightness(1.03);
}
[data-testid="stAppViewContainer"] div[data-testid="stButton"] > button[kind="primary"]:focus-visible,
[data-testid="stAppViewContainer"] .stButton > button[kind="primary"]:focus-visible,
[data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"]:focus-visible {
  outline: 2px solid rgba(45, 212, 191, 0.65) !important;
  outline-offset: 2px;
}
[data-testid="stAppViewContainer"] .main [data-testid="stCode"] pre {
  border-radius: 12px !important;
  border: 1px solid rgba(148, 163, 184, 0.35) !important;
}
[data-theme="dark"] [data-testid="stAppViewContainer"] .main [data-testid="stCode"] pre {
  border-color: rgba(71, 85, 105, 0.65) !important;
}

/* Gauge results cards (QC-style) */
.gr-card {
  background: rgba(255,255,255,0.95);
  border: 1px solid rgba(148,163,184,0.55);
  border-radius: 14px;
  padding: 14px 16px;
  margin: 10px 0;
  box-shadow: 0 6px 22px rgba(15,23,42,0.06);
}
[data-theme="dark"] .gr-card {
  background: rgba(15,23,42,0.55);
  border-color: rgba(148,163,184,0.25);
}
.gr-id {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 11px;
  color: rgba(100,116,139,1);
  margin-bottom: 6px;
}
.gr-title {
  font-size: 14px;
  font-weight: 700;
  color: rgba(15,23,42,1);
  margin-bottom: 4px;
}
[data-theme="dark"] .gr-title { color: rgba(226,232,240,1); }
.gr-sub {
  font-size: 12px;
  color: rgba(51,65,85,1);
  margin-bottom: 8px;
}
[data-theme="dark"] .gr-sub { color: rgba(203,213,225,1); }
.gr-row {
  display:flex;
  gap:10px;
  align-items:flex-start;
  font-size: 12px;
  color: rgba(71,85,105,1);
  margin: 4px 0;
}
[data-theme="dark"] .gr-row { color: rgba(148,163,184,1); }
.gr-badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center;}
.gr-badge{
  display:inline-flex;align-items:center;gap:6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 10px;font-weight:600;
  padding: 4px 8px;border-radius: 10px;
  border: 1px solid rgba(148,163,184,0.35);
  background: rgba(241,245,249,0.8);
  color: rgba(15,23,42,0.9);
}
[data-theme="dark"] .gr-badge{background: rgba(30,41,59,0.6); color: rgba(226,232,240,0.95); border-color: rgba(148,163,184,0.2);}
.gr-badge-warn{background: rgba(254,243,199,0.9); border-color: rgba(251,191,36,0.55); color: rgba(146,64,14,1);}
.gr-badge-ok{background: rgba(220,252,231,0.9); border-color: rgba(34,197,94,0.55); color: rgba(21,128,61,1);}
.gr-badge-info{background: rgba(219,234,254,0.9); border-color: rgba(59,130,246,0.55); color: rgba(30,64,175,1);}
</style>
""",
        unsafe_allow_html=True,
    )


# def _render_sticky_title() -> None:
#     st.markdown(
#         """
# <div class="hero-sticky-wrap">
#   <h1>Gemini Robotics-ER 1.6 - Physical AI / Agentic Vision</h1>
# </div>
# <div class="hero-fixed-spacer" aria-hidden="true"></div>
# """,
#         unsafe_allow_html=True,
#     )

def render_app(*, embedded: bool = False, api_key: Optional[str] = None) -> None:
    """
    Render the ER1.6 image-first Streamlit UI.

    When embedded=True, the caller owns page config (do not call st.set_page_config()).
    """
    key_prefix = "er16_embed_" if embedded else "er16_"
    prompt_key = f"{key_prefix}prompt_text"

    load_dotenv()
    if api_key:
        os.environ["GEMINI_API_KEY_1"] = api_key
    if not embedded:
        st.set_page_config(page_title="ER1.6 Jetson · Image first", layout="wide")
    _inject_app_styles()
    # _render_sticky_title()

    images_dir = Path(__file__).resolve().parent / "static"
    gallery_images = _load_gallery_images(images_dir)

    with st.sidebar:
        st.subheader("Model")
        st.caption(f"Locked model: `{MODEL_ID}`")
        annotation_scale = st.slider(
            "Annotation scale",
            min_value=0.8,
            max_value=2.5,
            value=1.4,
            step=0.1,
            help="Scales label text, line thickness, and point markers on the annotated result image.",
            key=f"{key_prefix}annotation_scale",
        )
        use_thinking = st.checkbox(
            "Thinking config",
            value=True,
            help="Same as booth demo: when off, thinking_config is omitted.",
            key=f"{key_prefix}use_thinking",
        )
        thinking_budget = st.slider(
            "Thinking budget",
            min_value=0,
            max_value=24576,
            value=8192,
            step=256,
            disabled=not use_thinking,
            key=f"{key_prefix}thinking_budget",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            key=f"{key_prefix}temperature",
        )
        st.subheader("Live Capture Source")
        live_api_base = st.text_input(
            "Go2 capture API URL",
            value="http://127.0.0.1:8091",
            key=f"{key_prefix}live_api_base",
        )
        live_timeout = st.slider(
            "Live fetch timeout (sec)",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            key=f"{key_prefix}live_timeout",
        )

    if prompt_key not in st.session_state:
        st.session_state[prompt_key] = DEFAULT_PROMPT_GAUGE_JSON

    # Layout: Input (left) + Result (right)
    # Wider panes so images render larger and use whitespace better.
    left, right = st.columns([1.25, 1.35], gap="large")

    # Session keys for persistence across reruns
    k_last_input = f"{key_prefix}last_input_bytes"
    k_last_name = f"{key_prefix}last_selected_name"
    k_last_overlay = f"{key_prefix}last_overlay_png"
    k_last_raw = f"{key_prefix}last_raw_text"
    k_last_raw_scene = f"{key_prefix}last_raw_scene_text"
    k_last_parsed = f"{key_prefix}last_parsed"
    k_last_overlay_err = f"{key_prefix}last_overlay_error"

    with left:
        st.markdown('<p class="section-label">Image input</p>', unsafe_allow_html=True)
        with st.container(border=True):
            source = st.radio(
                "Select source",
                options=["Gallery", "Upload", "Live Capture"],
                horizontal=True,
                key=f"{key_prefix}source",
            )

            selected_name = "image"
            image_bytes = None  # type: Optional[bytes]
            mime = "image/jpeg"

            if source == "Gallery":
                if not gallery_images:
                    st.warning("No sample images found in `google_robotics_agentic/static`.")
                else:
                    labels = [p.name for p in gallery_images]
                    selected = st.selectbox("Sample image", labels, index=0, key=f"{key_prefix}sample_image")
                    chosen = gallery_images[labels.index(selected)]
                    image_bytes = chosen.read_bytes()
                    selected_name = chosen.name
                    mime = "image/png" if chosen.suffix.lower() == ".png" else "image/jpeg"
                    st.image(image_bytes, caption="Selected gallery image", width=700)

            elif source == "Upload":
                up = st.file_uploader(
                    "Upload image",
                    type=["jpg", "jpeg", "png", "webp"],
                    key=f"{key_prefix}upload_image",
                )
                if up is not None:
                    image_bytes = up.read()
                    selected_name = up.name or "upload.jpg"
                    mime = up.type or "image/jpeg"
                    st.image(image_bytes, caption="Uploaded image", width=700)

            else:
                c1, c2 = st.columns([1, 2])
                with c1:
                    fetch_live = st.button(
                        "Fetch Live Frame",
                        type="primary",
                        key=f"{key_prefix}fetch_live",
                    )
                with c2:
                    health_clicked = st.button(
                        "Check Capture API",
                        key=f"{key_prefix}health",
                    )

                if health_clicked:
                    try:
                        r = requests.get(live_api_base.rstrip("/") + "/health", timeout=live_timeout)
                        r.raise_for_status()
                        st.success("Capture API OK: %s" % r.json())
                    except Exception as e:
                        st.error("Capture API check failed: %s" % (e,))

                if fetch_live:
                    try:
                        image_bytes = _fetch_live_frame(live_api_base, timeout_sec=live_timeout)
                        selected_name = "live_capture.jpg"
                        mime = "image/jpeg"
                        st.image(image_bytes, caption="Live frame from Go2 API", width=700)
                    except Exception as e:
                        st.error("Live frame fetch failed: %s" % (e,))

        with st.expander("Prompts (defaults: booth / zoom demo — click to expand)", expanded=False):
            st.caption("Edit only if you need a custom task; defaults are tuned for Go2 gauge reading with code execution.")
            colp1, colp2 = st.columns([1, 1])
            with colp1:
                if st.button("Use Gauge JSON Prompt", key=f"{key_prefix}use_gauge_prompt"):
                    st.session_state[prompt_key] = DEFAULT_PROMPT_GAUGE_JSON
            with colp2:
                if st.button("Use Scene Prompt", key=f"{key_prefix}use_scene_prompt"):
                    st.session_state[prompt_key] = DEFAULT_PROMPT_SCENE

            prompt = st.text_area(
                "User prompt (editable)",
                key=prompt_key,
                height=280,
                help="Default matches booth demo: Go2 inspection + zoom/crop steps before JSON. Full-res image is sent to Gemini.",
            )

        run = st.button("Run Gemini", type="primary", key=f"{key_prefix}run")
        if run:
            if not image_bytes:
                st.error("Select/upload/fetch an image first.")
            elif not prompt.strip():
                st.error("Prompt cannot be empty.")
            else:
                with st.spinner("Running Gemini..."):
                    try:
                        raw_text, parsed = infer_with_custom_prompt(
                            image_bytes=image_bytes,
                            mime=mime,
                            api_key=api_key or os.environ.get("GEMINI_API_KEY_1"),
                            prompt=prompt,
                            system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
                            model_id=MODEL_ID,
                            thinking_budget=thinking_budget if use_thinking else None,
                            temperature=temperature,
                        )
                    except Exception as e:
                        st.error("Inference failed: %s" % (e,))
                        parsed = None
                        raw_text = ""

                # If Scene prompt was used, it may not return JSON. In that case, run a second
                # pass with the Gauge JSON prompt so we can still render cards + overlay.
                scene_prompt_used = prompt.strip() == DEFAULT_PROMPT_SCENE.strip()
                raw_scene_text = raw_text if scene_prompt_used else ""
                if scene_prompt_used and parsed is None:
                    with st.spinner("Building structured gauge JSON for results…"):
                        try:
                            raw_json_text, parsed_json = infer_with_custom_prompt(
                                image_bytes=image_bytes,
                                mime=mime,
                                api_key=api_key or os.environ.get("GEMINI_API_KEY_1"),
                                prompt=DEFAULT_PROMPT_GAUGE_JSON,
                                system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
                                model_id=MODEL_ID,
                                thinking_budget=thinking_budget if use_thinking else None,
                                temperature=temperature,
                            )
                            raw_text = raw_json_text
                            parsed = parsed_json
                        except Exception as e:
                            st.error("Structured JSON pass failed: %s" % (e,))

                st.session_state[k_last_input] = image_bytes
                st.session_state[k_last_name] = selected_name
                st.session_state[k_last_raw] = raw_text
                st.session_state[k_last_raw_scene] = raw_scene_text
                st.session_state[k_last_parsed] = parsed
                st.session_state[k_last_overlay_err] = ""
                if parsed is not None:
                    try:
                        st.session_state[k_last_overlay] = render_labeled_image_with_role(
                            image_bytes,
                            parsed,
                            scale=annotation_scale,
                        )
                    except Exception as e:
                        st.session_state[k_last_overlay] = None
                        st.session_state[k_last_overlay_err] = str(e)

    with right:
        st.markdown('<p class="section-label">Result</p>', unsafe_allow_html=True)
        with st.container(border=True):
            last_input = st.session_state.get(k_last_input)
            last_name = st.session_state.get(k_last_name, "input")
            last_overlay = st.session_state.get(k_last_overlay)
            last_overlay_err = st.session_state.get(k_last_overlay_err, "")
            last_raw = st.session_state.get(k_last_raw, "")
            last_raw_scene = st.session_state.get(k_last_raw_scene, "")
            last_parsed = st.session_state.get(k_last_parsed)

            if last_input is None:
                st.info("Run an inspection to see results here.")
            else:
                tabs = ["Annotated Result"]
                if last_raw:
                    tabs.append("Raw Text")
                t_out, *rest = st.tabs(tabs)
                with t_out:
                    if last_parsed is None or not last_overlay:
                        st.warning("No annotated result available.")
                        if last_overlay_err:
                            st.error(f"Overlay render error: {last_overlay_err}")
                    else:
                        st.image(last_overlay, caption="Annotated result", width=700)
                        st.download_button(
                            "Download annotated PNG",
                            data=last_overlay,
                            file_name="annotated_with_role.png",
                            mime="image/png",
                            key=f"{key_prefix}download_annotated",
                        )
                if rest:
                    with rest[0]:
                        if last_raw_scene:
                            st.text_area("Scene prompt output", value=last_raw_scene, height=180, key=f"{key_prefix}raw_scene_text")
                            st.text_area("Structured JSON pass output", value=last_raw, height=220, key=f"{key_prefix}raw_text")
                        else:
                            st.text_area("Raw response", value=last_raw, height=360, key=f"{key_prefix}raw_text")

                if isinstance(last_parsed, dict):
                    st.divider()
                    if last_raw_scene:
                        st.markdown('<p class="section-label">Scene Description</p>', unsafe_allow_html=True)
                        render_scene_description_card(last_raw_scene, last_parsed)
                        st.markdown('<p class="section-label">Structured Results</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="section-label">Gauge Results</p>', unsafe_allow_html=True)
                    render_gauge_results_cards(last_parsed)
                    with st.expander("Full JSON", expanded=False):
                        st.code(json.dumps(last_parsed, indent=2, ensure_ascii=False), language="json")
                


def main() -> None:
    render_app(embedded=False)


if __name__ == "__main__":
    main()
