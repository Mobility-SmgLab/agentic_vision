#!/usr/bin/env python3
"""
Evaluate Gemini Robotics-ER 1.6 for analog gauge / dial reading (e.g. Go2 inspection camera).

Model: gemini-robotics-er-1.6-preview
Docs: https://ai.google.dev/gemini-api/docs/robotics-overview
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_ID = "gemini-robotics-er-1.6-preview"

GAUGE_PROMPT = """You are the vision module on a Unitree Go2 quadruped doing facility inspection.

Read every analog gauge or dial visible (pressure, vacuum, temperature, level, etc.).
Use the same style as Gemini Robotics spatial outputs: normalized integer coordinates in [y, x] with range 0-1000.

Return ONLY valid JSON (no markdown fences, no commentary) with this shape:
{
  "scene_summary": "<one sentence what the panel/device is>",
  "points": [
    {"point": [y, x], "label": "<short label including gauge name and approximate reading + unit if known>"}
  ],
  "gauges": [
    {
      "role": "<e.g. TANK, OUTLET, OIL, or unknown>",
      "reading": { "value": <number or null>, "unit": "<string or null>", "approximate": <true|false> },
      "needle_tip_point": [y, x],
      "dial_center_point": [y, x],
      "dial_bbox_2d": [ymin, xmin, ymax, xmax],
      "notes": "<optional clarifications, glare, occlusion>"
    }
  ]
}

Rules:
- needle_tip_point = yellow/colored needle tip; dial_center_point = pivot/center of the same dial.
- dial_bbox_2d encloses the circular dial face only, integers, normalized 0-1000 like the API docs for box_2d.
- If a value cannot be read confidently, set reading.value to null and approximate true.
- Zoom mentally on small needles; prefer consistency with major tick marks.
"""


def _mime_for_path(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    suf = path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suf == ".png":
        return "image/png"
    if suf == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _parse_json_object(text: str) -> dict:
    cleaned = _strip_json_fences(text)
    return json.loads(cleaned)


def _extract_first_json_object(text: str) -> dict:
    """Best-effort when code-execution or model emits extra prose."""
    cleaned = _strip_json_fences(text)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : i + 1])
    raise ValueError("Unbalanced JSON braces in model output.")


def read_gauges(
    image_path: Path,
    *,
    thinking_budget: int | None = 0,
    temperature: float = 0.4,
    use_code_execution: bool = False,
) -> tuple[dict, str]:
    load_dotenv()
    client = genai.Client()

    image_bytes = image_path.read_bytes()
    mime = _mime_for_path(image_path)

    tools = None
    if use_code_execution:
        tools = [types.Tool(code_execution=types.ToolCodeExecution())]

    thinking_cfg = None
    if thinking_budget is not None:
        thinking_cfg = types.ThinkingConfig(thinking_budget=thinking_budget)

    # Code execution runs may include logs; avoid forcing JSON MIME so we can brace-extract.
    response_mime = None if use_code_execution else "application/json"

    cfg = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=(
            "Be precise. When JSON is requested, reply with ONLY that JSON "
            "(no preface, no code fences)."
        ),
        thinking_config=thinking_cfg,
        response_mime_type=response_mime,
        tools=tools,
    )

    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            GAUGE_PROMPT,
        ],
        config=cfg,
    )

    raw = (resp.text or "").strip()
    if not raw:
        raise RuntimeError("Empty model response.")

    try:
        if use_code_execution:
            data = _extract_first_json_object(raw)
        else:
            data = _parse_json_object(raw)
    except (json.JSONDecodeError, ValueError):
        data = _extract_first_json_object(raw)

    return data, raw


def _get_pil_font(size: int):
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_labeled_image(image_path: Path, data: dict, output_path: Path) -> None:
    """
    Overlay dial_bbox_2d, needle/center points, and text from model JSON.
    Coordinates follow Gemini robotics docs: [y, x] and box [ymin, xmin, ymax, xmax] in 0-1000.
    """
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = _get_pil_font(max(16, min(w, h) // 45))
    font_small = _get_pil_font(max(12, min(w, h) // 55))

    summary = (data.get("scene_summary") or "").strip()
    if summary:
        draw.text(
            (10, 10),
            summary[:140],
            fill=(220, 235, 255),
            font=font_small,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    palette = ((0, 255, 200), (255, 200, 0), (120, 200, 255))
    gauges = data.get("gauges") or []
    for i, g in enumerate(gauges):
        color_box = palette[i % len(palette)]
        x0 = y0 = x1 = y1 = None
        bbox = g.get("dial_bbox_2d")
        if bbox and len(bbox) == 4:
            ymin, xmin, ymax, xmax = (int(v) for v in bbox)
            x0 = int(xmin / 1000.0 * w)
            y0 = int(ymin / 1000.0 * h)
            x1 = int(xmax / 1000.0 * w)
            y1 = int(ymax / 1000.0 * h)
            draw.rectangle([x0, y0, x1, y1], outline=color_box, width=max(2, min(w, h) // 400))

        for pt, col in (
            (g.get("needle_tip_point"), (255, 60, 60)),
            (g.get("dial_center_point"), (60, 140, 255)),
        ):
            if not pt or len(pt) != 2:
                continue
            py, px = int(pt[0] / 1000.0 * h), int(pt[1] / 1000.0 * w)
            r = max(4, min(w, h) // 120)
            draw.ellipse([px - r, py - r, px + r, py + r], outline=col, width=2)

        rd = g.get("reading") or {}
        val = rd.get("value")
        unit = (rd.get("unit") or "").strip()
        approx = bool(rd.get("approximate"))
        val_s = "?" if val is None else str(val)
        line1 = ("%s %s" % (val_s, unit)).strip()
        if approx:
            line1 += " ~"

        if x0 is not None:
            ty = max(32, y0 - 8 - 22)
            tx = max(8, min(x0, w - 8))
        else:
            pt = g.get("needle_tip_point")
            if not pt or len(pt) != 2:
                continue
            py, px = int(pt[0] / 1000.0 * h), int(pt[1] / 1000.0 * w)
            ty, tx = max(32, py - 28), max(8, min(px, w - 8))

        draw.text(
            (tx, ty),
            line1[:90],
            fill=(255, 255, 120),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    for p in data.get("points") or []:
        pt = p.get("point")
        lbl = (p.get("label") or "").strip()[:72]
        if not pt or len(pt) != 2 or not lbl:
            continue
        py, px = int(pt[0] / 1000.0 * h), int(pt[1] / 1000.0 * w)
        r = max(3, min(w, h) // 150)
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 255), width=2)
        lx = min(px + 10, max(8, w - 220))
        draw.text(
            (lx, max(40, py - 12)),
            lbl,
            fill=(255, 210, 255),
            font=font_small,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")


def main() -> int:
    default_img = Path(__file__).resolve().parent / "static" / "WhatsApp Image 2026-04-14 at 8.44.02 PM.jpeg"

    p = argparse.ArgumentParser(description="Read analog gauges with Gemini Robotics-ER 1.6 preview.")
    p.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=default_img if default_img.is_file() else None,
        help="Path to image (jpeg/png/webp). Defaults to bundled sample if present.",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="0 = fast spatial-style path; increase for harder scenes (see robotics overview).",
    )
    p.add_argument(
        "--no-thinking-config",
        action="store_true",
        help="Omit thinking_config entirely (SDK default).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.4,
    )
    p.add_argument(
        "--code-execution",
        action="store_true",
        help="Enable code execution tool (zoom/crop style workflows; higher latency).",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw model string before parsed JSON.",
    )
    p.add_argument(
        "--out-image",
        type=Path,
        default=None,
        help="Write an annotated PNG (boxes, needle/center markers, labels) to this path.",
    )
    p.add_argument(
        "--save-annotated",
        action="store_true",
        help="Write <input_stem>_gauges_er16.png next to the input image (ignored if --out-image is set).",
    )
    args = p.parse_args()

    if args.image is None:
        print("No image path provided and default sample not found.", file=sys.stderr)
        return 2

    path = args.image.expanduser().resolve()
    if not path.is_file():
        print("Image not found:", path, file=sys.stderr)
        return 2

    tb = None if args.no_thinking_config else args.thinking_budget

    try:
        data, raw = read_gauges(
            path,
            thinking_budget=tb,
            temperature=args.temperature,
            use_code_execution=args.code_execution,
        )
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        return 1

    if args.raw:
        print(raw)
        print("---")

    print(json.dumps(data, indent=2, ensure_ascii=False))

    out_image = args.out_image
    if out_image is None and args.save_annotated:
        out_image = path.parent / ("%s_gauges_er16.png" % (path.stem,))

    if out_image is not None:
        try:
            render_labeled_image(path, data, out_image)
        except Exception as e:
            print("Could not write annotated image:", e, file=sys.stderr)
            return 1
        print("Wrote annotated image:", str(out_image.resolve()), file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
