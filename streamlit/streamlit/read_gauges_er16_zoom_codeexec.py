#!/usr/bin/env python3
"""
Gauge reading with Gemini Robotics-ER 1.6 + code execution (zoom/crop in Python), then JSON.

Follows the pattern in:
https://ai.google.dev/gemini-api/docs/robotics-overview
("Read an analog gauge", fluid meter with code execution, zoom/crop examples.)

This script always enables code execution. The model may run Python to magnify dials
before emitting the same JSON schema as read_gauges_er16.py.

Annotated PNG overlay still uses Pillow locally (see read_gauges_er16.render_labeled_image).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from read_gauges_er16 import (
    MODEL_ID,
    _extract_first_json_object,
    _mime_for_path,
    render_labeled_image,
)

# Relaxed vs read_gauges_er16.py: allow tool use and intermediate reasoning; JSON is still required as the final payload.
SYSTEM_INSTRUCTION = """You may run Python in the code execution tool to zoom, crop, rotate, pad, or otherwise inspect subregions of the image so tick marks, printed units, and needle tips are easy to see.

When you are finished, your final assistant message must be exactly one JSON object matching the schema the user gave (raw JSON only: no markdown code fences, no prose before or after the object)."""

ZOOM_GAUGE_PROMPT = """You are the vision module on a Unitree Go2 quadruped doing facility inspection.

Task: read every analog gauge or dial (pressure, vacuum, temperature, level, etc.).

Step 1 — Use code execution as needed (same spirit as the Gemini Robotics cookbook):
- Load the image in the Python environment, then zoom/crop (or build tight crops) around each gauge face so major ticks, minor ticks, printed numbers, and the needle tip are legible.
- You may iterate: coarse localization, then a tighter crop, small rotations if the dial is skewed.

Step 2 — After inspection, emit ONE JSON object only (no markdown fences, no commentary) with this exact shape:
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

Spatial rules (Gemini Robotics style, relative to the ORIGINAL full-frame input image):
- All coordinates are integers in [0, 1000].
- Points use [y, x].
- dial_bbox_2d is [ymin, xmin, ymax, xmax] enclosing the circular dial face.
- needle_tip_point = needle tip; dial_center_point = pivot.
- Even if you cropped in Python for analysis, report bbox/points in the ORIGINAL image coordinate system (map back from crops if you used them).
"""


def read_gauges_zoom_codeexec(
    image_path: Path,
    *,
    thinking_budget: int | None = 8192,
    temperature: float = 0.3,
) -> tuple[dict, str]:
    load_dotenv()
    client = genai.Client()

    image_bytes = image_path.read_bytes()
    mime = _mime_for_path(image_path)

    thinking_cfg = None
    if thinking_budget is not None:
        thinking_cfg = types.ThinkingConfig(thinking_budget=thinking_budget)

    cfg = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=SYSTEM_INSTRUCTION,
        thinking_config=thinking_cfg,
        tools=[types.Tool(code_execution=types.ToolCodeExecution())],
    )

    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            ZOOM_GAUGE_PROMPT,
        ],
        config=cfg,
    )

    raw = (resp.text or "").strip()
    if not raw:
        raise RuntimeError("Empty model response (no text).")

    data = _extract_first_json_object(raw)
    return data, raw


def main() -> int:
    default_img = Path(__file__).resolve().parent / "static"

    p = argparse.ArgumentParser(
        description="Read analog gauges with Gemini Robotics-ER 1.6 + code execution (zoom/crop), then JSON."
    )
    p.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=default_img if default_img.is_file() else None,
        help="Path to image (jpeg/png/webp).",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=8192,
        help="Thinking budget for harder visual reasoning (default 8192). Use 0 for minimal.",
    )
    p.add_argument(
        "--no-thinking-config",
        action="store_true",
        help="Omit thinking_config entirely.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.3,
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw model text before parsed JSON.",
    )
    p.add_argument(
        "--out-image",
        type=Path,
        default=None,
        help="Write annotated PNG to this path.",
    )
    p.add_argument(
        "--save-annotated",
        action="store_true",
        help="Write <input_stem>_gauges_er16_codeexec.png next to the input (ignored if --out-image is set).",
    )
    args = p.parse_args()

    if args.image is None:
        print("No image path provided and no default sample found.", file=sys.stderr)
        return 2

    path = args.image.expanduser().resolve()
    if not path.is_file():
        print("Image not found:", path, file=sys.stderr)
        return 2

    tb = None if args.no_thinking_config else args.thinking_budget

    try:
        data, raw = read_gauges_zoom_codeexec(
            path,
            thinking_budget=tb,
            temperature=args.temperature,
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
        out_image = path.parent / ("%s_gauges_er16_codeexec.png" % (path.stem,))

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
