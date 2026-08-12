#!/usr/bin/env python
"""Compose Figure 2 as TWO SEPARATE figures — figure_2a and figure_2b — rather
than one combined a/b panel.

Both read the SAME process-bigraph composite (n1..n6 + hyperedges e1/e2/e3):
  - figure_2a  ← fig02a-hyperedges.png  (the Milner link-graph / hypergraph reading)
  - figure_2b  ← fig02b-processes.png   (the process-graph reading)

Each is emitted standalone (its own SVG + PNG, letter label top-left) so they can
be placed as independent Fig 2a / Fig 2b in the paper.

Panels must already be rendered (loom PNGs via scripts/render_loom_svgs.mjs).

    python scripts/build_figure2.py
"""
from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WS = Path(__file__).resolve().parents[1]
VIZ = WS / "studies" / "fig-02" / "visualizations"

# (panel png, letter, output stem)
PANELS = [
    ("fig02a-hyperedges.png", "a", "figure_2a"),
    ("fig02b-processes.png",  "b", "figure_2b"),
]
# Declared so the stitch step registers BOTH figures as the study's culminating
# visualizations (instead of the default single figure_<N>).
STITCHED_OUTPUTS = [
    ("figure_2a", "Figure 2a (hypergraph)"),
    ("figure_2b", "Figure 2b (process bigraph)"),
]
CONTENT_W = 900    # target panel width
PAD = 28
LABEL_H = 52       # space above the panel for its letter

_LABEL_FONTS = (
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/Library/Fonts/Georgia Bold.ttf",
    "DejaVuSans-Bold.ttf",
)


def _label_font(size: int):
    for name in _LABEL_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _dims(png: Path):
    with Image.open(png) as im:
        w, h = im.size
    scale = CONTENT_W / w
    return CONTENT_W, h * scale


def _emit_svg(png: Path, letter: str, stem: str) -> Path:
    cw, ch = _dims(png)
    fig_w = cw + 2 * PAD
    fig_h = ch + LABEL_H + PAD
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_w:.0f}" height="{fig_h:.0f}" '
        f'viewBox="0 0 {fig_w:.0f} {fig_h:.0f}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{PAD:.0f}" y="{PAD + 34:.0f}" font-family="Georgia, \'Times New Roman\', serif" '
        f'font-size="34" font-weight="bold" fill="#111827">{letter}.</text>',
        f'<image href="{_data_uri(png)}" x="{PAD:.0f}" y="{LABEL_H:.0f}" '
        f'width="{cw:.0f}" height="{ch:.0f}" preserveAspectRatio="xMidYMid meet"/>',
        "</svg>",
    ]
    out = VIZ / f"{stem}.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def _emit_png(png: Path, letter: str, stem: str) -> Path:
    cw, ch = _dims(png)
    fig_w = round(cw + 2 * PAD)
    fig_h = round(ch + LABEL_H + PAD)
    canvas = Image.new("RGB", (fig_w, fig_h), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, PAD - 4), f"{letter}.", fill="#111827", font=_label_font(34))
    with Image.open(png) as im:
        panel = im.convert("RGBA").resize((round(cw), round(ch)), Image.LANCZOS)
        canvas.paste(panel, (PAD, LABEL_H), panel)
    out = VIZ / f"{stem}.png"
    canvas.save(out, "PNG", optimize=True)
    return out


def build_figure2() -> Path:
    VIZ.mkdir(parents=True, exist_ok=True)
    first = None
    for panel, letter, stem in PANELS:
        png = VIZ / panel
        if not png.is_file():
            raise SystemExit(f"missing Figure 2 panel: {png} — render the panels first")
        out = _emit_svg(png, letter, stem)
        print(f"composed {out.relative_to(WS)}")
        first = first or out
    return first  # framework checks a returned Path exists


def build_figure2_png() -> Path:
    first = None
    for panel, letter, stem in PANELS:
        out = _emit_png(VIZ / panel, letter, stem)
        first = first or out
    return first


if __name__ == "__main__":
    build_figure2()
    build_figure2_png()
