#!/usr/bin/env python
"""Compose Figure 2 as ONE combined figure — panel (a) on the left, panel (b) on
the right — from the paper's two readings of the SAME place graph (nodes n1..n6):

  - (a) ← fig02a-hyperedges.png  (Milner link graph — hyperedges e1, e2, e3)
  - (b) ← fig02b-processes.png   (process graph — processes p1, p2, p3)

Both panels are scaled to a common content height so they sit level side by side,
each carrying its own letter label. A single `figure_2` SVG + PNG is emitted.

Panels must already be rendered (loom PNGs via scripts/render_loom_svgs.mjs).

    python scripts/build_figure2.py
"""
from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WS = Path(__file__).resolve().parents[1]
VIZ = WS / "studies" / "fig-02" / "visualizations"

# (panel png, letter) — laid out left→right in this order.
PANELS = [
    ("fig02a-hyperedges.png", "a"),
    ("fig02b-processes.png",  "b"),
]
# One combined culminating visualization for the study (was two separate 2a/2b).
STITCHED_OUTPUTS = [
    ("figure_2", "Figure 2 (hypergraph + process bigraph)"),
]
OUTPUT_STEM = "figure_2"
PANEL_H = 620      # common content height both panels are scaled to
PAD = 28
GAP = 64           # horizontal gap between the two panels
LABEL_H = 52       # space above the panels for their letters

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


def _panel_size(png: Path) -> tuple[float, float]:
    """Width, height when scaled to the common PANEL_H."""
    with Image.open(png) as im:
        w, h = im.size
    scale = PANEL_H / h
    return w * scale, float(PANEL_H)


def _layout() -> tuple[list[tuple[Path, str, float, float, float]], float, float]:
    """Return (placements, fig_w, fig_h). Each placement is
    (png, letter, x, width, height)."""
    placements: list[tuple[Path, str, float, float, float]] = []
    x = float(PAD)
    for i, (panel, letter) in enumerate(PANELS):
        png = VIZ / panel
        if not png.is_file():
            raise SystemExit(f"missing Figure 2 panel: {png} — render the panels first")
        w, h = _panel_size(png)
        placements.append((png, letter, x, w, h))
        x += w + (GAP if i < len(PANELS) - 1 else 0)
    fig_w = x + PAD
    fig_h = PANEL_H + LABEL_H + PAD
    return placements, fig_w, fig_h


def build_figure2() -> Path:
    VIZ.mkdir(parents=True, exist_ok=True)
    placements, fig_w, fig_h = _layout()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_w:.0f}" height="{fig_h:.0f}" '
        f'viewBox="0 0 {fig_w:.0f} {fig_h:.0f}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for png, letter, x, w, h in placements:
        parts.append(
            f'<text x="{x:.0f}" y="{PAD + 34:.0f}" '
            f'font-family="Georgia, \'Times New Roman\', serif" '
            f'font-size="34" font-weight="bold" fill="#111827">{letter}.</text>'
        )
        parts.append(
            f'<image href="{_data_uri(png)}" x="{x:.0f}" y="{LABEL_H:.0f}" '
            f'width="{w:.0f}" height="{h:.0f}" preserveAspectRatio="xMidYMid meet"/>'
        )
    parts.append("</svg>")
    out = VIZ / f"{OUTPUT_STEM}.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"composed {out.relative_to(WS)}")
    return out


def build_figure2_png() -> Path:
    placements, fig_w, fig_h = _layout()
    canvas = Image.new("RGB", (round(fig_w), round(fig_h)), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    font = _label_font(34)
    for png, letter, x, w, h in placements:
        draw.text((x, PAD - 4), f"{letter}.", fill="#111827", font=font)
        with Image.open(png) as im:
            panel = im.convert("RGBA").resize((round(w), round(h)), Image.LANCZOS)
            canvas.paste(panel, (round(x), LABEL_H), panel)
    out = VIZ / f"{OUTPUT_STEM}.png"
    canvas.save(out, "PNG", optimize=True)
    print(f"composed {out.relative_to(WS)}")
    return out


if __name__ == "__main__":
    build_figure2()
    build_figure2_png()
