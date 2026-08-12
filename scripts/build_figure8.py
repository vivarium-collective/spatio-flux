#!/usr/bin/env python
"""Compose Figure 8 — the spatio-flux reference model — as a VERTICAL stack:

    a   reference-model composite (loom)      ← full width, on top
    b   reference-model field/particle snapshots over time

Each row is justified to a common content width (so both panels fill the column),
with panel a stacked ABOVE panel b rather than side-by-side. Mirrors the
build_figure1 / build_figure7 scaffold-compose pattern.

Panels must already be rendered (fig08-reference-model.png via the loom render;
fig08b-reference-snapshots.png via scripts/build_fig08b_snapshots.py).

    python scripts/build_figure8.py
"""
from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WS = Path(__file__).resolve().parents[1]
VIZ = WS / "studies" / "fig-08" / "visualizations"

# One panel per row → a stacked vertically over b.
ROWS = [
    ["fig08-reference-model.png"],       # a
    ["fig08b-reference-snapshots.png"],  # b
]
CONTENT_W = 2200   # content width every row justifies to
PAD = 28           # outer margin
GAP = 22           # gap between rows
LABEL_H = 46       # space above each row for its panel letter
# Per-row height caps (row index -> max px). A capped row is centered instead of
# stretched, keeping the tall reference-model schematic from dominating the page.
ROW_MAX_H = {0: 1040}

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


def _sizes():
    out = {}
    for row in ROWS:
        for f in row:
            p = VIZ / f
            if not p.is_file():
                raise SystemExit(f"missing Figure 8 panel: {p} — render the panels first")
            with Image.open(p) as im:
                out[f] = im.size
    return out


def _layout():
    """Per-panel placement (path, x, y, w, h) + canvas size; one panel per row,
    each justified to CONTENT_W (capped rows centered)."""
    sizes = _sizes()
    placements = []
    y = float(PAD)
    for ri, row in enumerate(ROWS):
        aspects = [sizes[f][0] / sizes[f][1] for f in row]
        gaps = (len(row) - 1) * GAP
        h = (CONTENT_W - gaps) / sum(aspects)
        widths = [a * h for a in aspects]
        row_w = sum(widths) + gaps
        cap = ROW_MAX_H.get(ri)
        if cap is not None and h > cap:
            h = float(cap)
            widths = [a * h for a in aspects]
            row_w = sum(widths) + gaps
        x = PAD + (CONTENT_W - row_w) / 2.0
        y += LABEL_H
        for f, w in zip(row, widths):
            placements.append((VIZ / f, x, y, w, h))
            x += w + GAP
        y += h + GAP
    fig_w = CONTENT_W + 2 * PAD
    fig_h = y - GAP + PAD
    return placements, fig_w, fig_h


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_figure8() -> Path:
    placements, fig_w, fig_h = _layout()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_w:.0f}" height="{fig_h:.0f}" '
        f'viewBox="0 0 {fig_w:.0f} {fig_h:.0f}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for i, (p, x, y, w, h) in enumerate(placements):
        label = chr(ord("a") + i)
        parts.append(
            f'<text x="{x:.0f}" y="{y - 12:.0f}" font-family="Georgia, \'Times New Roman\', serif" '
            f'font-size="34" font-weight="bold" fill="#111827">{label}.</text>'
        )
        parts.append(
            f'<image href="{_data_uri(p)}" x="{x:.0f}" y="{y:.0f}" '
            f'width="{w:.0f}" height="{h:.0f}" preserveAspectRatio="xMidYMid meet"/>'
        )
    parts.append("</svg>")
    VIZ.mkdir(parents=True, exist_ok=True)
    out = VIZ / "figure_8.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"composed {out.relative_to(WS)} — {len(placements)} panels stacked (a over b)")
    return out


def build_figure8_png() -> Path:
    placements, fig_w, fig_h = _layout()
    canvas = Image.new("RGB", (round(fig_w), round(fig_h)), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    font = _label_font(34)
    for i, (p, x, y, w, h) in enumerate(placements):
        draw.text((round(x), round(y - 44)), f"{chr(ord('a') + i)}.", fill="#111827", font=font)
        with Image.open(p) as im:
            panel = im.convert("RGBA").resize((round(w), round(h)), Image.LANCZOS)
            canvas.paste(panel, (round(x), round(y)), panel)
    out = VIZ / "figure_8.png"
    canvas.save(out, "PNG", optimize=True)
    return out


if __name__ == "__main__":
    build_figure8()
    build_figure8_png()
