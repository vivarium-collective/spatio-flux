#!/usr/bin/env python
"""Compose Figure 7 in the paper's fixed 4-region layout (Spatio-Flux basic
components + behaviors), rather than the default auto-wrap shelf.

Target layout (matches the paper's Fig 6):

    a                                   ← community-dFBA loom, full width
    b        c        d                 ← Monod / dFBA / hybrid-community sims
    e                 f                  ← COMETS loom | COMETS field snapshots
    g                 h                  ← Brownian loom | particle traces

a / e / g are the bigraph-loom wiring diagrams; b / c / d / f / h are simulation
outputs. Rows are laid out *justified* — each row is scaled so its panels fill
the same content width — which removes the ragged right-edge whitespace the shelf
layout left. Panel a (naturally squarish from auto-layout) is height-capped so it
doesn't dominate the page; when capped it is centered in the content width.

Panels must already be rendered (loom PNGs via scripts/render_loom_svgs.mjs; the
sim PNGs are the study's simulation outputs). This only stitches. Writes
studies/fig-07/visualizations/figure_7.svg (+ .png twin via build_paper_figures).

    python scripts/build_figure7.py
"""
from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WS = Path(__file__).resolve().parents[1]
VIZ = WS / "studies" / "fig-07" / "visualizations"

# Row groups, in panel-label order (a, b, c, …). Each inner list is one row.
ROWS = [
    ["fig07-1-community-dfba.png"],                                       # a
    ["fig07b-monod-kinetics.png", "fig07c-dfba.png",
     "fig07d-hybrid-community.png"],                                      # b c d
    ["fig07-2-comets.png", "fig07f-comets-snapshots.png"],               # e f
    ["fig07-3-brownian-particles.png", "fig07h-brownian-traces.png"],    # g h
]

CONTENT_W = 2200   # target content width every row justifies to
PAD = 28           # outer margin
GAP = 12           # gap between panels and between rows (tight — squeeze rows)
LABEL_H = 26       # space above each row for its panel letters
# Per-row height caps (row index -> max px). A capped row is centered in the
# content width instead of stretched; an uncapped (or generously-capped) row
# justifies to fill CONTENT_W. Row 0 = panel a (full-width loom), given a tall cap
# so it fills the top row; row 1 = the b/c/d simulation plots, cap raised past
# their natural justified height so they stretch to the full content width.
ROW_MAX_H = {0: 840, 1: 640, 2: 560}  # cap e/f (row 2) shorter too
# Tighter label band above the lower rows (pulls them up, squeezing the figure).
# Must stay big enough to clear the panel-letter (~28px for the 26px bold serif).
ROW_GAP_ABOVE = {1: 16, 2: 16, 3: 16}
# Panels rendered at a WIDER target aspect (width/height) than their source — the
# b / c / d timeseries plots, which read too tall. Their box is stretched to this
# aspect (preserveAspectRatio=none), so the plots come out a bit shorter + wider
# and the whole b/c/d row is shorter — the main lever for fitting Figure 7 on the
# page. Timeseries tolerate a modest horizontal stretch.
WIDEN_ASPECT = {
    "fig07b-monod-kinetics.png": 1.85,
    "fig07c-dfba.png": 1.85,
    "fig07d-hybrid-community.png": 1.85,
}
# Whole-row scale applied AFTER justification (row is centered at the reduced
# size). Row 3 = g / h (Brownian loom + traces) — shrunk further to shorten it.
ROW_SCALE = {3: 0.8}

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
    """Return {filename: (w, h)} for every panel, erroring on any missing file."""
    out = {}
    for row in ROWS:
        for f in row:
            p = VIZ / f
            if not p.is_file():
                raise SystemExit(f"missing Figure 7 panel: {p} — render the panels first")
            with Image.open(p) as im:
                out[f] = im.size
    return out


def _layout():
    """Compute per-panel placement (path, x, y, w, h) + total canvas size.

    Justified rows: at a common row height h, panel i has width aspect_i * h, so
    to fill CONTENT_W we solve h = (CONTENT_W - gaps) / sum(aspect). A row listed
    in ROW_MAX_H is height-capped and centered when the justified height would
    exceed its cap (keeps panel a from dominating and b/c/d from towering)."""
    sizes = _sizes()
    placements = []  # (path, x, y, w, h)
    y = float(PAD)
    for ri, row in enumerate(ROWS):
        # A widened panel justifies at its OVERRIDE aspect (so the row is shorter);
        # its box is later stretched to fill (preserveAspectRatio=none).
        aspects = [WIDEN_ASPECT.get(f, sizes[f][0] / sizes[f][1]) for f in row]
        gaps = (len(row) - 1) * GAP
        h = (CONTENT_W - gaps) / sum(aspects)
        widths = [a * h for a in aspects]
        row_w = sum(widths) + gaps
        cap = ROW_MAX_H.get(ri)
        if cap is not None and h > cap:  # cap this row's height; center it
            h = float(cap)
            widths = [a * h for a in aspects]
            row_w = sum(widths) + gaps
        scale = ROW_SCALE.get(ri, 1.0)
        if scale != 1.0:  # shrink the whole row (both panels) and re-center it
            h *= scale
            widths = [w * scale for w in widths]
            row_w = sum(widths) + gaps
        x = PAD + (CONTENT_W - row_w) / 2.0  # center (row_w == CONTENT_W unless capped/scaled)
        y += ROW_GAP_ABOVE.get(ri, LABEL_H)
        for f, w in zip(row, widths):
            placements.append((VIZ / f, x, y, w, h))
            x += w + GAP
        y += h + GAP
    fig_w = CONTENT_W + 2 * PAD
    fig_h = y - GAP + PAD
    return placements, fig_w, fig_h


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_figure7() -> Path:
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
        # Widened panels stretch to fill their (wider) box; the rest letterbox-fit.
        par = "none" if p.name in WIDEN_ASPECT else "xMidYMid meet"
        parts.append(
            f'<image href="{_data_uri(p)}" x="{x:.0f}" y="{y:.0f}" '
            f'width="{w:.0f}" height="{h:.0f}" preserveAspectRatio="{par}"/>'
        )
    parts.append("</svg>")
    VIZ.mkdir(parents=True, exist_ok=True)
    out = VIZ / "figure_7.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"composed {out.relative_to(WS)} — {len(placements)} panels in 4 justified rows")
    return out


def build_figure7_png() -> Path:
    placements, fig_w, fig_h = _layout()
    canvas = Image.new("RGB", (round(fig_w), round(fig_h)), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    font = _label_font(34)
    for i, (p, x, y, w, h) in enumerate(placements):
        draw.text((round(x), round(y - 44)), f"{chr(ord('a') + i)}.", fill="#111827", font=font)
        with Image.open(p) as im:
            panel = im.convert("RGBA").resize((round(w), round(h)), Image.LANCZOS)
            canvas.paste(panel, (round(x), round(y)), panel)
    out = VIZ / "figure_7.png"
    canvas.save(out, "PNG", optimize=True)
    return out


if __name__ == "__main__":
    build_figure7()
    build_figure7_png()
