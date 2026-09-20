#!/usr/bin/env python
"""Compose Figure 1 (panels a / b / c) — publication-ready.

The a / b / c illustrations are the study's loom-rendered panel PNGs
(``studies/fig-01/visualizations/fig01{a,b,c}-*.png``). This script draws a
fresh, uniform header (letter badge + accent-coloured title + subtitle) over
each panel and scales the illustration to fill the card, then writes
``studies/fig-01/visualizations/figure_1.svg``.

Headers are drawn here (not kept from the scaffold) so title/subtitle text,
typography and spacing are fully controlled. Edit ``PANELS`` to change copy or
colour; edit the layout constants to change size.

    python scripts/build_figure1.py

The loom panel PNGs must already be rendered (they are committed under
studies/fig-01/visualizations/); this only stitches.
"""
from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path

from PIL import Image

WS = Path(__file__).resolve().parents[1]
VIZ = WS / "studies" / "fig-01" / "visualizations"

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# ── Per-panel content + colour ───────────────────────────────────────────────
# loom: the illustration PNG. tint/stroke: the card. accent: badge + title.
PANELS = [
    {
        "id": "a",
        "loom": "fig01a-draft-processes.png",
        "tint": "#fdecea", "stroke": "#e6a8a0", "accent": "#d33a2c",
        "title": "Every subsystem gets its own best model",
        "subtitle": ("Domain experts model each mechanism in the formalism that "
                     "fits it — ODEs, FBA, PDEs, agent-based, learned dynamics — "
                     "each with its own variables and scale."),
    },
    {
        "id": "b",
        "loom": "fig01b-multiscale-composite.png",
        "tint": "#e8f0fe", "stroke": "#a9c4f5", "accent": "#2b5bd0",
        "title": "Process Bigraphs make coupling explicit",
        "subtitle": ("Typed interfaces and explicit wiring let independent models "
                     "share state and run across scales as one system."),
    },
    {
        "id": "c",
        "loom": "fig01c-study-workflow.png",
        "tint": "#fef7e0", "stroke": "#e6d9a8", "accent": "#2f6b3f",
        "title": "Compositions become reusable simulations",
        "subtitle": ("A declarative spec runs on a shared engine — executable, "
                     "testable, shareable, and recomposable."),
    },
]

# ── Layout (all in SVG user units) ──────────────────────────────────────────
PANEL_W = 500          # card width (was 320) — wider cards → larger, readable illustrations
SIDE = 14              # inner L/R margin around the illustration
GAP = 30               # between cards
MARGIN = 26            # outer figure margin
BOTTOM = 18            # inner margin below the illustration
CARD_RX = 18           # card corner radius

PAD = 26               # header inner padding (badge + text inset)
BADGE = 34             # letter-badge square
TITLE_FS = 25          # title font size (was 14)
TITLE_LH = 30          # title line height
SUB_FS = 15            # subtitle font size (was ~11)
SUB_LH = 20            # subtitle line height
TITLE_GAP = 12         # gap between title block and subtitle
HEADER_GAP = 16        # gap between subtitle block and the illustration


def _q(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def _data_uri(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii")


def _aspect(png: Path) -> float:
    with Image.open(png) as im:
        w, h = im.size
    return w / h


def _wrap(text: str, max_w: float, fs: float, bold: bool) -> list[str]:
    """Greedy word-wrap using an average glyph-width estimate for Helvetica."""
    char_w = fs * (0.60 if bold else 0.53)
    max_chars = max(8, int(max_w / char_w))
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) <= max_chars or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _header_lines(panel: dict, text_w: float):
    title = _wrap(panel["title"], text_w, TITLE_FS, bold=True)
    sub = _wrap(panel["subtitle"], text_w, SUB_FS, bold=False)
    return title, sub


def _header_height(title_lines, sub_lines) -> float:
    return (PAD + max(BADGE, TITLE_LH * len(title_lines))
            + TITLE_GAP + SUB_LH * len(sub_lines) + HEADER_GAP)


def _draw_header(panel: dict, title_lines, sub_lines, header_h: float, cx: float) -> str:
    accent, sub_fill = panel["accent"], "#42474d"
    # letter badge — top-left
    by = PAD
    parts = [
        f'<rect x="{PAD}" y="{by}" width="{BADGE}" height="{BADGE}" rx="9" fill="{accent}"/>',
        f'<text x="{PAD + BADGE/2:.1f}" y="{by + BADGE/2 + 6.5:.1f}" text-anchor="middle" '
        f'font-size="19" font-weight="700" fill="#ffffff">{panel["id"]}</text>',
    ]
    # title — accent-coloured, centred, starts level with the badge
    ty = by + TITLE_FS + 1
    for i, line in enumerate(title_lines):
        parts.append(
            f'<text x="{cx:.1f}" y="{ty + i*TITLE_LH:.1f}" text-anchor="middle" '
            f'font-size="{TITLE_FS}" font-weight="700" fill="{accent}">{escape(line)}</text>')
    # subtitle — muted, centred, below the title block
    sy = by + max(BADGE, TITLE_LH * len(title_lines)) + TITLE_GAP + SUB_FS
    for i, line in enumerate(sub_lines):
        parts.append(
            f'<text x="{cx:.1f}" y="{sy + i*SUB_LH:.1f}" text-anchor="middle" '
            f'font-size="{SUB_FS}" font-weight="400" fill="{sub_fill}">{escape(line)}</text>')
    return "".join(parts)


def build_figure1() -> Path:
    cw = PANEL_W - 2 * SIDE          # illustration width inside a card
    text_w = PANEL_W - 2 * PAD       # header text wrap width
    cx = PANEL_W / 2                 # card centre (for centred header text)

    # Pre-pass: wrap headers, measure image + header heights, size all cards equal.
    prepared = []
    header_h = 0.0
    for p in PANELS:
        png = VIZ / p["loom"]
        if not png.is_file():
            raise SystemExit(f"missing loom panel PNG: {png} — render the panels first")
        title_lines, sub_lines = _header_lines(p, text_w)
        header_h = max(header_h, _header_height(title_lines, sub_lines))
        img_h = round(cw / _aspect(png))
        prepared.append((p, png, title_lines, sub_lines, img_h))

    panel_h = round(header_h + max(ih for *_, ih in prepared) + BOTTOM)

    # Build the SVG.
    n = len(prepared)
    fig_w = MARGIN + n * PANEL_W + (n - 1) * GAP + MARGIN
    fig_h = MARGIN + panel_h + MARGIN
    root = ET.Element(_q("svg"), {
        "width": str(fig_w), "height": str(fig_h),
        "viewBox": f"0 0 {fig_w} {fig_h}", "font-family": FONT,
    })
    ET.SubElement(root, _q("rect"), {"x": "0", "y": "0", "width": str(fig_w),
                                     "height": str(fig_h), "fill": "#ffffff"})

    x = MARGIN
    for p, png, title_lines, sub_lines, img_h in prepared:
        g = ET.SubElement(root, _q("g"), {"id": f"Panel {p['id']}",
                                          "transform": f"translate({x},{MARGIN})"})
        ET.SubElement(g, _q("rect"), {
            "x": "0", "y": "0", "width": str(PANEL_W), "height": str(panel_h),
            "rx": str(CARD_RX), "fill": p["tint"], "stroke": p["stroke"],
            "stroke-width": "1.4"})
        g.append(ET.fromstring(
            f'<g xmlns="{SVG_NS}">' + _draw_header(p, title_lines, sub_lines, header_h, cx) + "</g>"))
        img = ET.SubElement(g, _q("image"))
        img.set("href", _data_uri(png))
        img.set("x", str(SIDE)); img.set("y", f"{header_h:.1f}")
        img.set("width", str(cw)); img.set("height", str(img_h))
        img.set("preserveAspectRatio", "xMidYMin meet")
        x += PANEL_W + GAP

    out = VIZ / "figure_1.svg"
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    print(f"composed {out.relative_to(WS)} — {n} panels, {fig_w}x{fig_h}, header {header_h:.0f}px")
    return out


if __name__ == "__main__":
    build_figure1()
