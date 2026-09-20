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
        "hscale": 1.4,
        "loom": "fig01a-draft-processes.png",
        "tint": "#fdecea", "stroke": "#e6a8a0", "accent": "#d33a2c",
        "title": "Every subsystem gets its own best model",
        "subtitle": ("Domain experts model each mechanism in the formalism that "
                     "fits it — ODEs, FBA, PDEs, agent-based, learned dynamics — "
                     "each with its own variables and scale."),
    },
    {
        "id": "b",
        "hscale": 1.4,
        "loom": "fig01b-multiscale-composite.png",
        "tint": "#e8f0fe", "stroke": "#a9c4f5", "accent": "#2b5bd0",
        "title": "Process Bigraphs make coupling explicit",
        "subtitle": ("Typed interfaces and explicit wiring let independent models "
                     "share state and run across scales as one system."),
    },
    {
        "id": "c",
        "hscale": 1.4,
        "loom": "fig01c-study-workflow.png",
        "tint": "#fef7e0", "stroke": "#e6d9a8", "accent": "#2f6b3f",
        "title": "Compositions become reusable simulations",
        "subtitle": ("A declarative spec runs on a shared engine — executable, "
                     "testable, shareable, and recomposable."),
    },
]

# ── Layout (all in SVG user units) ──────────────────────────────────────────
IMG_H = 900            # common illustration height; each column's width follows its aspect
SIDE = 14              # inner L/R margin around the illustration
GAP = 30               # between cards
MARGIN = 26            # outer figure margin
BOTTOM = 18            # inner margin below the illustration
CARD_RX = 18           # card corner radius

# Header — editorial layout: an accent letter badge anchors the top-left, the
# title sits in a left-aligned column beside it (restrained dark ink, not a loud
# all-accent centered banner), a short accent rule carries the colour-coding, and
# the subtitle sits below in muted grey.
PAD = 28               # header inner padding
BADGE = 30             # letter-badge square
BADGE_GAP = 15         # gap from badge to the text column
TITLE_FS = 26          # title font size
TITLE_LH = 32          # title line height
TITLE_INK = "#1b2432"  # near-black title (accent is reserved for the badge/rule)
SUB_FS = 19            # subtitle font size
SUB_LH = 26            # subtitle line height
SUB_INK = "#4a5460"    # subtitle grey (a touch darker for readability)
RULE_W = 34            # length of the accent rule under the title
RULE_H = 3             # thickness of the accent rule
BADGE_DROP = 3         # badge top offset so it optically aligns with the title cap
TITLE_GAP = 13         # title block -> accent rule
RULE_GAP = 12          # accent rule -> subtitle
HEADER_GAP = 20        # subtitle -> illustration


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


def _text_x() -> float:
    return PAD + BADGE + BADGE_GAP


def _header_lines(panel: dict, text_w: float):
    title = _wrap(panel["title"], text_w, TITLE_FS, bold=True)
    sub = _wrap(panel["subtitle"], text_w, SUB_FS, bold=False)
    return title, sub


def _title_block(title_lines) -> float:
    return max(BADGE, TITLE_LH * len(title_lines))


def _header_height(title_lines, sub_lines) -> float:
    return (PAD + _title_block(title_lines) + TITLE_GAP + RULE_H + RULE_GAP
            + SUB_LH * len(sub_lines) + HEADER_GAP)


def _draw_header(panel: dict, title_lines, sub_lines, header_h: float, _cx: float) -> str:
    accent = panel["accent"]
    tx = _text_x()
    parts = [
        # accent letter badge, top-left
        f'<rect x="{PAD}" y="{PAD + BADGE_DROP}" width="{BADGE}" height="{BADGE}" '
        f'rx="8" fill="{accent}"/>',
        f'<text x="{PAD + BADGE/2:.1f}" y="{PAD + BADGE_DROP + BADGE/2 + 6:.1f}" '
        f'text-anchor="middle" font-size="18" font-weight="700" fill="#ffffff">'
        f'{panel["id"]}</text>',
    ]
    # title — restrained dark ink, left-aligned column beside the badge
    for i, line in enumerate(title_lines):
        parts.append(
            f'<text x="{tx:.1f}" y="{PAD + TITLE_FS + i*TITLE_LH:.1f}" '
            f'font-size="{TITLE_FS}" font-weight="700" letter-spacing="-0.01em" '
            f'fill="{TITLE_INK}">{escape(line)}</text>')
    # short accent rule — carries the panel colour-coding without a loud title
    ry = PAD + _title_block(title_lines) + TITLE_GAP
    parts.append(
        f'<rect x="{tx:.1f}" y="{ry:.1f}" width="{RULE_W}" height="{RULE_H}" '
        f'rx="{RULE_H/2:.1f}" fill="{accent}"/>')
    # subtitle — muted grey, left-aligned in the same column
    sy = ry + RULE_H + RULE_GAP + SUB_FS
    for i, line in enumerate(sub_lines):
        parts.append(
            f'<text x="{tx:.1f}" y="{sy + i*SUB_LH:.1f}" font-size="{SUB_FS}" '
            f'font-weight="400" fill="{SUB_INK}">{escape(line)}</text>')
    return "".join(parts)


def build_figure1() -> Path:
    # Each illustration renders at IMG_H * its own `hscale`, and its column width
    # follows its aspect ratio. Denser panels (b, c) get a larger hscale so they
    # occupy MORE of the figure — bigger, more legible — while the compact panel a
    # stays smaller. Cards are top-aligned; each card is as tall as its own content.
    prepared = []
    header_h = 0.0
    for p in PANELS:
        png = VIZ / p["loom"]
        if not png.is_file():
            raise SystemExit(f"missing loom panel PNG: {png} — render the panels first")
        img_h = round(IMG_H * p.get("hscale", 1.0))
        img_w = round(img_h * _aspect(png))
        panel_w = img_w + 2 * SIDE
        text_w = panel_w - _text_x() - PAD
        title_lines, sub_lines = _header_lines(p, text_w)
        header_h = max(header_h, _header_height(title_lines, sub_lines))
        prepared.append((p, png, title_lines, sub_lines, img_w, panel_w, img_h))

    # Per-panel card height (ragged bottoms), figure height = the tallest card.
    def _card_h(img_h):
        return round(header_h + img_h + BOTTOM)
    fig_h = round(2 * MARGIN + max(_card_h(ih) for *_, ih in prepared))

    fig_w = round(2 * MARGIN + sum(pw for *_, pw, _ in prepared) + GAP * (len(prepared) - 1))
    root = ET.Element(_q("svg"), {
        "width": str(fig_w), "height": str(fig_h),
        "viewBox": f"0 0 {fig_w} {fig_h}", "font-family": FONT,
    })
    ET.SubElement(root, _q("rect"), {"x": "0", "y": "0", "width": str(fig_w),
                                     "height": str(fig_h), "fill": "#ffffff"})

    x = MARGIN
    for p, png, title_lines, sub_lines, img_w, panel_w, img_h in prepared:
        g = ET.SubElement(root, _q("g"), {"id": f"Panel {p['id']}",
                                          "transform": f"translate({x},{MARGIN})"})
        ET.SubElement(g, _q("rect"), {
            "x": "0", "y": "0", "width": str(panel_w), "height": str(_card_h(img_h)),
            "rx": str(CARD_RX), "fill": p["tint"], "stroke": p["stroke"],
            "stroke-width": "1.4"})
        g.append(ET.fromstring(
            f'<g xmlns="{SVG_NS}">' + _draw_header(p, title_lines, sub_lines, header_h, 0) + "</g>"))
        img = ET.SubElement(g, _q("image"))
        img.set("href", _data_uri(png))
        img.set("x", str(SIDE)); img.set("y", f"{header_h:.1f}")
        img.set("width", str(img_w)); img.set("height", str(img_h))
        img.set("preserveAspectRatio", "xMidYMin meet")
        x += panel_w + GAP

    out = VIZ / "figure_1.svg"
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    print(f"composed {out.relative_to(WS)} — {len(prepared)} panels, {fig_w}x{fig_h}, header {header_h:.0f}px")
    return out


if __name__ == "__main__":
    build_figure1()
