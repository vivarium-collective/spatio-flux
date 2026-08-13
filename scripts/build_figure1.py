#!/usr/bin/env python
"""Compose Figure 1 from a hand-designed SVG scaffold, swapping the loom panels
into a / b / c and keeping panel d (the community drawing) + every panel's
header text.

Reproducible remake: reads the committed scaffold
``investigations/paper-figures/inputs/figure1-scaffold.svg``, and for panels
A / B / C keeps the 5 header elements (background, letter badge, title, caption)
while REPLACING the illustration with the study's loom panel PNG. Panel D
(community) is left untouched. Writes
``studies/fig-01/visualizations/figure_1.svg``.

The loom panel PNGs must already be rendered (scripts/render_loom_svgs.mjs or the
per-panel one-offs) — this only stitches. Run:

    python scripts/build_figure1.py
"""
from __future__ import annotations

import base64
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

WS = Path(__file__).resolve().parents[1]
SCAFFOLD = WS / "investigations" / "paper-figures" / "inputs" / "figure1-scaffold.svg"
VIZ = WS / "studies" / "fig-01" / "visualizations"

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

# Panel group id → the loom PNG panel to swap in. a←1a, b←1b, c←1c; panel d
# (community) is deliberately absent → kept verbatim from the scaffold.
PANEL_LOOM = {
    "Panel A - Different formalisms":   "fig01a-draft-processes.png",
    "Panel B - Composition interfaces": "fig01b-multiscale-composite.png",
    "Panel C - Process Bigraph":        "fig01c-study-workflow.png",
}
# Every panel opens with the SAME 5 header elements (verified across A–D):
#   [0] background rect (320×740)
#   [1] letter-badge rect   [2] letter-badge text
#   [3] title <g> (2 lines) [4] caption <g> (2 lines)
# …then the illustration. Keep the header, drop the rest.
HEADER_KEEP = 5
# Tight per-panel layout (removes white space): each card is sized to its loom
# image so the image FILLS it — no letterbox — with small margins, and the cards
# sit close together. PANEL_W stays 320 (the header text needs it); SIDE/TOP are
# the small inner margins, GAP the space between cards, MARGIN the outer border.
PANEL_W = 320
SIDE = 6      # inner left/right margin around the image
TOP = 102     # image top (below the header: 2-line title + 3-line caption, enlarged)
BOTTOM = 8    # inner margin below the image
GAP = 16      # between A / B / C  (was 40)
MARGIN = 20   # outer figure margin
CALLOUT_H = 74  # height of the A / B bottom callout strip


def _q(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def _data_uri(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii")


def _aspect(png: Path) -> float:
    with Image.open(png) as im:
        w, h = im.size
    return w / h


# ── Share & Reuse section (appended below panel C) ───────────────────────────
# A small loom-styled block: three rounded green "store" cards — a Process
# Registry, Schema Registry, and Model Repository — under a "Share & Reuse"
# heading, with publish / discover arrows. Drawn as an SVG <g> so it composites
# straight into the scaffold's panel-C group.
SHARE_PANEL = "Panel C - Process Bigraph"
SHARE_H = 126  # height of the appended section


def _sr_icon_process(cx: float) -> str:  # a small process/component network
    return (f'<g stroke="#2f6b3f" stroke-width="1.5">'
            f'<line x1="{cx}" y1="60" x2="{cx-9}" y2="71"/><line x1="{cx}" y1="60" x2="{cx+9}" y2="71"/>'
            f'<line x1="{cx-9}" y1="71" x2="{cx+9}" y2="71"/><line x1="{cx}" y1="60" x2="{cx}" y2="76"/></g>'
            f'<g fill="#2f6b3f"><circle cx="{cx}" cy="60" r="2.8"/><circle cx="{cx-9}" cy="71" r="2.8"/>'
            f'<circle cx="{cx+9}" cy="71" r="2.8"/><circle cx="{cx}" cy="76" r="2.8"/></g>')


def _sr_icon_schema(cx: float) -> str:  # a 2×2 grid of interface/type tiles
    return (f'<g fill="none" stroke="#2f6b3f" stroke-width="1.5">'
            f'<rect x="{cx-9}" y="59" width="8" height="8" rx="1.5"/><rect x="{cx+1}" y="59" width="8" height="8" rx="1.5"/>'
            f'<rect x="{cx-9}" y="69" width="8" height="8" rx="1.5"/><rect x="{cx+1}" y="69" width="8" height="8" rx="1.5"/></g>')


def _sr_icon_repo(cx: float) -> str:  # a database cylinder
    return (f'<g fill="#e7f1ea" stroke="#2f6b3f" stroke-width="1.6">'
            f'<path d="M{cx-9} 61 v11 a9 3.4 0 0 0 18 0 v-11"/><ellipse cx="{cx}" cy="61" rx="9" ry="3.4"/></g>'
            f'<path d="M{cx-9} 67.5 a9 3.4 0 0 0 18 0" fill="none" stroke="#2f6b3f" stroke-width="1.2" opacity="0.6"/>')


def _share_reuse_group(w: int):
    bw, y = 94, 47
    boxes = [(4, _sr_icon_process, "Process Registry", "Components"),
             (107, _sr_icon_schema, "Schema Registry", "Interfaces &amp; types"),
             (210, _sr_icon_repo, "Model Repository", "Composites")]
    parts = [
        f'<rect x="1" y="2" width="{w - 2}" height="120" rx="12" fill="#fdf8ea" stroke="#e6d9a8" stroke-width="1.3"/>',
        f'<text x="{w / 2:.0f}" y="24" text-anchor="middle" font-family="Georgia, \'Times New Roman\', serif" '
        f'font-size="16.5" font-weight="700" fill="#2f6b3f">Share &amp; Reuse</text>',
        f'<text x="{w / 2:.0f}" y="39" text-anchor="middle" font-size="9.5" fill="#64748b">'
        f'Publish reusable components and discover those from others.</text>',
    ]
    for x, icon, name, sub in boxes:
        cx = x + bw / 2
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="66" rx="9" fill="#ffffff" stroke="#2f6b3f" stroke-width="1.8"/>')
        parts.append(icon(cx))
        parts.append(f'<text x="{cx:.0f}" y="97" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1e293b">{name}</text>')
        parts.append(f'<text x="{cx:.0f}" y="109" text-anchor="middle" font-size="8" fill="#64748b">{sub}</text>')
    return ET.fromstring(f'<g xmlns="{SVG_NS}">' + "".join(parts) + "</g>")


# ── A / B bottom callout strips (icon-left, three items) ─────────────────────
def _ic_cluster(x, y, c):   # multi-scale — a cluster of cells
    return (f'<g fill="none" stroke="{c}" stroke-width="1.3">'
            f'<circle cx="{x}" cy="{y-4}" r="3.4"/><circle cx="{x-5}" cy="{y+2}" r="3.4"/>'
            f'<circle cx="{x+5}" cy="{y+2}" r="3.4"/><circle cx="{x}" cy="{y+6}" r="2.8"/></g>'
            f'<g fill="{c}"><circle cx="{x}" cy="{y-4}" r="1"/><circle cx="{x-5}" cy="{y+2}" r="1"/><circle cx="{x+5}" cy="{y+2}" r="1"/></g>')


def _ic_paradigm(x, y, c):  # multi-paradigm — a cell with mixed inner marks
    return (f'<circle cx="{x}" cy="{y}" r="8" fill="none" stroke="{c}" stroke-width="1.3" stroke-dasharray="2 1.6"/>'
            f'<g fill="{c}"><circle cx="{x-3}" cy="{y-2}" r="1.4"/><circle cx="{x+3}" cy="{y-1}" r="1.4"/>'
            f'<circle cx="{x}" cy="{y+3}" r="1.4"/><circle cx="{x-2}" cy="{y+2}" r="1"/></g>')


def _ic_gear(x, y, c):      # data-informed — a gear
    teeth = "".join(
        f'<line x1="{x + 5 * math.cos(math.radians(a)):.1f}" y1="{y + 5 * math.sin(math.radians(a)):.1f}" '
        f'x2="{x + 8 * math.cos(math.radians(a)):.1f}" y2="{y + 8 * math.sin(math.radians(a)):.1f}"/>'
        for a in range(0, 360, 45))
    return (f'<g stroke="{c}" stroke-width="1.3">{teeth}</g>'
            f'<circle cx="{x}" cy="{y}" r="4.5" fill="none" stroke="{c}" stroke-width="1.3"/>'
            f'<circle cx="{x}" cy="{y}" r="1.6" fill="{c}"/>')


def _ic_chip(x, y, c):      # typed interfaces — a node with typed ports
    return (f'<circle cx="{x}" cy="{y}" r="6" fill="none" stroke="{c}" stroke-width="1.3"/>'
            f'<g stroke="{c}" stroke-width="1.3"><line x1="{x-6}" y1="{y}" x2="{x-10}" y2="{y}"/>'
            f'<line x1="{x+6}" y1="{y}" x2="{x+10}" y2="{y}"/><line x1="{x}" y1="{y-6}" x2="{x}" y2="{y-10}"/></g>'
            f'<g fill="{c}"><circle cx="{x-10}" cy="{y}" r="1.6"/><circle cx="{x+10}" cy="{y}" r="1.6"/>'
            f'<circle cx="{x}" cy="{y-10}" r="1.6"/></g>')


def _ic_link(x, y, c):      # explicit coupling — two ports linked
    return (f'<line x1="{x-6}" y1="{y}" x2="{x+6}" y2="{y}" stroke="{c}" stroke-width="1.6"/>'
            f'<g fill="none" stroke="{c}" stroke-width="1.3"><circle cx="{x-8}" cy="{y}" r="2.6"/><circle cx="{x+8}" cy="{y}" r="2.6"/></g>')


def _ic_tree(x, y, c):      # hierarchical composition — parts → whole
    return (f'<g stroke="{c}" stroke-width="1.3"><line x1="{x}" y1="{y-7}" x2="{x-6}" y2="{y+5}"/>'
            f'<line x1="{x}" y1="{y-7}" x2="{x+6}" y2="{y+5}"/></g>'
            f'<g fill="{c}"><circle cx="{x}" cy="{y-7}" r="2.4"/><circle cx="{x-6}" cy="{y+5}" r="2.4"/>'
            f'<circle cx="{x+6}" cy="{y+5}" r="2.4"/></g>')


CALLOUT_A = [(_ic_cluster, ["Multi-scale"], ["Molecules →", "Cells → Organs"]),
             (_ic_paradigm, ["Multi-paradigm"], ["ODE, FBA, PDE,", "ABM, ML…"]),
             (_ic_gear, ["Data-informed"], ["Parameters,", "structure, …"])]
CALLOUT_B = [(_ic_chip, ["Typed interfaces"], ["Explicit ports", "&amp; data types"]),
             (_ic_link, ["Explicit coupling"], ["Connect variables,", "not code"]),
             (_ic_tree, ["Hierarchical", "composition"], ["parts → whole"])]


def _callout_group(w, items, fill, border, accent):
    n = len(items)
    col = w / n
    parts = [f'<rect x="1" y="2" width="{w - 2}" height="{CALLOUT_H - 4}" rx="10" '
             f'fill="{fill}" stroke="{border}" stroke-width="1.3"/>']
    for i, (icon, names, subs) in enumerate(items):
        cx0 = col * i
        parts.append(icon(cx0 + 15, CALLOUT_H / 2, accent))
        tx, y = cx0 + 28, 21
        for nm in names:
            parts.append(f'<text x="{tx:.0f}" y="{y:.0f}" font-size="9" font-weight="700" fill="{accent}">{nm}</text>')
            y += 11
        for s in subs:
            parts.append(f'<text x="{tx:.0f}" y="{y:.0f}" font-size="7.6" fill="#64748b">{s}</text>')
            y += 9.4
    return ET.fromstring(f'<g xmlns="{SVG_NS}">' + "".join(parts) + "</g>")


# Per-panel bottom block: (SVG <g>, height). A/B get a callout strip; C the
# Share & Reuse block; anything else none.
def _bottom_block(panel_id: str, w: int):
    if panel_id == "Panel A - Different formalisms":
        return _callout_group(w, CALLOUT_A, "#fdecea", "#e6a8a0", "#c0392b"), CALLOUT_H
    if panel_id == "Panel B - Composition interfaces":
        return _callout_group(w, CALLOUT_B, "#e8f0fe", "#a9c4f5", "#2b5bd0"), CALLOUT_H
    if panel_id == SHARE_PANEL:
        return _share_reuse_group(w), SHARE_H
    return None, 0


def build_figure1() -> Path:
    tree = ET.parse(SCAFFOLD)
    root = tree.getroot()
    # Drop panel D (community) — Figure 1 is now just A / B / C.
    for panel in root.findall(_q("g")):
        if panel.get("id", "") == "Panel D - Community":
            root.remove(panel)
    cw = PANEL_W - 2 * SIDE  # image width inside a card
    # Each loom image fills the card WIDTH at its native aspect (no letterbox).
    # Pre-pass: measure every panel's natural content height, then give ALL cards
    # the SAME height (the tallest) — the images sit top-aligned and any leftover
    # space is the card's own tint, not white, so the three panels line up.
    panels = [p for p in root.findall(_q("g")) if PANEL_LOOM.get(p.get("id", ""))]
    info = []
    for panel in panels:
        png = VIZ / PANEL_LOOM[panel.get("id", "")]
        if not png.is_file():
            raise SystemExit(f"missing loom panel PNG: {png} — render the panels first")
        ch = round(cw / _aspect(png))
        block, block_h = _bottom_block(panel.get("id", ""), cw)  # A/B callout, C Share & Reuse
        natural_h = TOP + ch + (14 + block_h if block is not None else 0) + BOTTOM
        info.append((panel, png, ch, block, block_h, natural_h))
    panel_h = max(n for *_, n in info)  # common height for all three cards
    x = MARGIN
    for panel, png, ch, block, block_h, _ in info:
        panel.find(_q("rect")).set("height", str(panel_h))  # background rect = first <rect>
        for extra in list(panel)[HEADER_KEEP:]:
            panel.remove(extra)  # drop the scaffold's placeholder illustration
        img = ET.SubElement(panel, _q("image"))
        img.set("href", _data_uri(png))
        img.set("x", str(SIDE)); img.set("y", str(TOP))
        img.set("width", str(cw)); img.set("height", str(ch))
        img.set("preserveAspectRatio", "none")  # box == image aspect → fills, no distortion
        if block is not None:  # bottom-align the block so all three line up at the base
            block.set("transform", f"translate({SIDE},{panel_h - BOTTOM - block_h})")
            panel.append(block)
        panel.set("transform", f"translate({x},{MARGIN})")  # pull the cards together
        x += PANEL_W + GAP
    max_h = panel_h
    swapped = len(info)
    fig_w = MARGIN + swapped * PANEL_W + (swapped - 1) * GAP + MARGIN
    fig_h = MARGIN + max_h + MARGIN
    root.set("width", str(fig_w)); root.set("height", str(fig_h))
    root.set("viewBox", f"0 0 {fig_w} {fig_h}")
    out = VIZ / "figure_1.svg"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"composed {out.relative_to(WS)} — {swapped} tight panels (A/B/C), dropped panel D")
    return out


if __name__ == "__main__":
    build_figure1()
