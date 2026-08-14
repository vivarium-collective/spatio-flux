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


def _q(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def _data_uri(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii")


def _aspect(png: Path) -> float:
    with Image.open(png) as im:
        w, h = im.size
    return w / h


# ── Unified bottom callout for a / b / c — icon-top columns, one keyword each ─
# A three-line abstract of the paper (per-column UPPERCASE keyword + support):
#   a  STATE / DYNAMICS / SCALE                       (why composition is hard)
#   b  INTERFACES / STORES / WIRING / ORCHESTRATION   (what's made explicit)
#   c  SUBSTITUTE / RECOMBINE / REPRODUCE / SHARE     (what becomes possible)
# All three share the SAME height + grammar; no box header, no tagline.
SHARE_PANEL = "Panel C - Process Bigraph"
BOTTOM_H = 44  # uniform height of every a/b/c bottom callout (icon + keyword only)


def _db(x, y, c):  # a database cylinder (STATE + SHARED STORES)
    return (f'<g fill="{c}" fill-opacity="0.12" stroke="{c}" stroke-width="1.7" stroke-linejoin="round">'
            f'<ellipse cx="{x}" cy="{y-6}" rx="8" ry="3"/><path d="M{x-8} {y-6} v12 a8 3 0 0 0 16 0 v-12"/></g>'
            f'<g fill="none" stroke="{c}" stroke-width="1.3" opacity="0.6">'
            f'<path d="M{x-8} {y-1} a8 3 0 0 0 16 0"/><path d="M{x-8} {y+3.5} a8 3 0 0 0 16 0"/></g>')


# ---- Panel A icons — State / Dynamics / Scale -------------------------------
def _ic_dynamics(x, y, c):  # a small molecular / process network
    return (f'<g stroke="{c}" stroke-width="1.5"><line x1="{x}" y1="{y-6}" x2="{x-7}" y2="{y+3}"/>'
            f'<line x1="{x}" y1="{y-6}" x2="{x+7}" y2="{y+3}"/><line x1="{x-7}" y1="{y+3}" x2="{x+7}" y2="{y+3}"/>'
            f'<line x1="{x}" y1="{y-6}" x2="{x}" y2="{y+7}"/></g>'
            f'<g fill="#ffffff" stroke="{c}" stroke-width="1.6"><circle cx="{x}" cy="{y-6}" r="2.6"/>'
            f'<circle cx="{x-7}" cy="{y+3}" r="2.6"/><circle cx="{x+7}" cy="{y+3}" r="2.6"/><circle cx="{x}" cy="{y+7}" r="2.3"/></g>')


def _ic_scale(x, y, c):     # stacked layers (molecules → cells → tissues)
    return (f'<g fill="{c}" fill-opacity="0.14" stroke="{c}" stroke-width="1.6" stroke-linejoin="round">'
            f'<path d="M{x} {y-8} l9 4.5 l-9 4.5 l-9 -4.5 z"/><path d="M{x} {y-1} l9 4.5 l-9 4.5 l-9 -4.5 z"/></g>')


# ---- Panel B icons — Interfaces / Stores / Wiring / Orchestration -----------
def _ic_interfaces(x, y, c):  # a typed-interface bar with ports
    return (f'<line x1="{x}" y1="{y-8}" x2="{x}" y2="{y+8}" stroke="{c}" stroke-width="1.8"/>'
            f'<g stroke="{c}" stroke-width="1.6" stroke-linecap="round"><line x1="{x}" y1="{y-4}" x2="{x-7}" y2="{y-4}"/>'
            f'<line x1="{x}" y1="{y+4}" x2="{x-7}" y2="{y+4}"/><line x1="{x}" y1="{y}" x2="{x+7}" y2="{y}"/></g>'
            f'<g fill="{c}"><circle cx="{x-7}" cy="{y-4}" r="1.9"/><circle cx="{x-7}" cy="{y+4}" r="1.9"/>'
            f'<circle cx="{x+7}" cy="{y}" r="1.9"/></g>')


def _ic_wiring(x, y, c):    # two interlocking chain links
    return (f'<g fill="none" stroke="{c}" stroke-width="1.9" stroke-linecap="round">'
            f'<rect x="{x-9}" y="{y-3.6}" width="10" height="7.2" rx="3.6"/>'
            f'<rect x="{x-1}" y="{y-3.6}" width="10" height="7.2" rx="3.6"/></g>')


def _ic_orchestration(x, y, c):  # orchestration — timing / a clock
    return (f'<circle cx="{x}" cy="{y}" r="8.5" fill="none" stroke="{c}" stroke-width="1.7"/>'
            f'<path d="M{x} {y} V{y-5} M{x} {y} L{x+4} {y+2}" fill="none" stroke="{c}" stroke-width="1.7" stroke-linecap="round"/>'
            f'<g fill="{c}"><circle cx="{x}" cy="{y-8.5}" r="1.1"/><circle cx="{x+8.5}" cy="{y}" r="1.1"/>'
            f'<circle cx="{x}" cy="{y+8.5}" r="1.1"/><circle cx="{x-8.5}" cy="{y}" r="1.1"/></g>')


# ---- Panel C icons — Substitute / Recombine / Reproduce / Share -------------
def _ic_substitute(x, y, c):  # two opposed swap arrows
    return (f'<g fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="M{x-8} {y-3} h13 m-3.5 -3.5 l3.5 3.5 l-3.5 3.5"/>'
            f'<path d="M{x+8} {y+3.5} h-13 m3.5 -3.5 l-3.5 3.5 l3.5 3.5"/></g>')


def _ic_recombine(x, y, c):  # a puzzle piece
    return (f'<path d="M{x-7.5} {y-7.5} h4.5 a2.2 2.2 0 0 1 4.6 0 h4.4 v4.4 a2.2 2.2 0 0 0 0 4.6 v4.5 '
            f'h-4.4 a2.2 2.2 0 0 1 -4.6 0 h-4.5 v-4.5 a2.2 2.2 0 0 0 0 -4.6 z" '
            f'fill="{c}" fill-opacity="0.14" stroke="{c}" stroke-width="1.6" stroke-linejoin="round"/>')


def _ic_reproduce(x, y, c):  # a play button on a card (run anywhere)
    return (f'<rect x="{x-9}" y="{y-7}" width="18" height="14" rx="2.6" fill="{c}" fill-opacity="0.1" '
            f'stroke="{c}" stroke-width="1.6"/>'
            f'<path d="M{x-2.6} {y-3.6} l6 3.6 l-6 3.6 z" fill="{c}"/>')


def _ic_share(x, y, c):     # three community members
    def person(px, py):
        return (f'<circle cx="{px}" cy="{py}" r="2.3" fill="{c}"/>'
                f'<path d="M{px-3.4} {py+6.2} a3.4 3.4 0 0 1 6.8 0 z" fill="{c}"/>')
    return person(x, y - 3) + person(x - 7.5, y + 1.5) + person(x + 7.5, y + 1.5)


# Content per box: (icon, KEYWORD). Just the icon + keyword — no support text.
BOTTOM_A = [(_db, "STATE"), (_ic_dynamics, "DYNAMICS"), (_ic_scale, "SCALE")]
BOTTOM_B = [(_ic_interfaces, "TYPED INTERFACES"), (_db, "SHARED STORES"),
            (_ic_wiring, "EXPLICIT WIRING"), (_ic_orchestration, "ORCHESTRATION")]
BOTTOM_C = [(_ic_substitute, "SUBSTITUTE"), (_ic_recombine, "RECOMBINE"),
            (_ic_reproduce, "REPRODUCE"), (_ic_share, "SHARE")]


def _bottom_callout(w, items, fill, border, accent):
    n = len(items)
    col = w / n
    # keyword font shrinks a touch when the longest keyword is wide (4-col boxes).
    kw_fs = 7.1 if max(len(k) for _, k in items) > 8 else 8.4
    parts = [f'<rect x="1" y="2" width="{w - 2}" height="{BOTTOM_H - 4}" rx="10" '
             f'fill="{fill}" stroke="{border}" stroke-width="1.3"/>']
    icon_cy = 17
    for i, (icon, keyword) in enumerate(items):
        cx = col * (i + 0.5)
        parts.append(icon(cx, icon_cy, accent))
        parts.append(f'<text x="{cx:.0f}" y="{icon_cy + 18:.1f}" text-anchor="middle" font-size="{kw_fs}" '
                     f'font-weight="700" letter-spacing="0.03em" fill="{accent}">{keyword}</text>')
    return ET.fromstring(f'<g xmlns="{SVG_NS}">' + "".join(parts) + "</g>")


# Per-panel bottom block. Disabled: the panel headers + graphs already carry the
# heterogeneity / composition / reuse story, so no bottom callout is drawn.
# (The _bottom_callout + BOTTOM_A/B/C content above are kept for easy re-enable.)
def _bottom_block(panel_id: str, w: int):
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
        natural_h = TOP + ch + (8 + block_h if block is not None else 0) + BOTTOM
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
