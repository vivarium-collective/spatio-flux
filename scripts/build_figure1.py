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
TOP = 78      # image top (just below the ~74px header block)
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


def build_figure1() -> Path:
    tree = ET.parse(SCAFFOLD)
    root = tree.getroot()
    # Drop panel D (community) — Figure 1 is now just A / B / C.
    for panel in root.findall(_q("g")):
        if panel.get("id", "") == "Panel D - Community":
            root.remove(panel)
    cw = PANEL_W - 2 * SIDE  # image width inside a card
    x = MARGIN
    max_h = 0
    swapped = 0
    for panel in root.findall(_q("g")):
        loom = PANEL_LOOM.get(panel.get("id", ""))
        if not loom:
            continue  # defs / other groups: untouched
        png = VIZ / loom
        if not png.is_file():
            raise SystemExit(f"missing loom panel PNG: {png} — render the panels first")
        # Size the image to fill the card width at its native aspect (no letterbox);
        # the card grows/shrinks to fit → the image fills it with only SIDE/TOP/BOTTOM
        # margins. Cards are top-aligned (ragged bottoms) so none carries dead space.
        ch = round(cw / _aspect(png))
        panel_h = TOP + ch + BOTTOM
        panel.find(_q("rect")).set("height", str(panel_h))  # background rect = first <rect>
        for extra in list(panel)[HEADER_KEEP:]:
            panel.remove(extra)  # drop the scaffold's placeholder illustration
        img = ET.SubElement(panel, _q("image"))
        img.set("href", _data_uri(png))
        img.set("x", str(SIDE)); img.set("y", str(TOP))
        img.set("width", str(cw)); img.set("height", str(ch))
        img.set("preserveAspectRatio", "none")  # box == image aspect → fills, no distortion
        panel.set("transform", f"translate({x},{MARGIN})")  # pull the cards together
        x += PANEL_W + GAP
        max_h = max(max_h, panel_h)
        swapped += 1
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
