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
# Content region inside a 320×740 panel, below the ~88px header (matches where
# the scaffold's own illustration sat: translate(14,96), 292 wide).
CX, CY, CW, CH = 14, 90, 292, 636


def _q(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def _data_uri(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii")


def build_figure1() -> Path:
    tree = ET.parse(SCAFFOLD)
    root = tree.getroot()
    swapped = 0
    for panel in root.findall(_q("g")):
        loom = PANEL_LOOM.get(panel.get("id", ""))
        if not loom:
            continue  # panel d + defs/other groups: untouched
        png = VIZ / loom
        if not png.is_file():
            raise SystemExit(f"missing loom panel PNG: {png} — render the panels first")
        for extra in list(panel)[HEADER_KEEP:]:
            panel.remove(extra)  # drop the scaffold's placeholder illustration
        img = ET.SubElement(panel, _q("image"))
        img.set("href", _data_uri(png))
        img.set("x", str(CX)); img.set("y", str(CY))
        img.set("width", str(CW)); img.set("height", str(CH))
        img.set("preserveAspectRatio", "xMidYMid meet")
        swapped += 1
    out = VIZ / "figure_1.svg"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"composed {out.relative_to(WS)} — swapped {swapped}/3 loom panels, kept panel d")
    return out


if __name__ == "__main__":
    build_figure1()
