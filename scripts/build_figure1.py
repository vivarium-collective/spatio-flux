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


# ── Share & Reuse section (appended below panel C) ───────────────────────────
# A small loom-styled block: three rounded green "store" cards — a Process
# Registry, Schema Registry, and Model Repository — under a "Share & Reuse"
# heading, with publish / discover arrows. Drawn as an SVG <g> so it composites
# straight into the scaffold's panel-C group.
SHARE_PANEL = "Panel C - Process Bigraph"
SHARE_H = 158  # height of the appended section


def _sr_icon_process(cx: float) -> str:  # a small process/component network
    return (f'<g stroke="#16a34a" stroke-width="1.5">'
            f'<line x1="{cx}" y1="60" x2="{cx-9}" y2="71"/><line x1="{cx}" y1="60" x2="{cx+9}" y2="71"/>'
            f'<line x1="{cx-9}" y1="71" x2="{cx+9}" y2="71"/><line x1="{cx}" y1="60" x2="{cx}" y2="76"/></g>'
            f'<g fill="#16a34a"><circle cx="{cx}" cy="60" r="2.8"/><circle cx="{cx-9}" cy="71" r="2.8"/>'
            f'<circle cx="{cx+9}" cy="71" r="2.8"/><circle cx="{cx}" cy="76" r="2.8"/></g>')


def _sr_icon_schema(cx: float) -> str:  # a 2×2 grid of interface/type tiles
    return (f'<g fill="none" stroke="#3b82f6" stroke-width="1.5">'
            f'<rect x="{cx-9}" y="59" width="8" height="8" rx="1.5"/><rect x="{cx+1}" y="59" width="8" height="8" rx="1.5"/>'
            f'<rect x="{cx-9}" y="69" width="8" height="8" rx="1.5"/><rect x="{cx+1}" y="69" width="8" height="8" rx="1.5"/></g>')


def _sr_icon_repo(cx: float) -> str:  # a database cylinder
    return (f'<g fill="#dcfce7" stroke="#16a34a" stroke-width="1.6">'
            f'<path d="M{cx-9} 61 v11 a9 3.4 0 0 0 18 0 v-11"/><ellipse cx="{cx}" cy="61" rx="9" ry="3.4"/></g>'
            f'<path d="M{cx-9} 67.5 a9 3.4 0 0 0 18 0" fill="none" stroke="#16a34a" stroke-width="1.2" opacity="0.6"/>')


def _share_reuse_group(w: int):
    bw, y = 94, 47
    boxes = [(4, _sr_icon_process, "Process Registry", "Components"),
             (107, _sr_icon_schema, "Schema Registry", "Interfaces &amp; types"),
             (210, _sr_icon_repo, "Model Repository", "Composites")]
    parts = [
        f'<defs><marker id="sr_ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker></defs>',
        f'<rect x="1" y="2" width="{w - 2}" height="152" rx="12" fill="#f6fdfa" stroke="#bbf7d0" '
        f'stroke-width="1.5" stroke-dasharray="5 4"/>',
        f'<text x="{w / 2:.0f}" y="24" text-anchor="middle" font-family="Georgia, \'Times New Roman\', serif" '
        f'font-size="15" font-weight="700" fill="#16a34a">Share &amp; Reuse</text>',
        f'<text x="{w / 2:.0f}" y="39" text-anchor="middle" font-size="8.5" fill="#64748b">'
        f'Publish reusable components and discover those from others.</text>',
    ]
    for x, icon, name, sub in boxes:
        cx = x + bw / 2
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="66" rx="9" fill="#ffffff" stroke="#22c55e" stroke-width="1.8"/>')
        parts.append(icon(cx))
        parts.append(f'<text x="{cx:.0f}" y="97" text-anchor="middle" font-size="8.5" font-weight="700" fill="#1e293b">{name}</text>')
        parts.append(f'<text x="{cx:.0f}" y="108" text-anchor="middle" font-size="7" fill="#64748b">{sub}</text>')
    for lx, lbl in ((78, "publish"), (228, "discover &amp; reuse")):
        parts.append(f'<path d="M{lx} 141 V127" fill="none" stroke="#94a3b8" stroke-width="1.4" '
                     f'stroke-dasharray="4 3" marker-end="url(#sr_ar)"/>')
        parts.append(f'<text x="{lx}" y="150" text-anchor="middle" font-size="8" font-style="italic" fill="#64748b">{lbl}</text>')
    return ET.fromstring(f'<g xmlns="{SVG_NS}">' + "".join(parts) + "</g>")


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
        is_share = panel.get("id", "") == SHARE_PANEL
        natural_h = (TOP + ch + 14 + SHARE_H + BOTTOM) if is_share else (TOP + ch + BOTTOM)
        info.append((panel, png, ch, is_share, natural_h))
    panel_h = max(n for *_, n in info)  # common height for all three cards
    x = MARGIN
    for panel, png, ch, is_share, _ in info:
        panel.find(_q("rect")).set("height", str(panel_h))  # background rect = first <rect>
        for extra in list(panel)[HEADER_KEEP:]:
            panel.remove(extra)  # drop the scaffold's placeholder illustration
        img = ET.SubElement(panel, _q("image"))
        img.set("href", _data_uri(png))
        img.set("x", str(SIDE)); img.set("y", str(TOP))
        img.set("width", str(cw)); img.set("height", str(ch))
        img.set("preserveAspectRatio", "none")  # box == image aspect → fills, no distortion
        if is_share:  # append the loom-styled Share & Reuse block under the image
            grp = _share_reuse_group(cw)
            grp.set("transform", f"translate({SIDE},{TOP + ch + 14})")
            panel.append(grp)
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
