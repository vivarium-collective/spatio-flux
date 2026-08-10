#!/usr/bin/env python
"""Stitch the paper-figures investigation's per-study visualization panels into
publication-ready, subpanel-labeled figures.

For each member study of the ``paper-figures`` investigation, grabs its declared
``visualizations`` (the panel images under ``studies/<slug>/visualizations/``)
and composites them into one figure — a shelf/masonry layout with bold "a.",
"b.", … subpanel labels — written to
``investigations/paper-figures/figures/figure_<N>.svg`` (+ a rasterized .png).

Panels are embedded as high-res PNGs (reliable in any SVG rasterizer; loom's
foreignObject SVGs don't render when referenced via <image>), while the layout
and labels stay vector.

Run:  python scripts/build_paper_figures.py
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import yaml
from PIL import Image

WS = Path(__file__).resolve().parents[1]
INV = WS / "investigations" / "paper-figures"
OUT = INV / "figures"

# study slug -> figure number (fig-01 -> 1, fig-07 -> 7, …)
def _fig_num(slug: str) -> str:
    m = re.match(r"fig-0*(\d+)", slug)
    return m.group(1) if m else slug

ROW_H = 460       # every panel scaled to this height
GAP = 34          # gap between panels / rows
MAX_W = 2000      # wrap to a new row past this width
LABEL_H = 40      # space above each panel for its "a." label
PAD = 26


def _panel_png(study_dir: Path, viz: dict) -> Path | None:
    """Resolve a visualization entry to its PNG panel (prefer a .png sibling of
    the declared image, since loom SVGs don't embed reliably)."""
    addr = str(viz.get("address", ""))
    if not addr.startswith("image:"):
        return None
    rel = addr[len("image:"):]
    p = study_dir / rel
    png = p.with_suffix(".png")
    if png.exists():
        return png
    return p if p.exists() and p.suffix == ".png" else None


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_figure(study: str) -> Path | None:
    sdir = WS / "studies" / study
    spec_f = sdir / "study.yaml"
    if not spec_f.is_file():
        return None
    spec = yaml.safe_load(spec_f.read_text()) or {}
    panels: list[Path] = []
    for v in (spec.get("visualizations") or []):
        if isinstance(v, dict):
            p = _panel_png(sdir, v)
            if p:
                panels.append(p)
    if not panels:
        return None

    # Shelf layout: scale each panel to ROW_H, pack left→right, wrap past MAX_W.
    scaled = []  # (path, w, h)
    for p in panels:
        with Image.open(p) as im:
            w, h = im.size
        scaled.append((p, w * ROW_H / h, float(ROW_H)))
    rows: list[list] = []
    cur: list = []
    cur_w = 0.0
    for item in scaled:
        sw = item[1]
        if cur and cur_w + sw + GAP > MAX_W:
            rows.append(cur)
            cur, cur_w = [], 0.0
        cur.append(item)
        cur_w += sw + GAP
    if cur:
        rows.append(cur)

    fig_w = max(sum(sw + GAP for _, sw, _ in row) - GAP for row in rows) + 2 * PAD
    fig_h = len(rows) * (ROW_H + LABEL_H + GAP) - GAP + 2 * PAD

    title = f"Figure {_fig_num(study)}"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_w:.0f}" height="{fig_h:.0f}" '
        f'viewBox="0 0 {fig_w:.0f} {fig_h:.0f}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    li = 0
    y = float(PAD)
    for row in rows:
        x = float(PAD)
        for p, sw, sh in row:
            label = chr(ord("a") + li)
            li += 1
            parts.append(
                f'<text x="{x:.0f}" y="{y + 26:.0f}" font-family="Georgia, \'Times New Roman\', serif" '
                f'font-size="30" font-weight="bold" fill="#111827">{label}.</text>'
            )
            parts.append(
                f'<image href="{_data_uri(p)}" x="{x:.0f}" y="{y + LABEL_H:.0f}" '
                f'width="{sw:.0f}" height="{sh:.0f}" preserveAspectRatio="xMidYMid meet"/>'
            )
            x += sw + GAP
        y += ROW_H + LABEL_H + GAP
    parts.append(f'<!-- {title}: {li} panels from studies/{study} -->')
    parts.append("</svg>")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"figure_{_fig_num(study)}.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def _write_gallery(built: list[tuple[str, Path]]) -> Path:
    """Emit a self-contained gallery page linking every stitched figure — the
    investigation's 'Figures' view."""
    cards = []
    for study, out in built:
        n = _fig_num(study)
        cards.append(
            f'<section style="margin:0 0 48px"><h2 style="font:600 20px system-ui;color:#111827;margin:0 0 10px">'
            f'Figure {n}</h2>'
            f'<img src="{out.name}" alt="Figure {n}" '
            f'style="max-width:100%;border:1px solid #e5e7eb;border-radius:8px"/></section>'
        )
    html = (
        '<!doctype html><meta charset="utf-8"><title>Paper figures</title>'
        '<body style="max-width:1200px;margin:32px auto;padding:0 20px;background:#fff">'
        '<h1 style="font:700 28px Georgia,serif;color:#111827">Process Bigraph paper — figures</h1>'
        '<p style="color:#6b7280;font:14px system-ui">Publication-ready figures, each stitched from its '
        'study\'s subpanels (see <code>scripts/build_paper_figures.py</code>).</p>'
        + "".join(cards) + "</body>"
    )
    page = OUT / "index.html"
    page.write_text(html, encoding="utf-8")
    return page


def main() -> None:
    inv = yaml.safe_load((INV / "investigation.yaml").read_text()) or {}
    studies = inv.get("studies") or []
    if not studies:
        print("no member studies in paper-figures investigation")
        sys.exit(1)
    built = []
    for s in studies:
        out = build_figure(s)
        if out:
            print(f"  OK  Figure {_fig_num(s)} -> {out.relative_to(WS)}")
            built.append((s, out))
        else:
            print(f"  skip {s}: no panels")
    gallery = _write_gallery(built)
    print(f"built {len(built)}/{len(studies)} figures + gallery {gallery.relative_to(WS)}")


if __name__ == "__main__":
    main()
