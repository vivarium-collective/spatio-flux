#!/usr/bin/env python
"""Stitch each paper-figure study's panels into one publication-ready,
subpanel-labeled figure — added back as a visualization ON that study.

This is the paper-figures investigation's stitch-together step: it runs AFTER
the figure studies and turns each study's per-panel visualizations (the images
under ``studies/<slug>/visualizations/``) into one composited figure — a
shelf/masonry layout with bold "a.", "b.", … subpanel labels. For each figure
study it emits two files into that study's own ``visualizations/`` dir:

  - ``figure_<N>.svg`` — full-vector (panels embedded as high-res PNGs; layout +
    labels vector), and
  - ``figure_<N>.png`` — a PIL-composited raster (loom's foreignObject SVGs
    balloon to megabytes; the raster is an order of magnitude smaller and is
    what the Visualizations tab / snapshot embeds).

It then registers the composite as a ``visualizations:`` entry on the study
(``image:visualizations/figure_<N>.svg``, idempotently). That rides the existing
study-viz pipeline with NO workbench change: the composite shows in the study's
Visualizations tab, is downloadable via the Runs "Viz" button, flows into the
SPA-composed investigation report, and is staged into the read-only snapshot —
downloadable both live and read-only. The stitch step skips any panel named
``figure_*`` so re-running never composites its own output.

Run:  python scripts/build_paper_figures.py
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

WS = Path(__file__).resolve().parents[1]
INV = WS / "investigations" / "paper-figures"

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
    the declared image, since loom SVGs don't embed reliably).

    Returns None for the stitched composite itself (``figure_*``) so re-running
    the step never feeds a figure back in as one of its own panels."""
    addr = str(viz.get("address", ""))
    if not addr.startswith("image:"):
        return None
    rel = addr[len("image:"):]
    p = study_dir / rel
    if p.name.startswith("figure_"):
        return None
    png = p.with_suffix(".png")
    if png.exists():
        return png
    return p if p.exists() and p.suffix == ".png" else None


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _viz_dir(study: str) -> Path:
    return WS / "studies" / study / "visualizations"


def _shelf(study: str):
    """Resolve a study's panels and pack them into shelf-layout rows.

    Returns (rows, fig_w, fig_h) where rows is a list of rows, each a list of
    (path, scaled_w, scaled_h). Returns None when the study has no panels."""
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
    return rows, fig_w, fig_h


def build_figure(study: str) -> Path | None:
    """Vector figure — publication-ready SVG (panels embedded as PNGs, layout +
    labels vector). Written into the study's own visualizations/ dir."""
    shelf = _shelf(study)
    if shelf is None:
        return None
    rows, fig_w, fig_h = shelf

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

    vd = _viz_dir(study)
    vd.mkdir(parents=True, exist_ok=True)
    out = vd / f"figure_{_fig_num(study)}.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


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


def build_figure_png(study: str) -> Path | None:
    """Raster figure — same shelf layout as build_figure, composited with PIL.

    This is what the Visualizations tab / snapshot embeds: loom's per-panel SVGs
    balloon to megabytes each (foreignObject + inlined computed styles), so the
    composite's PNG sibling keeps the study-viz payload small. The panels are
    already raster, so a composited PNG loses nothing visible and is an order of
    magnitude smaller — practical to serve live and bundle into the snapshot."""
    shelf = _shelf(study)
    if shelf is None:
        return None
    rows, fig_w, fig_h = shelf

    canvas = Image.new("RGB", (round(fig_w), round(fig_h)), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    font = _label_font(30)
    li = 0
    y = float(PAD)
    for row in rows:
        x = float(PAD)
        for p, sw, sh in row:
            label = f"{chr(ord('a') + li)}."
            li += 1
            draw.text((round(x), round(y)), label, fill="#111827", font=font)
            with Image.open(p) as im:
                panel = im.convert("RGBA").resize((round(sw), round(sh)), Image.LANCZOS)
                canvas.paste(panel, (round(x), round(y + LABEL_H)), panel)
            x += sw + GAP
        y += ROW_H + LABEL_H + GAP

    vd = _viz_dir(study)
    vd.mkdir(parents=True, exist_ok=True)
    out = vd / f"figure_{_fig_num(study)}.png"
    canvas.save(out, "PNG", optimize=True)
    return out


def register_stitched_viz(study: str) -> bool:
    """Add (idempotently) the stitched composite as a ``visualizations:`` entry on
    the study, so it rides the existing study-viz pipeline. Addresses the SVG
    (with its PNG sibling), matching the study's existing loom-image entries.

    Returns True if study.yaml was written (added or moved-to-end)."""
    n = _fig_num(study)
    spec_f = WS / "studies" / study / "study.yaml"
    spec = yaml.safe_load(spec_f.read_text()) or {}
    viz = list(spec.get("visualizations") or [])
    entry = {
        "name": f"Figure {n} (composite)",
        "address": f"image:visualizations/figure_{n}.svg",
        "chart": "image",
    }
    # Idempotent: drop any prior composite entry, then append a fresh one last so
    # the full figure reads as the study's culminating visualization.
    kept = [
        v for v in viz
        if not (isinstance(v, dict) and str(v.get("address", "")).endswith(f"figure_{n}.svg"))
    ]
    new = kept + [entry]
    if new == viz:
        return False  # already present + last; nothing to do
    spec["visualizations"] = new
    spec_f.write_text(
        yaml.safe_dump(spec, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return True


# Studies whose figure is composed from a hand-designed layout (fixed regions,
# panels placed by a dedicated module) rather than the default shelf-stitch.
# Fig 1 keeps a scaffold SVG (rasterized via node); Fig 7 places its panels into
# the paper's fixed 4-region grid and emits both SVG + PNG itself.
SCAFFOLD_STUDIES = {"fig-01": "build_figure1", "fig-07": "build_figure7"}


def _build_scaffold_figure(study: str) -> bool:
    """Compose a scaffold/hand-laid figure via its dedicated module. If the module
    exposes a `build_figure<N>_png` twin it emits the PNG itself; otherwise the SVG
    is rasterized via node (best-effort — the SVG is the source of truth)."""
    import importlib
    import subprocess
    mod = importlib.import_module(SCAFFOLD_STUDIES[study])
    n = _fig_num(study)
    build = getattr(mod, f"build_figure{n}")
    out = build()  # writes studies/<study>/visualizations/figure_<N>.svg
    png_builder = getattr(mod, f"build_figure{n}_png", None)
    if png_builder is not None:
        png_builder()  # module composites its own PNG twin
    else:
        png = out.with_suffix(".png")
        try:
            subprocess.run(
                ["node", str(Path(__file__).resolve().parent / "rasterize_svg.mjs"), str(out), str(png), "2"],
                check=True, capture_output=True, timeout=240)
        except Exception as exc:  # node/playwright missing → keep the SVG, warn
            print(f"     (figure PNG not rasterized: {exc}; run scripts/rasterize_svg.mjs)")
    return out.is_file()


def main() -> None:
    inv = yaml.safe_load((INV / "investigation.yaml").read_text()) or {}
    studies = inv.get("studies") or []
    if not studies:
        print("no member studies in paper-figures investigation")
        sys.exit(1)
    built = 0
    for s in studies:
        if s in SCAFFOLD_STUDIES:
            if _build_scaffold_figure(s):
                register_stitched_viz(s)
                print(f"  OK  Figure {_fig_num(s)} -> scaffold-composed ({SCAFFOLD_STUDIES[s]}.py)")
                built += 1
            else:
                print(f"  skip {s}: scaffold compose failed")
            continue
        svg = build_figure(s)
        png = build_figure_png(s)
        if svg and png:
            wrote = register_stitched_viz(s)
            print(f"  OK  Figure {_fig_num(s)} -> {svg.relative_to(WS)} + {png.name}"
                  f"{'  (+viz entry)' if wrote else ''}")
            built += 1
        else:
            print(f"  skip {s}: no panels")
    print(f"stitched {built}/{len(studies)} figures into their studies' visualizations")


if __name__ == "__main__":
    main()
