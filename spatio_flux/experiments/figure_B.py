"""
Figure B: run the spatioflux reference composite and assemble a multicomponent figure.

Layout:
  Rows 0–1: (a) {TEST_NAME}_viz.png (full width)
  Row 2:    (b) *_snapshots.png (full width)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image, ImageDraw, ImageFont
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as fm

from spatio_flux.experiments.test_suite import (  # type: ignore
    SIMULATIONS,
    DEFAULT_RUNTIME_LONG,
    allocate_core,
    prepare_output_dir,
    run_composite_document,
)

# -------------------------
# Config
# -------------------------
TEST_NAME = "spatioflux_reference_demo"
OUT_DIR = Path("out")
OUT_FIGURE = OUT_DIR / "mega-composite.png"

DPI = 300
FIGSIZE = (10, 12)
WSPACE = 0.06
HSPACE = 0.10

LABEL_BBOX = dict(
    boxstyle="round,pad=0.15",
    facecolor="white",
    edgecolor="none",
    alpha=0.9,
)

# Grid configuration
NROWS = 4
NCOLS = 2
ROW_HEIGHTS = [1.0, 1.0, 1.0, 1.0]  # tweak if you want


# -------------------------
# Run simulation + generate panels
# -------------------------
def run_reference_composite_and_plots() -> None:
    prepare_output_dir(str(OUT_DIR))
    core = allocate_core()

    if TEST_NAME not in SIMULATIONS:
        raise KeyError(f"SIMULATIONS has no entry '{TEST_NAME}'")

    sim_info = SIMULATIONS[TEST_NAME]
    config = sim_info.get("config", {}) or {}

    print(f"\n🚀 Running test: {TEST_NAME}")
    doc = sim_info["doc_func"](core=core, config=config)

    runtime = sim_info.get("time", DEFAULT_RUNTIME_LONG)
    t0 = time.time()
    results = run_composite_document(
        doc,
        core=core,
        name=TEST_NAME,
        time=runtime,
        outdir=str(OUT_DIR),
    )
    dt = time.time() - t0
    print(f"✅ Completed sim: {TEST_NAME} in {dt:.2f}s")

    plot_config = sim_info.get("plot_config", {}) or {}
    sim_info["plot_func"](results, doc.get("state", doc), config=plot_config)
    print("✅ Plots done.")


# -------------------------
# File resolution helpers
# -------------------------
def _load_img(path: Path):
    return mpimg.imread(str(path))


def _find_first_by_suffix(out_dir: Path, suffix: str) -> Optional[Path]:
    matches = sorted(out_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _panel_png_path(out_dir: Path, test_name: str, kind: str) -> Optional[Path]:
    """
    kind:
      - "viz"         -> {test_name}_viz.png
      - "snapshots"   -> first * _snapshots.png
      - "mass"        -> first * _mass.png
      - "timeseries"  -> first * _timeseries.png
    """
    if kind == "viz":
        p = out_dir / f"{test_name}_viz.png"
        return p if p.exists() else None
    if kind == "snapshots":
        return _find_first_by_suffix(out_dir, "_snapshots.png")
    if kind == "mass":
        return _find_first_by_suffix(out_dir, "_submasses.png")
    if kind == "timeseries":
        return _find_first_by_suffix(out_dir, "_timeseries.png")
    raise ValueError(f"Unknown kind: {kind}")


def _draw_panel(ax, png: Optional[Path], missing_text: str) -> None:
    ax.axis("off")
    if png and png.exists():
        ax.imshow(_load_img(png), interpolation="nearest")
    else:
        ax.text(0.5, 0.5, missing_text, ha="center", va="center", fontsize=11)


def _label_panel(ax, letter: str) -> None:
    ax.text(
        0.01,
        0.99,
        f"{letter}.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox=LABEL_BBOX,
    )


def _draw_panel_label(
    img: Image.Image,
    label: str,
    *,
    x: int,
    y: int,
    region_width: int,
    frac: float = 0.02,   # % of panel width
):
    draw = ImageDraw.Draw(img)

    # Scale font to PANEL width, not canvas
    font_size = int(max(48, region_width * frac))

    # Robust font resolution: use Matplotlib's DejaVu
    font_path = fm.findfont("DejaVu Sans", fallback_to_default=True)
    font = ImageFont.truetype(font_path, font_size)

    text = f"{label}."
    bbox = draw.textbbox((x, y), text, font=font)
    pad = int(font_size * 0.3)

    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(255, 255, 255, 235),
    )
    draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)

# -------------------------
# Placement-based assembler (fixes duplicate viz)
# -------------------------
@dataclass(frozen=True)
class Placement:
    key: str          # 'a', 'b', 'c', 'd'
    kind: str         # 'viz', 'snapshots', 'mass', 'timeseries'
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1


PLACEMENTS: List[Placement] = [
    # (a) viz spans top 2 rows, full width
    Placement("a", "viz", row=0, col=0, row_span=2, col_span=2),
    # (b) snapshots full width
    Placement("b", "snapshots", row=2, col=0, row_span=2, col_span=2),
    # # (c) mass left, (d) timeseries right
    # Placement("c", "mass", row=3, col=0, row_span=1, col_span=1),
    # Placement("d", "timeseries", row=3, col=1, row_span=1, col_span=1),
]



def assemble_figure_B(
    out_dir: Path,
    test_name: str,
    out_path: Path,
) -> None:
    # -------------------------
    # Resolve panel PNGs once
    # -------------------------
    pngs: Dict[str, Optional[Path]] = {
        "viz": _panel_png_path(out_dir, test_name, "viz"),
        "snapshots": _panel_png_path(out_dir, test_name, "snapshots"),
        "mass": _panel_png_path(out_dir, test_name, "mass"),
        "timeseries": _panel_png_path(out_dir, test_name, "timeseries"),
    }

    def load_rgba(p: Optional[Path]) -> Optional[Image.Image]:
        if p and p.exists():
            return Image.open(p).convert("RGBA")
        return None

    imgs: Dict[str, Optional[Image.Image]] = {k: load_rgba(v) for k, v in pngs.items()}

    # require at least viz + snapshots for your current placements
    if imgs.get("viz") is None or imgs.get("snapshots") is None:
        raise FileNotFoundError("Missing viz or snapshots panel PNG(s).")

    # -------------------------
    # Layout params (pixel-based)
    # -------------------------
    # Choose a "column width" so 2 columns roughly match your widest full-width panel.
    # Full-width panels span 2 columns, so total width = 2*col_w + gap
    gap_x = 30
    gap_y = 40
    outer_pad = 30

    # Use viz width as the target full-width by default (you can change to max of all).
    viz_w = imgs["viz"].width
    full_w = viz_w  # could use max(img.width for img in imgs.values() if img)
    col_w = max(1, (full_w - gap_x) // 2)

    # Row heights: use your ROW_HEIGHTS weights, but in pixels.
    # We'll pick a base row height from snapshots height / row_span if available.
    # (Your snapshots spans 2 rows in PLACEMENTS as written.)
    base_h_guess = max(1, imgs["snapshots"].height // 2)
    total_weight = sum(ROW_HEIGHTS)
    # Scale so that sum(row_h) ~ base_h_guess * NROWS
    scale = (base_h_guess * NROWS) / total_weight
    row_h = [int(max(1, w * scale)) for w in ROW_HEIGHTS]

    # Canvas size in pixels
    canvas_w = outer_pad * 2 + (col_w * NCOLS) + gap_x * (NCOLS - 1)
    canvas_h = outer_pad * 2 + sum(row_h) + gap_y * (NROWS - 1)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    # Precompute row/col origin positions
    col_x0 = []
    x = outer_pad
    for c in range(NCOLS):
        col_x0.append(x)
        x += col_w + (gap_x if c < NCOLS - 1 else 0)

    row_y0 = []
    y = outer_pad
    for r in range(NROWS):
        row_y0.append(y)
        y += row_h[r] + (gap_y if r < NROWS - 1 else 0)

    # Helper: size of a placement region
    def region_rect(p: Placement):
        x0 = col_x0[p.col]
        y0 = row_y0[p.row]
        w = sum(col_w for _ in range(p.col_span)) + gap_x * (p.col_span - 1)
        h = sum(row_h[p.row + i] for i in range(p.row_span)) + gap_y * (p.row_span - 1)
        return x0, y0, w, h

    # Helper: resize to fill region (like matplotlib imshow in an axis)
    # - "cover": fills region completely, may crop (usually what people expect visually)
    # - "contain": fits inside region, may letterbox (keeps full image)
    RESIZE_MODE = "contain"  # change to "cover" for crop behavior

    def paste_resized(img: Image.Image, x0: int, y0: int, w: int, h: int):
        if RESIZE_MODE == "contain":
            # preserve entire image; may letterbox
            scale = min(w / img.width, h / img.height)
            nw = max(1, int(img.width * scale))
            nh = max(1, int(img.height * scale))
            resized = img.resize((nw, nh), resample=Image.LANCZOS)
            px = x0 + (w - nw) // 2
            py = y0 + (h - nh) // 2
            canvas.paste(resized, (px, py), resized)
            return

        # "cover": fill region; crop overflow
        scale = max(w / img.width, h / img.height)
        nw = max(1, int(math.ceil(img.width * scale)))
        nh = max(1, int(math.ceil(img.height * scale)))
        resized = img.resize((nw, nh), resample=Image.LANCZOS)

        # crop to exact region
        left = (nw - w) // 2
        top = (nh - h) // 2
        cropped = resized.crop((left, top, left + w, top + h))
        canvas.paste(cropped, (x0, y0), cropped)

    # -------------------------
    # Draw each placement
    # -------------------------
    for p in PLACEMENTS:
        img = imgs.get(p.kind)
        x0, y0, w, h = region_rect(p)

        if img is None:
            # simple missing box
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([x0, y0, x0 + w, y0 + h], outline=(0, 0, 0, 255), width=3)
            draw.text((x0 + 20, y0 + 20), f"Missing {p.kind}", fill=(0, 0, 0, 255))
        else:
            paste_resized(img, x0, y0, w, h)

        # label in top-left of the region (bigger + consistent)
        _draw_panel_label(
            canvas,
            p.key,
            x=x0 + 18,
            y=y0 + 18,
            region_width=w,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, format="PNG", optimize=True)
    print(f"\n🖼️ Saved: {out_path}")



def main() -> None:
    run_reference_composite_and_plots()
    assemble_figure_B(out_dir=OUT_DIR, test_name=TEST_NAME, out_path=OUT_FIGURE)


if __name__ == "__main__":
    main()
