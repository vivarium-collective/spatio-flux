"""
Figure B: run mega_composite and assemble a multicomponent figure.

Layout:
  Rows 0–1: (a) {TEST_NAME}_viz.png (full width; spans 2 rows)
  Row 2:    (b) *_snapshots.png (full width)
  Row 3:    (c) *_mass.png | (d) *_timeseries.png
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
TEST_NAME = "mega_composite"
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
def run_mega_composite_and_plots() -> None:
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
    # Resolve panel PNGs once
    pngs: Dict[str, Optional[Path]] = {
        "viz": _panel_png_path(out_dir, test_name, "viz"),
        "snapshots": _panel_png_path(out_dir, test_name, "snapshots"),
        "mass": _panel_png_path(out_dir, test_name, "mass"),
        "timeseries": _panel_png_path(out_dir, test_name, "timeseries"),
    }

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(
        nrows=NROWS,
        ncols=NCOLS,
        height_ratios=ROW_HEIGHTS,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    for p in PLACEMENTS:
        ax = fig.add_subplot(gs[p.row : p.row + p.row_span, p.col : p.col + p.col_span])
        _draw_panel(ax, pngs.get(p.kind), missing_text=f"Missing {p.kind} panel")
        _label_panel(ax, p.key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"\n🖼️ Saved: {out_path}")


def main() -> None:
    run_mega_composite_and_plots()
    assemble_figure_B(out_dir=OUT_DIR, test_name=TEST_NAME, out_path=OUT_FIGURE)


if __name__ == "__main__":
    main()
