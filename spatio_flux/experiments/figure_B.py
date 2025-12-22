"""
Figure B: run mega_composite and assemble a multicomponent figure.

Layout:
  Row 1–2: (a) mega_composite _viz.png (spans both rows, full width)
  Row 3:   (b) _mass.png | (c) _timeseries.png
  Row 4:   (d) _snapshots.png (full width)

Notes:
- Uses SIMULATIONS['mega_composite'] from spatio_flux.experiments.test_suite
- Runs the sim + its plot_func to generate PNGs in out/
- Assembles a single figure_B.png
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

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
WSPACE = 0.06
HSPACE = 0.10

LABEL_BBOX = dict(
    boxstyle="round,pad=0.15",
    facecolor="white",
    edgecolor="none",
    alpha=0.9,
)

# -------------------------
# Helpers
# -------------------------
def _load_img(path: Path):
    return mpimg.imread(str(path))


def _find_first_by_suffix(out_dir: Path, suffix: str) -> Optional[Path]:
    """Return the first png matching *suffix, or None."""
    matches = sorted(out_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _panel_png_path(out_dir: Path, test_name: str, kind: str) -> Optional[Path]:
    """
    Resolve panel file paths.

    kind:
      - "viz"         -> {test_name}_viz.png
      - "mass"        -> first * _mass.png
      - "timeseries"  -> first * _timeseries.png
      - "snapshots"   -> first * _snapshots.png
    """
    if kind == "viz":
        p = out_dir / f"{test_name}_viz.png"
        return p if p.exists() else None
    if kind == "mass":
        return _find_first_by_suffix(out_dir, "_mass.png")
    if kind == "timeseries":
        return _find_first_by_suffix(out_dir, "_timeseries.png")
    if kind == "snapshots":
        return _find_first_by_suffix(out_dir, "_snapshots.png")
    raise ValueError(f"Unknown kind: {kind}")


# -------------------------
# Main pipeline
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


def assemble_figure_B(out_dir: Path, test_name: str, out_path: Path) -> None:
    """
    Assemble Figure B with the layout:

      Row 1–2: (a) viz (full width)
      Row 3:   (b) mass | (c) timeseries
      Row 4:   (d) snapshots (full width)
    """
    viz_png = _panel_png_path(out_dir, test_name, "viz")
    mass_png = _panel_png_path(out_dir, test_name, "mass")
    timeseries_png = _panel_png_path(out_dir, test_name, "timeseries")
    snapshots_png = _panel_png_path(out_dir, test_name, "snapshots")

    fig = plt.figure(figsize=(10, 12), dpi=DPI)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[1.2, 1.2, 1.0, 1.2],
        hspace=HSPACE,
        wspace=WSPACE,
    )

    axes = {}

    # a: viz (rows 0–1, full width)
    ax_a = fig.add_subplot(gs[0:2, :])
    ax_a.axis("off")
    if viz_png and viz_png.exists():
        ax_a.imshow(_load_img(viz_png), interpolation="nearest")
    else:
        ax_a.text(0.5, 0.5, "Missing viz panel", ha="center", va="center", fontsize=11)
    axes["a"] = ax_a

    # b: mass
    ax_b = fig.add_subplot(gs[2, 0])
    ax_b.axis("off")
    if mass_png and mass_png.exists():
        ax_b.imshow(_load_img(mass_png), interpolation="nearest")
    else:
        ax_b.text(0.5, 0.5, "Missing mass plot", ha="center", va="center", fontsize=11)
    axes["b"] = ax_b

    # c: timeseries
    ax_c = fig.add_subplot(gs[2, 1])
    ax_c.axis("off")
    if timeseries_png and timeseries_png.exists():
        ax_c.imshow(_load_img(timeseries_png), interpolation="nearest")
    else:
        ax_c.text(0.5, 0.5, "Missing timeseries plot", ha="center", va="center", fontsize=11)
    axes["c"] = ax_c

    # d: snapshots (full width)
    ax_d = fig.add_subplot(gs[3, :])
    ax_d.axis("off")
    if snapshots_png and snapshots_png.exists():
        ax_d.imshow(_load_img(snapshots_png), interpolation="nearest")
    else:
        ax_d.text(0.5, 0.5, "Missing snapshots", ha="center", va="center", fontsize=11)
    axes["d"] = ax_d

    # Panel labels (consistent in figure coords)
    x_pad = 0.006
    y_pad = 0.010
    for letter, ax in axes.items():
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - x_pad,
            bbox.y1 + y_pad,
            f"{letter}.",
            transform=fig.transFigure,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox=LABEL_BBOX,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"\n🖼️ Saved: {out_path}")


def main() -> None:
    run_mega_composite_and_plots()
    assemble_figure_B(out_dir=OUT_DIR, test_name=TEST_NAME, out_path=OUT_FIGURE)


if __name__ == "__main__":
    main()
