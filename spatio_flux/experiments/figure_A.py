"""
Run selected test_suite tests and assemble a multi-component figure with a custom layout,
with an extra TOP ROW:

Row 0: (a) process_overview.png (from overview_fig.assemble_process_figures)
Row 1+: previous panels shift to (b), (c), ...

Notes:
- No per-panel titles
- No overall suptitle
- Panels are lettered (a.), (b.), ...
"""

from __future__ import annotations

import os
import time
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from spatio_flux.experiments.test_suite import (
    SIMULATIONS,
    DEFAULT_RUNTIME_LONG,
    allocate_core,
    prepare_output_dir,
    run_composite_document,
)

# generate the process overview panel
from spatio_flux.experiments.overview_fig import assemble_process_figures


# -------------------------
# Config
# -------------------------
OUT_DIR = Path("out")
OUT_FIGURE = OUT_DIR / "multicomponent_spatioflux.png"

PROCESS_OVERVIEW_PNG = OUT_DIR / "process_overview.png"

TESTS_TO_RUN = [
    "monod_kinetics",
    "ecoli_core_dfba",
    "community_dfba",
    "comets_diffusion",
    "br_particles_kinetics",
]

RESULT_PNG_BY_TEST = {
    "monod_kinetics": None,
    "ecoli_core_dfba": None,
    "community_dfba": None,
    "comets_diffusion": None,
    "br_particles_kinetics": None,
}

RESULT_PNG_SUFFIX_PREFERENCE = [
    "_snapshots.png",
    "_timeseries.png",
    "_mass.png",
    ".png",
]

N_COLS = 6  # unchanged

LAYOUT_ROWS = [
    # Row 1: b. community dFBA bigraph across full width
    [
        ("a", "viz", "community_dfba", 0, 6),
    ],

    # Row 2: c/d/e three outputs
    [
        ("b", "result", "monod_kinetics",  0, 2),
        ("c", "result", "ecoli_core_dfba", 2, 2),
        ("d", "result", "community_dfba",  4, 2),
    ],

    # Row 3: comets composite
    [
        ("e", "viz",    "comets_diffusion", 0, 3),
        ("f", "result", "comets_diffusion", 3, 6),
    ],

    # Row 4: particles composite
    [
        ("g", "viz",    "br_particles_kinetics", 0, 3),
        ("h", "result", "br_particles_kinetics", 3, 6),
    ],
]


PANEL_SIZE = (4.2, 3.2)  # inches, used for sizing
WSPACE = 0.04
HSPACE = 0.10
DPI = 300

LABEL_BBOX = dict(
    boxstyle="round,pad=0.15",
    facecolor="white",
    edgecolor="none",
    alpha=0.9,
)


# -------------------------
# Helpers: PNG resolution
# -------------------------
def _load_img(path: Path):
    return mpimg.imread(str(path))


def _viz_png_path(test_name: str) -> Path:
    return OUT_DIR / f"{test_name}_viz.png"


def _panel_png_path(kind: str, test_name: str) -> Optional[str]:
    if kind == "viz":
        p = _viz_png_path(test_name)
        return p if os.path.exists(p) else None

    if kind == "result":
        return _pick_result_png(test_name)

    if kind == "process_overview":
        p = PROCESS_OVERVIEW_PNG
        return p if os.path.exists(p) else None

    raise ValueError(f"Unknown panel kind: {kind}")


def _pick_result_png(test_name: str) -> Optional[Path]:
    explicit = RESULT_PNG_BY_TEST.get(test_name)
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    sim_info = SIMULATIONS[test_name]
    base = (sim_info.get("plot_config", {}) or {}).get("filename", test_name)

    for suffix in RESULT_PNG_SUFFIX_PREFERENCE:
        candidate = OUT_DIR / f"{base}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _static_png_path(name: str) -> Optional[Path]:
    if name == "process_overview":
        return PROCESS_OVERVIEW_PNG if PROCESS_OVERVIEW_PNG.exists() else None
    # extend here if you add more static panels
    return None


def resolve_panel_png(kind: str, name: str) -> Optional[Path]:
    if kind == "viz":
        p = _viz_png_path(name)
        return p if p.exists() else None
    if kind == "result":
        return _pick_result_png(name)
    if kind == "static":
        return _static_png_path(name)
    raise ValueError(f"Unknown kind: {kind}")


# -------------------------
# Pipeline: run tests
# -------------------------
def run_tests() -> None:
    prepare_output_dir(str(OUT_DIR))
    core = allocate_core()

    for name in TESTS_TO_RUN:
        if name not in SIMULATIONS:
            print(f"⚠️  Unknown test '{name}' (skipping)")
            continue

        sim_info = SIMULATIONS[name]
        config = sim_info.get("config", {}) or {}

        print(f"\n🚀 Running test: {name}")
        doc = sim_info["doc_func"](core=core, config=config)

        runtime = sim_info.get("time", DEFAULT_RUNTIME_LONG)
        t0 = time.time()
        results = run_composite_document(
            doc, core=core, name=name, time=runtime, outdir=str(OUT_DIR)
        )
        dt = time.time() - t0
        print(f"✅ Completed sim: {name} in {dt:.2f}s")

        plot_config = sim_info.get("plot_config", {}) or {}
        sim_info["plot_func"](results, doc.get("state", doc), config=plot_config)
        print("✅ Plots done.")


def ensure_process_overview(core=None) -> None:
    """
    Generate process_overview.png if missing.
    Keeps this script robust even if the overview hasn't been generated yet.
    """
    if PROCESS_OVERVIEW_PNG.exists():
        return
    core = core or allocate_core()
    print("\n🧩 Generating process overview panel...")
    assemble_process_figures(core, outdir=OUT_DIR, n_rows=2, save_name=PROCESS_OVERVIEW_PNG.name)


# -------------------------
# Figure assembler
# -------------------------
def assemble_multicomponent_figure(layout_rows) -> None:
    n_rows = len(layout_rows)
    n_cols = N_COLS

    fig_w = PANEL_SIZE[0] * (n_cols / 2)
    fig_h = PANEL_SIZE[1] * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    axes_by_letter = {}

    for r, row in enumerate(layout_rows):
        # turn entire row off
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")

        # populate requested panels
        for (letter, kind, test_name, c0, span) in row:
            ax = fig.add_subplot(gs[r, c0:c0 + span])
            ax.axis("off")

            png_path = _panel_png_path(kind, test_name)
            if png_path and os.path.exists(png_path):
                ax.imshow(_load_img(png_path), interpolation="nearest")
            else:
                ax.text(
                    0.5, 0.5,
                    f"Missing panel\n{test_name}",
                    ha="center", va="center",
                    fontsize=11,
                )

            axes_by_letter[letter] = ax

    # panel labels (same as you already have, using axes_by_letter)
    x_pad = 0.006
    y_pad = 0.010
    label_bbox = dict(
        boxstyle="round,pad=0.15",
        facecolor="white",
        edgecolor="none",
        alpha=0.9,
    )

    for letter, ax in axes_by_letter.items():
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
            bbox=label_bbox,
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_FIGURE, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"\n🖼️ Saved: {OUT_FIGURE}")


if __name__ == "__main__":
    run_tests()
    # ensure_process_overview()
    assemble_multicomponent_figure(LAYOUT_ROWS)
