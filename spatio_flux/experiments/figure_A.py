"""
Run selected test_suite tests and assemble a multi-component figure with a custom layout.

Layout:
Row 1: (a) monod_kinetics bigraph | (b) monod_kinetics output | (c) ecoli_core_dfba output
Row 2: (d) community_dfba bigraph | (e) community_dfba output
Row 3: (f) comets bigraph         | (g) comets output
Row 4: (h) dfba_brownian_particles bigraph | (i) dfba_brownian_particles output

Notes:
- No per-panel titles like "X -- bigraph" / "X -- results"
- No overall suptitle
- Panels are lettered (a.), (b.), ... in the corner
- Uses test_suite to run tests and uses the generated PNGs in out/
"""

from __future__ import annotations

import os
import time
from typing import Optional, List, Tuple

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
OUT_DIR = "out"
OUT_FIGURE = os.path.join(OUT_DIR, "multicomponent.png")

# Tests we actually need to run to generate the required PNGs
TESTS_TO_RUN = [
    "monod_kinetics",
    "ecoli_core_dfba",
    "community_dfba",
    "comets",
    "dfba_brownian_particles",
]

# Which output png to pick per test (explicit mapping gives you maximum control)
# Set to None to auto-pick using RESULT_PNG_SUFFIX_PREFERENCE.
RESULT_PNG_BY_TEST = {
    "monod_kinetics": None,
    "ecoli_core_dfba": None,
    "community_dfba": None,
    "comets": None,
    "dfba_brownian_particles": None,
}

# Auto-pick order (if RESULT_PNG_BY_TEST[test] is None)
RESULT_PNG_SUFFIX_PREFERENCE = [
    "_snapshots.png",
    "_timeseries.png",
    "_mass.png",
    ".png",
]

N_COLS = 6   # virtual columns

# Custom layout specification:
# Each entry: (panel_letter, kind, test_name)
# kind in {"viz", "result"}
# Each entry: (letter, kind, test, col_start, col_span)
LAYOUT_ROWS = [
    # Row 1: Monod + DFBA (normal)
    [
        ("a", "viz",    "monod_kinetics",    0, 2),
        ("b", "result", "monod_kinetics",    2, 2),
        ("c", "result", "ecoli_core_dfba",   4, 2),
    ],

    # Row 2: community_dfba (wide bigraph)
    [
        ("d", "viz",    "community_dfba",    0, 4),  # wider
        ("e", "result", "community_dfba",    4, 2),
    ],

    # Row 3: comets (normal)
    [
        ("f", "viz",    "comets",             0, 2),
        ("g", "result", "comets",             2, 4),
    ],

    # Row 4: dfba brownian particles
    [
        ("h", "viz",    "dfba_brownian_particles", 0, 3),
        ("i", "result", "dfba_brownian_particles", 3, 3),
    ],
]


# Figure sizing control:
# - panel_size controls each panel width/height in inches
# - you can tweak wspace/hspace as needed
PANEL_SIZE = (4.2, 3.2)  # (width, height) per panel
WSPACE = 0.04
HSPACE = 0.10
DPI = 300

# Panel letter label styling
LABEL_KWARGS = dict(
    fontsize=14,
    fontweight="bold",
    ha="left",
    va="top",
    color="black",
)
LABEL_PAD = (0.02, 0.98)  # (x,y) in axes fraction


# -------------------------
# Helpers
# -------------------------
def _load_img(path: str):
    return mpimg.imread(path)


def _pick_result_png(test_name: str) -> Optional[str]:
    # explicit override path?
    explicit = RESULT_PNG_BY_TEST.get(test_name)
    if explicit:
        return explicit if os.path.exists(explicit) else None

    sim_info = SIMULATIONS[test_name]
    base = (sim_info.get("plot_config", {}) or {}).get("filename", test_name)

    for suffix in RESULT_PNG_SUFFIX_PREFERENCE:
        candidate = os.path.join(OUT_DIR, f"{base}{suffix}")
        if os.path.exists(candidate):
            return candidate

    return None


def _viz_png_path(test_name: str) -> str:
    return os.path.join(OUT_DIR, f"{test_name}_viz.png")


def _panel_png_path(kind: str, test_name: str) -> Optional[str]:
    if kind == "viz":
        p = _viz_png_path(test_name)
        return p if os.path.exists(p) else None
    if kind == "result":
        return _pick_result_png(test_name)
    raise ValueError(f"Unknown kind: {kind}")


# -------------------------
# Main pipeline
# -------------------------
def run_tests() -> None:
    prepare_output_dir(OUT_DIR)
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
        results = run_composite_document(doc, core=core, name=name, time=runtime, outdir=OUT_DIR)
        dt = time.time() - t0
        print(f"✅ Completed sim: {name} in {dt:.2f}s")

        plot_config = sim_info.get("plot_config", {}) or {}
        sim_info["plot_func"](results, doc.get("state", doc), config=plot_config)
        print("✅ Plots done.")



def assemble_multicomponent_figure() -> None:
    n_rows = len(LAYOUT_ROWS)
    n_cols = N_COLS

    fig_w = PANEL_SIZE[0] * (n_cols / 2)   # scale width sensibly
    fig_h = PANEL_SIZE[1] * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    axes_by_letter = {}

    for r, row in enumerate(LAYOUT_ROWS):
        # Turn entire row off first
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")

        # Populate requested panels
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

    # ---- Panel labels (figure coords, consistent)
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
    assemble_multicomponent_figure()
