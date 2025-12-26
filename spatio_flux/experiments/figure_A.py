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
import argparse
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
    "brownian_particles",
]

RESULT_PNG_BY_TEST = {
    "monod_kinetics": OUT_DIR / "monod_kinetics.png",
    "ecoli_core_dfba": OUT_DIR / "ecoli_core_dfba.png",
    "community_dfba": OUT_DIR / "community_dfba.png",
    "comets_diffusion": OUT_DIR / "comets_diffusion_snapshots.png",
    "brownian_particles": OUT_DIR / "brownian_particles_particles_traces.png",
}

RESULT_PNG_SUFFIX_PREFERENCE = [
    "_particles_traces.png"  # top one is preferred if exists
    "_snapshots.png",
    "_timeseries.png",
    "_mass.png",
    ".png",
]

N_COLS = 6

LAYOUT_PANELS = [
    # letter, kind, name, r0, rowspan, c0, colspan
    # ("a", "process_overview", "process_overview",   0, 1, 0, 6),
    ("a", "viz",    "community_dfba",               0, 1, 0, 6),

    ("b", "result", "monod_kinetics",               1, 1, 0, 2),
    ("c", "result", "ecoli_core_dfba",              1, 1, 2, 2),
    ("d", "result", "community_dfba",               1, 1, 4, 2),

    ("e", "viz",    "comets_diffusion",             2, 1, 0, 2),
    ("f", "result", "comets_diffusion",             2, 1, 2, 4),

    ("g", "viz",    "brownian_particles",           3, 2, 0, 3),  # spans 2 rows (taller)
    ("h", "result", "brownian_particles",           3, 2, 3, 3),
]

ROW_HEIGHT_RATIOS = [
    1.2,  # row 0 (a)
    0.9,  # row 1 (b,c,d)
    1.2,  # row 2 (e,f)
    0.7,  # row 3 (g,h)  ← shrink
    0.7,  # row 4 (g,h)  ← shrink
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

def have_outputs_for_test(name: str) -> bool:
    """
    True if the expected panels for this test already exist on disk.
    (viz + result if those are used in your layout)
    """
    viz_ok = _viz_png_path(name).exists()
    result_ok = _pick_result_png(name) is not None
    return viz_ok and result_ok

# -------------------------
# Pipeline: run tests
# -------------------------
def run_tests(*, skip_existing: bool = False) -> None:
    prepare_output_dir(str(OUT_DIR))
    core = allocate_core()

    for name in TESTS_TO_RUN:
        if name not in SIMULATIONS:
            print(f"⚠️  Unknown test '{name}' (skipping)")
            continue

        if skip_existing and have_outputs_for_test(name):
            print(f"⏭️  Skipping '{name}' (outputs already exist)")
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
def assemble_multicomponent_figure(layout_panels) -> None:
    # total rows needed = max(r0 + rowspan)
    n_rows = max(r0 + rs for (_, _, _, r0, rs, _, _) in layout_panels)
    n_cols = N_COLS

    fig_w = PANEL_SIZE[0] * (n_cols / 2)
    fig_h = PANEL_SIZE[1] * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        height_ratios=ROW_HEIGHT_RATIOS,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    # Optional: pre-turn everything off once (not strictly required)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")

    axes_by_letter = {}

    for (letter, kind, name, r0, rowspan, c0, colspan) in layout_panels:
        ax = fig.add_subplot(gs[r0:r0 + rowspan, c0:c0 + colspan])
        ax.axis("off")

        png_path = _panel_png_path(kind, name)
        if png_path and os.path.exists(png_path):
            ax.imshow(_load_img(png_path), interpolation="nearest")
        else:
            ax.text(
                0.5, 0.5,
                f"Missing panel\n{name}",
                ha="center", va="center",
                fontsize=11,
            )

        axes_by_letter[letter] = ax

    # labels
    for letter, ax in axes_by_letter.items():
        ax.text(
            -0.06, 1.02,  # a bit left + above the axes
            f"{letter}.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            bbox=LABEL_BBOX,
            clip_on=False,  # allow drawing outside the axes box
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_FIGURE, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"\n🖼️ Saved: {OUT_FIGURE}")


def parse_args():
    p = argparse.ArgumentParser(description="Run spatio_flux tests and/or assemble a multi-panel figure.")
    p.add_argument(
        "--mode",
        choices=["all", "run", "assemble"],
        default="all",
        help="all = run tests then assemble; run = run tests only; assemble = assemble only (no sims).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="When running tests, skip any test whose expected PNG outputs already exist.",
    )
    p.add_argument(
        "--ensure-overview",
        action="store_true",
        help="Generate process_overview.png if missing before assembly.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode in ("all", "run"):
        run_tests(skip_existing=args.skip_existing)

    if args.mode in ("all", "assemble"):
        if args.ensure_overview:
            ensure_process_overview()
        assemble_multicomponent_figure(LAYOUT_PANELS)

