"""
Run default test_suite tests and assemble a multi-component figure.

For each test:
  Left panel:  out/{test}_viz.png
  Right panel: one "result" PNG (picked from a preference list)

Output:
  out/multicomponent.png
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from test_suite import (  # type: ignore
    SIMULATIONS,
    DEFAULT_RUNTIME_LONG,
    allocate_core,
    prepare_output_dir,
    run_composite_document,
)


# -------------------------
# Config: edit as desired
# -------------------------
OUT_DIR = "out"
OUT_FIGURE = os.path.join(OUT_DIR, "multicomponent.png")

DEFAULT_TESTS = [
    "monod_kinetics",
    "ecoli_core_dfba",
    "community_dfba",
    "dfba_kinetics_community",
    # "comets",
    # "dfba_brownian_particles",
]

# For the "one test result png" panel, we try these (in order),
# using plot_config['filename'] as base.
RESULT_PNG_SUFFIX_PREFERENCE = [
    "_snapshots.png",
    "_timeseries.png",
    "_mass.png",
    # "_model_grid.png",
    ".png",
]


# -------------------------
# Helpers
# -------------------------
def _pick_result_png(test_name: str) -> Optional[str]:
    sim_info = SIMULATIONS[test_name]
    base = (sim_info.get("plot_config", {}) or {}).get("filename", test_name)

    for suffix in RESULT_PNG_SUFFIX_PREFERENCE:
        candidate = os.path.join(OUT_DIR, f"{base}{suffix}")
        if os.path.exists(candidate):
            return candidate

    return None


def _load_img(path: str):
    return mpimg.imread(path)


# -------------------------
# Main pipeline
# -------------------------
def run_default_tests() -> None:
    prepare_output_dir(OUT_DIR)
    core = allocate_core()

    for name in DEFAULT_TESTS:
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

        # Generate the existing plots (including the “result png” you want)
        plot_config = sim_info.get("plot_config", {}) or {}
        sim_info["plot_func"](results, doc.get("state", doc), config=plot_config)
        print("✅ Plots done.")


def assemble_multicomponent_figure() -> None:
    rows = len(DEFAULT_TESTS)
    cols = 2  # viz + result

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    if rows == 1:
        axes = [axes]  # normalize indexing

    for i, test_name in enumerate(DEFAULT_TESTS):
        ax_viz = axes[i][0]
        ax_res = axes[i][1]

        # --- Left: bigraph viz ---
        viz_path = os.path.join(OUT_DIR, f"{test_name}_viz.png")
        ax_viz.axis("off")
        ax_viz.set_title(f"{test_name} — bigraph", fontsize=10)

        if os.path.exists(viz_path):
            ax_viz.imshow(_load_img(viz_path))
        else:
            ax_viz.text(0.5, 0.5, f"Missing:\n{os.path.basename(viz_path)}",
                        ha="center", va="center")

        # --- Right: one result png ---
        res_path = _pick_result_png(test_name)
        ax_res.axis("off")
        ax_res.set_title(f"{test_name} — result", fontsize=10)

        if res_path is not None:
            ax_res.imshow(_load_img(res_path))
        else:
            ax_res.text(0.5, 0.5, "Missing result PNG\n(check plot_config['filename'] & naming)",
                        ha="center", va="center")

    fig.suptitle("Multi-component figure: bigraph + one result plot per test", fontsize=14, y=0.995)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n🖼️ Saved: {OUT_FIGURE}")


if __name__ == "__main__":
    run_default_tests()
    assemble_multicomponent_figure()
