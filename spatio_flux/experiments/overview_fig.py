# spatio_flux/experiments/overview_fig.py

import os
import io
import json
import math
import time
import glob
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from process_bigraph import register_types as register_process_types
from vivarium.vivarium import VivariumTypes
from bigraph_viz import plot_bigraph

# spatio-flux core types & helpers
from spatio_flux import register_types, TYPES_DICT
from spatio_flux.library.helpers import run_composite_document, prepare_output_dir

# import the simulations table and their doc/plot funcs
from spatio_flux.experiments.test_suite import (
    SIMULATIONS,
    DEFAULT_RUNTIME_LONG,  # used if a SIM has no explicit time
)

# ---------- Single-process docs (one node per plot) ----------

def get_dfba_single_doc(core=None, config=None):
    return {
        'dfba': {
            "_type": "process",
            "address": "local:DynamicFBA",
            "config": {"model_file": "textbook"},
        }
    }

def get_spatial_dfba_doc(core=None, config=None):
    return {
        'spatial_dfba': {
            "_type": "process",
            "address": "local:SpatialDFBA",
            "config": {"n_bins": (5, 10)},
        }
    }

def get_diffusion_advection_doc(core=None, config=None):
    return {
        'diffusion_advection': {
            "_type": "process",
            "address": "local:DiffusionAdvection",
            "config": {
                "n_bins": (5, 10),
                "bounds": (5.0, 10.0),
                "default_diffusion_rate": 1e-1,
                "default_diffusion_dt": 1e-1,
                "diffusion_coeffs": {},
                "advection_coeffs": {},
            },
        }
    }

def get_particles_doc(core=None, config=None):
    return {
        'particles': {
            "_type": "process",
            "address": "local:Particles",
            "config": {
                "n_bins": (5, 10),
                "bounds": (5.0, 10.0),
                "diffusion_rate": 1e-1,
                "advection_rate": (0.0, -0.1),
                "add_probability": 0.0,
            },
        }
    }

def get_minimal_kinetic_doc(core=None, config=None):
    return {
        'minimal_kinetic': {
            "_type": "process",
            "address": "local:MinimalParticle",
            "config": {},
        }
    }

PROCESS_DOCS = {
    'dfba_single': get_dfba_single_doc,
    'spatial_dfba_single': get_spatial_dfba_doc,
    'diffusion_advection_single': get_diffusion_advection_doc,
    'particles_single': get_particles_doc,
    'minimal_kinetic_single': get_minimal_kinetic_doc,
}

# ---------- Utilities ----------

def _ensure_outdir(outdir: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)

def _load_rgba(path: Path) -> Image.Image:
    img = Image.open(path)
    try:
        return img.convert("RGBA")
    except Exception:
        return img

def _grid_shape(n_items: int, max_cols: int) -> Tuple[int, int]:
    cols = min(max_cols, max(1, n_items))
    rows = math.ceil(n_items / cols)
    return rows, cols

def _panel_from_images(ax_array, images: List[Image.Image], titles: List[str] = None):
    for i, ax in enumerate(ax_array):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i])
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9, pad=3)

def _text_to_png(text: str, out_path: Path, width_px: int = 1200, pad: int = 8, max_lines: int = 200):
    """
    Render monospaced text to a PNG for inclusion in the composite figure.
    """
    wrapped = []
    for line in text.splitlines():
        wrapped.extend(textwrap.wrap(line, width=140) or [""])
        if len(wrapped) >= max_lines:
            wrapped = wrapped[:max_lines] + ["... (truncated)"]
            break

    fig_h = max(2.0, min(12.0, 0.18 * len(wrapped)))  # heuristic
    fig = plt.figure(figsize=(width_px / 100, fig_h), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.01, 0.99, "\n".join(wrapped), va="top", ha="left",
            family="monospace", fontsize=8)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=pad/100.0)
    plt.close(fig)

def _read_schema_json_text(json_path: Path) -> str:
    """
    Prefer showing 'composition' if present; otherwise dump the whole JSON.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if "composition" in data:
            return json.dumps(data["composition"], indent=2)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"⚠ Could not read schema JSON at {json_path}:\n{e}"

def _preferred_output_images(base_prefix: str, outdir: Path) -> List[Path]:
    """
    Return a list of simulation output images for a given test base name,
    skipping any *viz* images. Preference order:
      1) *_timeseries.png
      2) {base}.png
      3) other {base}*.png
      4) {base}*.gif (first frame used)
    """
    candidates: List[Path] = []

    # 1) *_timeseries.png
    candidates += sorted(outdir.glob(f"{base_prefix}_timeseries.png"))

    # 2) {base}.png
    p = outdir / f"{base_prefix}.png"
    if p.exists():
        candidates.append(p)

    # 3) any other {base}*.png (skip *_viz.png which are topology figures)
    others = [p for p in sorted(outdir.glob(f"{base_prefix}_*.png"))
              if "_viz" not in p.name and p not in candidates and not p.name.endswith("_timeseries.png")]
    candidates += others

    # 4) any {base}*.gif
    gifs = [p for p in sorted(outdir.glob(f"{base_prefix}_*.gif")) if "_viz" not in p.name]
    candidates += gifs

    # Remove dupes while preserving order
    seen = set()
    unique = []
    for p in candidates:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique

def _first_frame(path: Path) -> Image.Image:
    img = Image.open(path)
    try:
        img.seek(0)
    except Exception:
        pass
    return img.convert("RGBA")

# ---------- Figure assembly ----------

def assemble_overview_figure(
    outdir: str,
    process_pngs: List[Path],
    type_pngs: List[Path],
    sim_rows: List[Tuple[str, Path, Path, List[Path]]],
    figsize=(18, 14),
    save_name="overview_figure.png",
):
    """
    Build a composite figure with subpanels:
      a. Processes
      b. Types
      c. Selected SIMULATIONS (per row: schema, bigraph viz, example outputs)
    """
    outdir = Path(outdir)
    fig = plt.figure(figsize=figsize, dpi=200)

    # Compute grid sizes
    pr_rows, pr_cols = _grid_shape(len(process_pngs), max_cols=3)
    ty_rows, ty_cols = _grid_shape(len(type_pngs), max_cols=4)

    # c-panel: one row per sim, 3 columns (schema, viz, outputs)
    n_sims = len(sim_rows)
    heights = [max(pr_rows, 1), max(ty_rows, 1), max(n_sims, 1)]
    gs_root = GridSpec(3, 1, height_ratios=heights, hspace=0.22, figure=fig)

    # --- Panel a: Processes ---
    gs_a = GridSpecFromSubplotSpec(pr_rows, pr_cols, subplot_spec=gs_root[0], wspace=0.06, hspace=0.18)
    proc_axes = [fig.add_subplot(gs_a[i // pr_cols, i % pr_cols]) for i in range(pr_rows * pr_cols)]
    proc_images = [_load_rgba(p) for p in process_pngs]
    _panel_from_images(proc_axes, proc_images, titles=[p.stem for p in process_pngs])
    pos_a = gs_root[0].get_position(fig)
    fig.text(pos_a.x0, pos_a.y1 + 0.01, "a. Processes", fontsize=14, weight="bold", va="bottom", ha="left")

    # --- Panel b: Types ---
    gs_b = GridSpecFromSubplotSpec(ty_rows, ty_cols, subplot_spec=gs_root[1], wspace=0.06, hspace=0.18)
    type_axes = [fig.add_subplot(gs_b[i // ty_cols, i % ty_cols]) for i in range(ty_rows * ty_cols)]
    type_images = [_load_rgba(p) for p in type_pngs]
    _panel_from_images(type_axes, type_images, titles=[p.stem for p in type_pngs])
    pos_b = gs_root[1].get_position(fig)
    fig.text(pos_b.x0, pos_b.y1 + 0.01, "b. Types", fontsize=14, weight="bold", va="bottom", ha="left")

    # --- Panel c: SIMULATIONS (schema, bigraph, outputs) ---
    if n_sims > 0:
        gs_c = GridSpecFromSubplotSpec(n_sims, 3, subplot_spec=gs_root[2], wspace=0.08, hspace=0.24)

        for r, (sim_name, schema_png, viz_png, output_pngs) in enumerate(sim_rows):
            # col 0: schema
            ax0 = fig.add_subplot(gs_c[r, 0]); ax0.axis("off")
            if schema_png and schema_png.exists():
                ax0.imshow(_load_rgba(schema_png))
            ax0.set_title(f"{sim_name} schema", fontsize=10)

            # col 1: bigraph viz
            ax1 = fig.add_subplot(gs_c[r, 1]); ax1.axis("off")
            if viz_png and viz_png.exists():
                ax1.imshow(_load_rgba(viz_png))
            ax1.set_title(f"{sim_name} bigraph", fontsize=10)

            # col 2: preferred outputs (stack up to 2)
            ax2 = fig.add_subplot(gs_c[r, 2]); ax2.axis("off")
            shown = []
            for p in output_pngs:
                if p.suffix.lower() == ".gif":
                    shown.append(_first_frame(p))
                else:
                    shown.append(_load_rgba(p))
                if len(shown) == 2:
                    break

            if len(shown) == 0:
                ax2.text(0.5, 0.5, "No outputs found", ha="center", va="center")
            elif len(shown) == 1:
                ax2.imshow(shown[0])
            else:
                # simple vertical stack
                h = shown[0].height + shown[1].height
                w = max(shown[0].width, shown[1].width)
                stacked = Image.new("RGBA", (w, h), (255, 255, 255, 0))
                y = 0
                for im in shown:
                    if im.width < w:
                        im = ImageOps.pad(im, (w, im.height), color=(255, 255, 255, 0))
                    stacked.paste(im, (0, y))
                    y += im.height
                ax2.imshow(stacked)
            ax2.set_title(f"{sim_name} outputs", fontsize=10)

        pos_c = gs_root[2].get_position(fig)
        fig.text(pos_c.x0, pos_c.y1 + 0.01, "c. Composites (SIMULATIONS)", fontsize=14, weight="bold",
                 va="bottom", ha="left")

    # Save
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / save_name
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Overview saved to: {out_path}")
    print(f"Overview (PDF) saved to: {out_path.with_suffix('.pdf')}")

# ---------- Main ----------

def main():
    outdir = Path("out")
    prepare_output_dir(outdir)  # clears/recreates 'out'
    _ensure_outdir(outdir)

    # establish core type system
    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    # 1) Plot single processes
    process_pngs: List[Path] = []
    for name, get_doc in PROCESS_DOCS.items():
        doc = get_doc(core=core)
        fname = f"{name}_process"
        plot_bigraph(
            state=doc,
            core=core,
            out_dir=str(outdir),
            filename=fname,
            dpi="300",
            collapse_redundant_processes=True,
        )
        process_pngs.append(outdir / f"{fname}.png")

    # 2) Plot types (one per image)
    type_pngs: List[Path] = []
    for type_name, type_schema in TYPES_DICT.items():
        fname = f"{type_name}_type"
        plot_bigraph(
            state={type_name: type_schema},
            show_types=True,
            core=core,
            out_dir=str(outdir),
            filename=fname,
            dpi="300",
            collapse_redundant_processes=True,
        )
        type_pngs.append(outdir / f"{fname}.png")

    # 3) Select and RUN a subset of SIMULATIONS (like test_suite)
    #    You can tweak this list as you see fit.
    tests_to_run = [
        "ecoli_core_dfba",
        "spatial_dfba_process",
        "diffusion_process",
        "comets",
        "particles",
        # add more if desired...
    ]

    sim_rows: List[Tuple[str, Path, Path, List[Path]]] = []
    runtimes: Dict[str, float] = {}
    total_sim_time = 0.0

    for name in tests_to_run:
        print(f"\n🚀 Running test: {name}")
        if name not in SIMULATIONS:
            print(f"Skipping unknown test: '{name}'")
            continue

        sim_info = SIMULATIONS[name]

        # Create document
        print("Creating document...")
        config = sim_info.get("config", {})
        doc = sim_info["doc_func"](core=core, config=config)

        # Run and collect results
        print("Sending document...")
        runtime = sim_info.get("time", DEFAULT_RUNTIME_LONG)
        sim_start = time.time()
        results = run_composite_document(doc, core=core, name=name, time=runtime)
        sim_end = time.time()

        sim_elapsed = sim_end - sim_start
        runtimes[name] = sim_elapsed
        total_sim_time += sim_elapsed

        # Generate plots for the sim using its original plotter
        print("Generating plots...")
        plot_config = sim_info.get("plot_config", {})
        sim_info["plot_func"](results, doc.get("state", doc), config=plot_config)
        print(f"✅ Completed: {name} in {sim_elapsed:.2f} seconds")

        # 3a) Schema PNG (from {name}.json written by run_composite_document)
        schema_json = outdir / f"{name}.json"
        schema_png = outdir / f"{name}_schema.png"
        schema_text = _read_schema_json_text(schema_json)
        _text_to_png(schema_text, schema_png, width_px=1200, pad=6, max_lines=180)

        # 3b) Bigraph viz (always saved by helpers as {name}_viz.png)
        viz_png = outdir / f"{name}_viz.png"

        # 3c) Preferred output images (use plot_config filename if present; else the sim name)
        base_prefix = plot_config.get("filename") or name
        output_candidates = _preferred_output_images(base_prefix, outdir)

        sim_rows.append((name, schema_png, viz_png, output_candidates))

    # 4) Assemble master overview figure
    assemble_overview_figure(
        str(outdir),
        process_pngs,
        type_pngs,
        sim_rows=sim_rows,
        figsize=(18, 14),
        save_name="overview_figure.png",
    )


if __name__ == "__main__":
    main()
