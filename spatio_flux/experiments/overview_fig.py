# spatio_flux/experiments/overview_fig.py

import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

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
            "address": "local:MonodKinetics",
            "config": {},
        }
    }

def get_division_doc(core=None, config=None):
    return {
        'particle_division': {
            "_type": "process",
            "address": "local:ParticleDivision",
            "config": {
                "mass_threshold": 2.0,
            },
        }
    }

PROCESS_DOCS = {
    'dfba': get_dfba_single_doc,
    'spatial_dfba': get_spatial_dfba_doc,
    'diffusion_advection': get_diffusion_advection_doc,
    'minimal_kinetic': get_minimal_kinetic_doc,
    'particle_movement': get_particles_doc,
    'particle_division': get_division_doc,
}

# ---------- Utilities ----------

def _ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

def _load_rgba(path: Path) -> Image.Image | None:
    """Load an image as RGBA or return None if missing/unreadable."""
    try:
        with Image.open(path) as im:
            return im.convert("RGBA")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        print(f"⚠ Skipping missing/unreadable image: {path} ({e.__class__.__name__})")
        return None

def _grid_shape(n_items: int, max_cols: int) -> Tuple[int, int]:
    """Return (rows, cols) with cols in [1..max_cols] and rows >= 0 (may be 0 if n_items==0)."""
    cols = min(max_cols, max(1, n_items))
    rows = math.ceil(n_items / cols) if n_items > 0 else 0
    return rows, cols

def _panel_from_images(ax_array, images: List[Image.Image], titles: List[str] = None):
    for i, ax in enumerate(ax_array):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i])
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9, pad=3)

def _first_frame(path: Path) -> Image.Image | None:
    """Return the first frame of an animated image as RGBA, closing the file handle."""
    try:
        with Image.open(path) as im:
            try:
                im.seek(0)
            except Exception:
                pass
            return im.convert("RGBA")
    except Exception as e:
        print(f"⚠ Could not extract first frame: {path} ({e.__class__.__name__})")
        return None

def _preferred_output_images(base_prefix: str, outdir: Path) -> List[Path]:
    """
    Return the single best simulation output image for a given test base name.

    Preference order:
      1) *_snapshots.png
      2) *_timeseries.png
      3) {base}.png
      4) other {base}*.png (excluding *_viz.png and *_schema.png)
      5) {base}*.gif (excluding *_viz.gif)
    Always returns a list (possibly empty).
    """
    # 1) Prefer *_snapshots.png
    snap = outdir / f"{base_prefix}_snapshots.png"
    if snap.exists():
        return [snap]

    # 2) Then *_timeseries.png
    ts = outdir / f"{base_prefix}_timeseries.png"
    if ts.exists():
        return [ts]

    # 3) Then exact {base}.png
    base_png = outdir / f"{base_prefix}.png"
    if base_png.exists():
        return [base_png]

    # 4) Any other PNG (excluding viz/schema/snapshots/timeseries)
    for p in sorted(outdir.glob(f"{base_prefix}_*.png")):
        if "_viz" in p.name or "_schema" in p.name:
            continue
        if p.name.endswith("_snapshots.png") or p.name.endswith("_timeseries.png"):
            continue
        return [p]

    # 5) Fallback: first valid GIF (excluding viz), case-insensitive
    gif_candidates = sorted(outdir.glob(f"{base_prefix}_*.gif")) + sorted(outdir.glob(f"{base_prefix}_*.GIF"))
    for p in gif_candidates:
        if "_viz" not in p.name:
            return [p]

    return []

# ---------- Build sections ----------

def build_process_figs(core, outdir: Path) -> List[Path]:
    """Generate one-node process plots (panel a)."""
    _ensure_outdir(outdir)
    generated: List[Path] = []
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
        png = outdir / f"{fname}.png"
        if png.exists():
            generated.append(png)
    return generated

def build_type_figs(core, outdir: Path) -> List[Path]:
    """Generate individual type figures (panel b)."""
    _ensure_outdir(outdir)
    generated: List[Path] = []
    for type_name in sorted(TYPES_DICT.keys()):
        type_schema = TYPES_DICT[type_name]
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
        png = outdir / f"{fname}.png"
        if png.exists():
            generated.append(png)
    return generated

def run_selected_sims(core, outdir: Path, tests_to_run: List[str]) -> List[Tuple[str, Path, List[Path]]]:
    """
    Run a subset of SIMULATIONS and produce:
      - bigraph viz PNG path
      - curated list of output images (prefer png, fallback gif first frames)
    Returns list of tuples: (sim_name, viz_png_path, output_png_paths_or_gifs)
    """
    _ensure_outdir(outdir)
    results_summary: List[Tuple[str, Path, List[Path]]] = []

    for name in tests_to_run:
        print(f"\n🚀 Running test: {name}")
        if name not in SIMULATIONS:
            print(f"Skipping unknown test: '{name}'")
            continue

        sim_info = SIMULATIONS[name]

        print("Creating document...")
        config = sim_info.get("config", {})
        doc = sim_info["doc_func"](core=core, config=config)

        print("Sending document...")
        runtime = sim_info.get("time", DEFAULT_RUNTIME_LONG)
        sim_start = time.time()
        results = run_composite_document(doc, core=core, name=name, time=runtime)
        sim_end = time.time()
        print(f"✅ Completed: {name} in {sim_end - sim_start:.2f} s")

        print("Generating plots...")
        plot_config = sim_info.get("plot_config", {})
        sim_info["plot_func"](results, doc.get("state", doc), config=plot_config)

        # bigraph viz saved by run_composite_document as {name}_viz.png
        viz_png = outdir / f"{name}_viz.png"

        # Figure out best outputs from the sim's plotter naming
        base_prefix = plot_config.get("filename") or name
        output_candidates = _preferred_output_images(base_prefix, outdir)
        results_summary.append((name, viz_png, output_candidates or []))

    return results_summary

# ---------- Figure assembly ----------

def assemble_overview_figure(
    outdir: Path,
    process_pngs: List[Path],
    type_pngs: List[Path],
    sim_rows: List[Tuple[str, Path, List[Path]]],
    figsize=(20, 26),
    dpi=200,
    save_name="overview_figure.png",
):
    """
    Build a composite figure with subpanels:
      a. Processes
      b. Types
      c. Selected SIMULATIONS (per row: bigraph viz, example outputs)

    Note: Schema panel removed (only matplotlib outputs shown).
    """
    _ensure_outdir(outdir)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Compact a+b, let c take most space
    pr_rows, pr_cols = _grid_shape(len(process_pngs), max_cols=4)
    ty_rows, ty_cols = _grid_shape(len(type_pngs), max_cols=6)
    n_sims = len(sim_rows)

    # Height ratios: give 70% to sims, split remaining for a/b
    a_h = max(1, pr_rows)  # ensure positive
    b_h = max(1, ty_rows)  # ensure positive
    c_h = max(6, 3 * max(1, n_sims))  # ensure nonzero even if no sims
    total = a_h + b_h + c_h
    heights = [a_h/total, b_h/total, c_h/total]

    gs_root = GridSpec(3, 1, height_ratios=heights, hspace=0.18, figure=fig)

    # --- Panel a: Processes ---
    if len(process_pngs) > 0:
        gs_a = GridSpecFromSubplotSpec(pr_rows, pr_cols, subplot_spec=gs_root[0], wspace=0.05, hspace=0.12)
        proc_axes = [fig.add_subplot(gs_a[i // pr_cols, i % pr_cols]) for i in range(pr_rows * pr_cols)]
        proc_images = [im for p in process_pngs for im in [_load_rgba(p)] if im is not None]
        _panel_from_images(
            proc_axes,
            proc_images,
            titles=[p.stem.replace("_", " ") for p in process_pngs if (outdir / p.name).exists()],
        )
    else:
        ax = fig.add_subplot(gs_root[0])
        ax.axis("off")
        ax.text(0.5, 0.5, "No process figures found", ha="center", va="center", fontsize=11)

    pos_a = gs_root[0].get_position(fig)
    fig.text(pos_a.x0, pos_a.y1 + 0.008, "a. Processes", fontsize=14, weight="bold", va="bottom", ha="left")

    # --- Panel b: Types ---
    if len(type_pngs) > 0:
        gs_b = GridSpecFromSubplotSpec(ty_rows, ty_cols, subplot_spec=gs_root[1], wspace=0.04, hspace=0.10)
        type_axes = [fig.add_subplot(gs_b[i // ty_cols, i % ty_cols]) for i in range(ty_rows * ty_cols)]
        type_images = [im for p in type_pngs for im in [_load_rgba(p)] if im is not None]
        _panel_from_images(
            type_axes,
            type_images,
            titles=[p.stem.replace("_", " ") for p in type_pngs if (outdir / p.name).exists()],
        )
    else:
        ax = fig.add_subplot(gs_root[1])
        ax.axis("off")
        ax.text(0.5, 0.5, "No type figures found", ha="center", va="center", fontsize=11)

    pos_b = gs_root[1].get_position(fig)
    fig.text(pos_b.x0, pos_b.y1 + 0.008, "b. Types", fontsize=14, weight="bold", va="bottom", ha="left")

    # --- Panel c: SIMULATIONS (bigraph + outputs only) ---
    if n_sims > 0:
        # 2 columns per row: [bigraph viz | outputs (stack up to 2)]
        gs_c = GridSpecFromSubplotSpec(n_sims, 2, subplot_spec=gs_root[2], wspace=0.08, hspace=0.25)

        for r, (sim_name, viz_png, output_pngs) in enumerate(sim_rows):
            # col 0: bigraph viz
            ax0 = fig.add_subplot(gs_c[r, 0]); ax0.axis("off")
            viz_img = _load_rgba(viz_png)
            if viz_img is not None:
                ax0.imshow(viz_img)
            else:
                ax0.text(0.5, 0.5, "Viz not found", ha="center", va="center")
            ax0.set_title(f"{sim_name.replace('_', ' ')} bigraph", fontsize=11)

            # col 1: preferred outputs (stack up to 2)
            ax1 = fig.add_subplot(gs_c[r, 1]); ax1.axis("off")

            shown = []
            for p in (output_pngs or []):
                if p.suffix.lower() == ".gif":
                    im = _first_frame(p)
                else:
                    im = _load_rgba(p)
                if im is not None:
                    shown.append(im)
                if len(shown) == 2:
                    break

            if len(shown) == 0:
                ax1.text(0.5, 0.5, "No outputs found", ha="center", va="center")
            elif len(shown) == 1:
                ax1.imshow(shown[0])
            else:
                # vertical stack
                h = shown[0].height + shown[1].height
                w = max(shown[0].width, shown[1].width)
                stacked = Image.new("RGBA", (w, h), (255, 255, 255, 0))
                y = 0
                for im in shown:
                    if im.width < w:
                        im = ImageOps.pad(im, (w, im.height), color=(255, 255, 255, 0))
                    stacked.paste(im, (0, y))
                    y += im.height
                ax1.imshow(stacked)
            ax1.set_title(f"{sim_name.replace('_', ' ')} outputs", fontsize=11)

        pos_c = gs_root[2].get_position(fig)
        fig.text(pos_c.x0, pos_c.y1 + 0.008, "c. Simulation outputs", fontsize=14, weight="bold",
                 va="bottom", ha="left")
    else:
        ax = fig.add_subplot(gs_root[2])
        ax.axis("off")
        ax.text(0.5, 0.5, "No simulation outputs found", ha="center", va="center", fontsize=11)
        pos_c = gs_root[2].get_position(fig)
        fig.text(pos_c.x0, pos_c.y1 + 0.008, "c. Simulation outputs", fontsize=14, weight="bold",
                 va="bottom", ha="left")

    out_path = outdir / save_name
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, metadata={"Creator": "spatio_flux overview_fig.py"})
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Overview saved to: {out_path}")
    print(f"Overview (PDF) saved to: {out_path.with_suffix('.pdf')}")

# ---------- CLI + Main ----------

DEFAULT_TESTS = [
    "ecoli_core_dfba",
    "spatial_many_dfba",
    "diffusion_process",
    "comets",
    "particles",
    "particle_comets",
    "particle_dfba_fields",
    "particle_dfba_comets"
]

def parse_args():
    p = argparse.ArgumentParser(description="Build spatio-flux overview panels")
    p.add_argument(
        "--section",
        choices=["processes", "types", "simulate", "assemble", "all"],
        default="assemble",  # TODO -- change to "all" once everything is fast
        help="Which part(s) to run (default: assemble only)"
    )
    p.add_argument("--output", default="out", help="Output directory")
    p.add_argument(
        "--tests", nargs="*", default=None,
        help="Subset of SIMULATIONS to run (names from test_suite.SIMULATIONS)"
    )
    p.add_argument(
        "--clean", action="store_true",
        help="If set, clears the output directory before running"
    )
    p.add_argument("--dpi", type=int, default=200, help="DPI for the overview figure")
    p.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(20, 26),
                   help="Figure size in inches (width height)")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.output)
    if args.clean or args.section in ("all",):
        # Only wipe when requested or doing a full rebuild
        prepare_output_dir(outdir)

    _ensure_outdir(outdir)

    # establish core type system
    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    # Collect outputs incrementally (so assemble can be run standalone)
    process_pngs: List[Path] = sorted(outdir.glob("*_process.png"))
    type_pngs: List[Path] = sorted(outdir.glob("*_type.png"))
    sim_rows_input: List[Tuple[str, Path, List[Path]]] = []

    # Section: processes
    if args.section in ("processes", "all"):
        process_pngs = build_process_figs(core, outdir)

    # Section: types
    if args.section in ("types", "all"):
        type_pngs = build_type_figs(core, outdir)

    # Section: simulate
    if args.section in ("simulate", "all"):
        tests_to_run = args.tests if args.tests else DEFAULT_TESTS
        sim_rows_input = run_selected_sims(core, outdir, tests_to_run)

    # If we're only assembling, harvest whatever already exists on disk
    if args.section == "assemble":
        # processes/types already read from disk earlier
        # For sims, reconstruct pairs (sim_name, viz, outputs) based on existing files
        sim_rows_input = []
        # Heuristic: find all *_viz.png and pair with best outputs using name or plot_config filename
        for viz in sorted(outdir.glob("*_viz.png")):
            sim_name = viz.name[:-8]  # strip "_viz.png"
            # guess best base prefixes to search (sim_name + any known plot filenames)
            base_prefixes = {sim_name}
            if sim_name in SIMULATIONS:
                plot_cfg = SIMULATIONS[sim_name].get("plot_config", {})
                if "filename" in plot_cfg and plot_cfg["filename"]:
                    base_prefixes.add(plot_cfg["filename"])
            # collect candidates from any of those prefixes
            outs: List[Path] = []
            for bp in base_prefixes:
                outs.extend(_preferred_output_images(bp, outdir))
            # dedupe while preserving order
            dedupe: List[Path] = []
            seen = set()
            for p in outs:
                if p not in seen:
                    dedupe.append(p)
                    seen.add(p)
            sim_rows_input.append((sim_name, viz, dedupe))

    # Section: assemble (or all)
    if args.section in ("assemble", "all"):
        assemble_overview_figure(
            outdir,
            sorted(process_pngs),
            sorted(type_pngs),
            sim_rows=sim_rows_input,
            figsize=tuple(args.figsize),  # tall page to emphasize simulations
            dpi=args.dpi,
            save_name="overview_figure.png",
        )

if __name__ == "__main__":
    main()
