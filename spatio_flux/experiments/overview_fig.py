# spatio_flux/experiments/overview_fig.py

import math
import time
import argparse
from pathlib import Path
from typing import List, Tuple
import textwrap
from PIL import Image, ImageOps, UnidentifiedImageError

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from process_bigraph import register_types as register_process_types
from vivarium.vivarium import VivariumTypes
from bigraph_viz import plot_bigraph

# spatio-flux core types & helpers
from spatio_flux import register_types, TYPES_DICT
from spatio_flux.library.helpers import run_composite_document, prepare_output_dir
from spatio_flux.experiments.test_suite import SIMULATIONS, DEFAULT_RUNTIME_LONG
from spatio_flux.processes import PROCESS_DOCS


# ---------- Utilities ----------

def _estimate_wrap(schema_str: str, chars_per_inch: float, col_inches: float, max_chars: int = 4000) -> str:
    """
    Heuristically wrap schema text to fit the schema column width.
    Monospace at small sizes is ~8–10 chars/inch; we pass chars_per_inch explicitly.
    """
    w = max(20, int(chars_per_inch * col_inches))
    s = schema_str.strip()
    if len(s) > max_chars:
        s = s[:max_chars] + "\n[...]"
    return "\n".join(textwrap.fill(line, width=w, break_long_words=False, replace_whitespace=False)
                     for line in s.splitlines())

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
    snap = outdir / f"{base_prefix}_snapshots.png"
    if snap.exists():
        return [snap]

    ts = outdir / f"{base_prefix}_timeseries.png"
    if ts.exists():
        return [ts]

    base_png = outdir / f"{base_prefix}.png"
    if base_png.exists():
        return [base_png]

    for p in sorted(outdir.glob(f"{base_prefix}_*.png")):
        if any(s in p.name for s in ("_viz", "_schema")):
            continue
        if p.name.endswith("_snapshots.png") or p.name.endswith("_timeseries.png"):
            continue
        return [p]

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

        viz_png = outdir / f"{name}_viz.png"  # produced by run_composite_document
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
      c. Composite simulations (per row: name | schema | bigraph | outputs)
    """
    _ensure_outdir(outdir)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Make a/b a bit bigger so embedded text is readable
    # (You can tweak these three weights to taste.)
    panel_weights = (0.22, 0.22, 0.56)  # (a, b, c)
    gs_root = GridSpec(3, 1, height_ratios=panel_weights, hspace=0.06, figure=fig)

    # --- Panel a: Processes (tighter grid spacing, bigger tiles) ---
    pr_rows, pr_cols = _grid_shape(len(process_pngs), max_cols=4)
    if len(process_pngs) > 0 and pr_rows > 0:
        gs_a = GridSpecFromSubplotSpec(pr_rows, pr_cols, subplot_spec=gs_root[0], wspace=0.015, hspace=0.04)
        proc_axes = [fig.add_subplot(gs_a[i // pr_cols, i % pr_cols]) for i in range(pr_rows * pr_cols)]
        proc_images = [im for p in process_pngs for im in [_load_rgba(p)] if im is not None]
        for i, ax in enumerate(proc_axes):
            ax.axis("off")
            if i < len(proc_images):
                ax.imshow(proc_images[i])
    else:
        ax = fig.add_subplot(gs_root[0]); ax.axis("off")
        ax.text(0.5, 0.5, "No process figures found", ha="center", va="center", fontsize=11)

    pos_a = gs_root[0].get_position(fig)
    fig.text(pos_a.x0, pos_a.y1 + 0.005, "a. Processes", fontsize=14, weight="bold", va="bottom", ha="left")

    # --- Panel b: Types (tighter grid spacing, bigger tiles) ---
    ty_rows, ty_cols = _grid_shape(len(type_pngs), max_cols=6)
    if len(type_pngs) > 0 and ty_rows > 0:
        gs_b = GridSpecFromSubplotSpec(ty_rows, ty_cols, subplot_spec=gs_root[1], wspace=0.015, hspace=0.04)
        type_axes = [fig.add_subplot(gs_b[i // ty_cols, i % ty_cols]) for i in range(ty_rows * ty_cols)]
        type_images = [im for p in type_pngs for im in [_load_rgba(p)] if im is not None]
        for i, ax in enumerate(type_axes):
            ax.axis("off")
            if i < len(type_images):
                ax.imshow(type_images[i])
    else:
        ax = fig.add_subplot(gs_root[1]); ax.axis("off")
        ax.text(0.5, 0.5, "No type figures found", ha="center", va="center", fontsize=11)

    pos_b = gs_root[1].get_position(fig)
    fig.text(pos_b.x0, pos_b.y1 + 0.005, "b. Types", fontsize=14, weight="bold", va="bottom", ha="left")

    # --- Panel c: Composite simulations (name | schema | bigraph | outputs) ---
    fig_label = "c. Composite simulations"
    n_sims = len(sim_rows)
    if n_sims > 0:
        # Column widths (name | schema | viz | outputs)
        # Make the first column narrower; give schema extra width for wrapped text.
        width_ratios = [0.8, 2.2, 3.0, 2.5]
        gs_c = GridSpecFromSubplotSpec(
            n_sims, 4, subplot_spec=gs_root[2],
            wspace=0.06, hspace=0.12,
            width_ratios=width_ratios,
        )

        # Precompute schema wrapping width from figure size and ratios
        fig_width_in = figsize[0]
        col1_inches = fig_width_in * (width_ratios[1] / sum(width_ratios))
        # Monospace at fontsize=6 ≈ 9–10 chars/inch; start with 9
        chars_per_inch = 9.0

        for r, (sim_name, viz_png, output_pngs) in enumerate(sim_rows):
            # col 0: composite name (narrow)
            ax_name = fig.add_subplot(gs_c[r, 0]); ax_name.axis("off")
            ax_name.text(0.0, 0.5, sim_name.replace("_", " "), ha="left", va="center", fontsize=11)

            # col 1: schema text representation (wrapped to column width)
            ax_schema = fig.add_subplot(gs_c[r, 1]); ax_schema.axis("off")
            schema_txt = outdir / f"{sim_name}_schema.txt"
            if not schema_txt.exists():
                legacy = outdir / f"{sim_name}_schema_representation.txt"
                if legacy.exists():
                    schema_txt = legacy

            if schema_txt.exists():
                try:
                    with open(schema_txt, "r", encoding="utf-8", errors="replace") as f:
                        schema_str = f.read().strip()
                    # Hard cutoff – no wrapping; avoids overflow
                    max_chars = 800
                    if len(schema_str) > max_chars:
                        schema_str = schema_str[:max_chars] + "\n[...]"

                    ax_schema.text(
                        0, 1, schema_str,
                        fontsize=6,
                        va="top", ha="left",
                        family="monospace",
                        clip_on=True,
                    )
                    ax_schema.set_xlim(0, 1)
                    ax_schema.set_ylim(0, 1)
                except Exception as e:
                    ax_schema.text(0.5, 0.5, f"⚠ Error reading schema: {e}", ha="center", va="center", fontsize=8)

            else:
                ax_schema.text(0.5, 0.5, "Schema not found", ha="center", va="center", fontsize=8)

            # col 2: bigraph viz
            ax_viz = fig.add_subplot(gs_c[r, 2]); ax_viz.axis("off")
            viz_img = _load_rgba(viz_png)
            if viz_img is not None:
                ax_viz.imshow(viz_img)
            else:
                ax_viz.text(0.5, 0.5, "Viz not found", ha="center", va="center")

            # col 3: preferred outputs (stack up to 2 vertically)
            ax_out = fig.add_subplot(gs_c[r, 3]); ax_out.axis("off")
            shown = []
            for p in (output_pngs or []):
                im = _first_frame(p) if p.suffix.lower() == ".gif" else _load_rgba(p)
                if im is not None:
                    shown.append(im)
                if len(shown) == 2:
                    break
            if len(shown) == 0:
                ax_out.text(0.5, 0.5, "No outputs found", ha="center", va="center")
            elif len(shown) == 1:
                ax_out.imshow(shown[0])
            else:
                h = shown[0].height + shown[1].height
                w = max(shown[0].width, shown[1].width)
                stacked = Image.new("RGBA", (w, h), (255, 255, 255, 0))
                y = 0
                for im in shown:
                    if im.width < w:
                        im = ImageOps.pad(im, (w, im.height), color=(255, 255, 255, 0))
                    stacked.paste(im, (0, y)); y += im.height
                ax_out.imshow(stacked)

        pos_c = gs_root[2].get_position(fig)
        fig.text(pos_c.x0, pos_c.y1 + 0.005, fig_label, fontsize=14, weight="bold", va="bottom", ha="left")
    else:
        ax = fig.add_subplot(gs_root[2]); ax.axis("off")
        ax.text(0.5, 0.5, "No simulation outputs found", ha="center", va="center", fontsize=11)
        pos_c = gs_root[2].get_position(fig)
        fig.text(pos_c.x0, pos_c.y1 + 0.005, fig_label, fontsize=14, weight="bold", va="bottom", ha="left")

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
        default="assemble",
        help="Which part(s) to run (default: all)"
    )
    p.add_argument("--output", default="out", help="Output directory")
    p.add_argument("--tests", nargs="*", default=None,
                   help="Subset of SIMULATIONS to run (names from test_suite.SIMULATIONS)")
    p.add_argument("--clean", action="store_true", help="If set, clears the output directory before running")
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
        sim_rows_input = []
        for viz in sorted(outdir.glob("*_viz.png")):
            sim_name = viz.name[:-8]  # strip "_viz.png"
            base_prefixes = {sim_name}
            if sim_name in SIMULATIONS:
                plot_cfg = SIMULATIONS[sim_name].get("plot_config", {})
                if plot_cfg.get("filename"):
                    base_prefixes.add(plot_cfg["filename"])
            outs: List[Path] = []
            for bp in base_prefixes:
                outs.extend(_preferred_output_images(bp, outdir))
            dedupe: List[Path] = []
            seen = set()
            for p in outs:
                if p not in seen:
                    dedupe.append(p); seen.add(p)
            sim_rows_input.append((sim_name, viz, dedupe))

    if args.section in ("assemble", "all"):
        assemble_overview_figure(
            outdir,
            sorted(process_pngs),
            sorted(type_pngs),
            sim_rows=sim_rows_input,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            save_name="overview_figure.png",
        )

if __name__ == "__main__":
    main()
