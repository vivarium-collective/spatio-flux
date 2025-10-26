# spatio_flux/experiments/overview_fig.py

from process_bigraph import register_types as register_process_types
from vivarium.vivarium import VivariumTypes
from bigraph_viz import plot_bigraph
from spatio_flux import register_types, TYPES_DICT

# import composite doc(s) from the test suite
from spatio_flux.experiments.test_suite import (
    get_comets_doc, get_particles_doc, get_particle_comets_doc, get_particle_dfba_doc,
    get_particle_dfba_comets_doc
)

import os
from pathlib import Path
from PIL import Image
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# ---- Single-process docs (one node per plot) ------------------------

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

# ---- Composites imported from test_suite ----------------------------

COMPOSITE_DOCS = {
    'comets': get_comets_doc,
    'particles': get_particles_doc,
    'particle_comets': get_particle_comets_doc,
    'particle_dfba': get_particle_dfba_doc,
    'particle_dfba_comets': get_particle_dfba_comets_doc,
}


# ---- Figure assembly helpers ----------------------------------------

def _ensure_outdir(outdir: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)

def _load_png(path: Path):
    return Image.open(path).convert("RGBA")

def _grid_shape(n_items: int, max_cols: int = 3):
    cols = min(max_cols, max(1, n_items))
    rows = math.ceil(n_items / cols)
    return rows, cols

def _panel_from_images(ax_array, images, titles=None):
    for i, ax in enumerate(ax_array):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i])
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9, pad=3)

def assemble_overview_figure(outdir: str,
                             process_pngs: list[Path],
                             type_pngs: list[Path],
                             composite_pngs: list[Path] = None,
                             max_cols_process: int = 3,
                             max_cols_types: int = 4,
                             max_cols_composites: int = 2,
                             figsize=(16, 12),
                             save_name="overview_figure.png"):
    """
    Build a composite figure with subpanels:
      a. Processes
      b. Types
      c. Composites (optional; appears if composite_pngs is non-empty)
    """
    composite_pngs = composite_pngs or []

    # Load images
    proc_imgs = [_load_png(p) for p in process_pngs]
    type_imgs = [_load_png(p) for p in type_pngs]
    comp_imgs = [_load_png(p) for p in composite_pngs]

    pr_rows, pr_cols = _grid_shape(len(proc_imgs), max_cols=max_cols_process)
    ty_rows, ty_cols = _grid_shape(len(type_imgs), max_cols=max_cols_types)
    cp_rows, cp_cols = (0, 0)
    if comp_imgs:
        cp_rows, cp_cols = _grid_shape(len(comp_imgs), max_cols=max_cols_composites)

    # Determine how many rows of panels we need (2 or 3)
    n_panels = 2 + (1 if comp_imgs else 0)
    fig = plt.figure(figsize=figsize, dpi=200)

    # Height ratios roughly proportional to the rows in each section
    heights = [max(pr_rows, 1), max(ty_rows, 1)]
    if comp_imgs:
        heights.append(max(cp_rows, 1))
    gs_root = GridSpec(n_panels, 1, height_ratios=heights, hspace=0.22, figure=fig)

    # Panel a: processes
    gs_a = GridSpecFromSubplotSpec(pr_rows, pr_cols, subplot_spec=gs_root[0], wspace=0.06, hspace=0.20)
    proc_axes = [fig.add_subplot(gs_a[i // pr_cols, i % pr_cols]) for i in range(pr_rows * pr_cols)]
    _panel_from_images(proc_axes, proc_imgs, titles=[p.stem for p in process_pngs])

    pos_a = gs_root[0].get_position(fig)
    fig.text(pos_a.x0, pos_a.y1 + 0.01, "a. Processes", fontsize=14, weight="bold",
             va="bottom", ha="left")

    # Panel b: types
    gs_b = GridSpecFromSubplotSpec(ty_rows, ty_cols, subplot_spec=gs_root[1], wspace=0.06, hspace=0.20)
    type_axes = [fig.add_subplot(gs_b[i // ty_cols, i % ty_cols]) for i in range(ty_rows * ty_cols)]
    _panel_from_images(type_axes, type_imgs, titles=[p.stem for p in type_pngs])

    pos_b = gs_root[1].get_position(fig)
    fig.text(pos_b.x0, pos_b.y1 + 0.01, "b. Types", fontsize=14, weight="bold",
             va="bottom", ha="left")

    # Panel c: composites (optional)
    if comp_imgs:
        gs_c = GridSpecFromSubplotSpec(cp_rows, cp_cols, subplot_spec=gs_root[2], wspace=0.06, hspace=0.20)
        comp_axes = [fig.add_subplot(gs_c[i // cp_cols, i % cp_cols]) for i in range(cp_rows * cp_cols)]
        _panel_from_images(comp_axes, comp_imgs, titles=[p.stem for p in composite_pngs])

        pos_c = gs_root[2].get_position(fig)
        fig.text(pos_c.x0, pos_c.y1 + 0.01, "c. Composites", fontsize=14, weight="bold",
                 va="bottom", ha="left")

    # Save (PNG + PDF)
    _ensure_outdir(outdir)
    out_path = Path(outdir) / save_name
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Overview saved to: {out_path}")
    print(f"Overview (PDF) saved to: {out_path.with_suffix('.pdf')}")

# ---- Main ------------------------------------------------------------

def main():
    outdir = 'out'
    _ensure_outdir(outdir)

    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    # 1) Plot single processes
    process_pngs = []
    for name, get_doc in PROCESS_DOCS.items():
        document = get_doc(core=core)
        fname = f'{name}_process'
        plot_bigraph(
            state=document,
            core=core,
            out_dir=outdir,
            filename=fname,
            dpi='300',
            collapse_redundant_processes=True
        )
        process_pngs.append(Path(outdir) / f"{fname}.png")

    # 2) Plot types (one per image)
    type_pngs = []
    for type_name, type_schema in TYPES_DICT.items():
        fname = f'{type_name}_type'
        plot_bigraph(
            state={type_name: type_schema},
            show_types=True,
            core=core,
            out_dir=outdir,
            filename=fname,
            dpi='300',
            collapse_redundant_processes=True
        )
        type_pngs.append(Path(outdir) / f"{fname}.png")

    # 3) Plot composites imported from test_suite
    composite_pngs = []
    for name, get_doc in COMPOSITE_DOCS.items():
        # test_suite composites expect a doc with mixed schema/state fragments
        document = get_doc(core=core, config={})
        fname = f'{name}_composite'
        plot_bigraph(
            state=document,          # visualize the composite spec
            core=core,
            out_dir=outdir,
            filename=fname,
            dpi='300',
            collapse_redundant_processes=True
        )
        composite_pngs.append(Path(outdir) / f"{fname}.png")

    # 4) Assemble master overview (now with c. Composites)
    assemble_overview_figure(
        outdir,
        process_pngs,
        type_pngs,
        composite_pngs=composite_pngs,       # shows panel c if non-empty
        max_cols_process=3,
        max_cols_types=4,
        max_cols_composites=2,
        figsize=(16, 12),
        save_name="overview_figure.png"
    )


if __name__ == '__main__':
    main()
