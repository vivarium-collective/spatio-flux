from process_bigraph import register_types as register_process_types
from vivarium.vivarium import VivariumTypes
from bigraph_viz import plot_bigraph
from spatio_flux import register_types, TYPES_DICT

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
            "config": {
                "n_bins": (5, 10)
            },
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


# ---- Figure assembly helpers ----------------------------------------

def _ensure_outdir(outdir: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)


def _load_png(path: Path):
    img = Image.open(path).convert("RGBA")
    return img


def _grid_shape(n_items: int, max_cols: int = 3):
    """Nice rectangular-ish grid for n items."""
    cols = min(max_cols, max(1, n_items))
    rows = math.ceil(n_items / cols)
    return rows, cols


def _panel_from_images(ax_array, images, titles=None):
    """Draw PIL images into provided axes array (flattened)."""
    for i, ax in enumerate(ax_array):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i])
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9, pad=3)


def assemble_overview_figure(outdir: str,
                             process_pngs: list[Path],
                             type_pngs: list[Path],
                             max_cols_process: int = 3,
                             max_cols_types: int = 4,
                             figsize=(14, 10),
                             save_name="overview_figure.png"):
    """
    Build a composite figure with two subpanels:
      a. all processes
      b. all types
    """
    # Load images
    proc_imgs = [_load_png(p) for p in process_pngs]
    type_imgs = [_load_png(p) for p in type_pngs]

    # Grids
    pr_rows, pr_cols = _grid_shape(len(proc_imgs), max_cols=max_cols_process)
    ty_rows, ty_cols = _grid_shape(len(type_imgs), max_cols=max_cols_types)

    # Create figure layout: two vertical panels
    fig = plt.figure(figsize=figsize, dpi=200)

    # Heights proportional to rows (give a little extra to processes)
    total_rows = pr_rows + ty_rows if (pr_rows + ty_rows) > 0 else 1
    pr_h = pr_rows / total_rows + 0.05
    ty_h = ty_rows / total_rows

    gs_root = GridSpec(2, 1, height_ratios=[max(pr_h, 0.05), max(ty_h, 0.05)], hspace=0.22, figure=fig)

    # --- Panel a: processes -------------------------------------------------
    # Subdivide top panel into a grid for process images
    gs_a = GridSpecFromSubplotSpec(pr_rows, pr_cols, subplot_spec=gs_root[0], wspace=0.06, hspace=0.20)
    proc_axes = [fig.add_subplot(gs_a[i // pr_cols, i % pr_cols]) for i in range(pr_rows * pr_cols)]
    proc_titles = [p.stem for p in process_pngs]
    _panel_from_images(proc_axes, proc_imgs, titles=proc_titles)

    # Title for panel a
    pos_a = gs_root[0].get_position(fig)
    fig.text(pos_a.x0, pos_a.y1 + 0.01, "a. Processes", fontsize=14, weight="bold", va="bottom", ha="left")

    # --- Panel b: types -----------------------------------------------------
    gs_b = GridSpecFromSubplotSpec(ty_rows, ty_cols, subplot_spec=gs_root[1], wspace=0.06, hspace=0.20)
    type_axes = [fig.add_subplot(gs_b[i // ty_cols, i % ty_cols]) for i in range(ty_rows * ty_cols)]
    type_titles = [p.stem for p in type_pngs]
    _panel_from_images(type_axes, type_imgs, titles=type_titles)

    # Title for panel b
    pos_b = gs_root[1].get_position(fig)
    fig.text(pos_b.x0, pos_b.y1 + 0.01, "b. Types", fontsize=14, weight="bold", va="bottom", ha="left")

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

    # 1) Plot the processes (single-node per plot)
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

    # 2) Plot the types (one type per plot)
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

    # 3) Assemble comprehensive overview figure
    assemble_overview_figure(outdir, process_pngs, type_pngs,
                             max_cols_process=3, max_cols_types=4,
                             figsize=(16, 12),
                             save_name="overview_figure.png")


if __name__ == '__main__':
    main()
