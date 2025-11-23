# spatio_flux/experiments/overview_fig.py

from pathlib import Path
from typing import List, Optional, Sequence

import math
from PIL import Image, UnidentifiedImageError

import matplotlib.pyplot as plt

from process_bigraph import register_types as register_process_types
from vivarium.vivarium import VivariumTypes
from bigraph_viz import plot_bigraph

from spatio_flux import register_types, SPATIO_FLUX_TYPES
from spatio_flux.processes import PROCESS_DOCS
from spatio_flux.library.colors import build_plot_settings


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ensure_outdir(outdir: Path) -> None:
    """Create the output directory if it does not exist."""
    outdir.mkdir(parents=True, exist_ok=True)


def _build_core():
    """Construct and return a Vivarium core with process + spatio-flux types registered."""
    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)
    return core


def _load_rgba(path: Path) -> Image.Image | None:
    """Load an image as RGBA or return None if missing/unreadable."""
    try:
        with Image.open(path) as im:
            return im.convert("RGBA")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        print(f"⚠ Skipping missing/unreadable image: {path} ({e.__class__.__name__})")
        return None


def _grid_shape(n_items: int, max_cols: int) -> tuple[int, int]:
    """Return (rows, cols) with cols in [1..max_cols] and rows >= 0 (may be 0 if n_items==0)."""
    if n_items <= 0:
        return 0, 0
    cols = min(max_cols, n_items)
    rows = math.ceil(n_items / cols)
    return rows, cols


# ---------------------------------------------------------------------
# Public API: generate individual figs
# ---------------------------------------------------------------------


def plot_all_processes(
    output: str | Path = "out",
    core: Optional[object] = None,
) -> List[Path]:
    """
    Generate bigraph plots for all registered processes (PROCESS_DOCS).

    Parameters
    ----------
    output : str or Path
        Directory where the PNGs will be written.
    core : object or None
        Existing type core. If None, a new Vivarium core is created and the
        process + spatio-flux types are registered.

    Returns
    -------
    List[Path]
        List of paths to the generated PNG files.
    """
    outdir = Path(output)
    _ensure_outdir(outdir)

    if core is None:
        core = _build_core()

    generated: List[Path] = []

    plot_settings = build_plot_settings(particle_ids=['simple_particle', 'complex_particle'])
    plot_settings.update(
        dict(
            dpi="300",
            show_values=True,
            show_types=True,
            collapse_redundant_processes={
                "exclude": [("particle_movement",), ("particle_division",)]
            },
            value_char_limit=20,
            type_char_limit=40,
        )
    )

    for name, get_doc in PROCESS_DOCS.items():
        doc = get_doc(core=core)
        fname = f"{name}_process"
        plot_bigraph(
            state=doc,
            core=core,
            out_dir=str(outdir),
            filename=fname,
            **plot_settings,
        )
        png = outdir / f"{fname}.png"
        if png.exists():
            generated.append(png)

    return generated


def plot_all_types(
    output: str | Path = "out",
    core: Optional[object] = None,
) -> List[Path]:
    """
    Generate bigraph plots for all registered spatio-flux types (SPATIO_FLUX_TYPES).

    Parameters
    ----------
    output : str or Path
        Directory where the PNGs will be written.
    core : object or None
        Existing type core. If None, a new Vivarium core is created and the
        process + spatio-flux types are registered.

    Returns
    -------
    List[Path]
        List of paths to the generated PNG files.
    """
    outdir = Path(output)
    _ensure_outdir(outdir)

    if core is None:
        core = _build_core()

    generated: List[Path] = []

    plot_settings = build_plot_settings(
        particle_ids=['simple_particle', 'particle']
    )
    plot_settings.update(
        dict(
            dpi="300",
            # show_values=True,
            show_types=True,
            collapse_redundant_processes={
                "exclude": [("particle_movement",), ("particle_division",)]
            },
            value_char_limit=20,
            type_char_limit=40,
        )
    )

    for type_name in sorted(SPATIO_FLUX_TYPES.keys()):
        type_schema = SPATIO_FLUX_TYPES[type_name]
        type_state = core.default(type_schema)
        fname = f"{type_name}_type"
        plot_bigraph(
            state={type_name: type_state},
            core=core,
            out_dir=str(outdir),
            filename=fname,
            **plot_settings
        )
        png = outdir / f"{fname}.png"
        if png.exists():
            generated.append(png)

    return generated


# ---------------------------------------------------------------------
# Public API: assemble overview figures
# ---------------------------------------------------------------------


def assemble_process_figure(
    output: str | Path = "out",
    image_paths: Optional[Sequence[Path]] = None,
    *,
    max_cols: int = 4,
    figsize: tuple[float, float] = (20, 10),
    dpi: int = 200,
    save_name: str = "process_overview.png",
) -> Path:
    """
    Assemble all process PNGs into a single overview figure.

    Parameters
    ----------
    output : str or Path
        Directory where the input PNGs live and where the overview figure is saved.
    image_paths : sequence[Path] or None
        Explicit list of process PNGs. If None, we search for '*_process.png' in `output`.
    max_cols : int
        Maximum number of columns in the grid.
    figsize : (float, float)
        Matplotlib figure size in inches.
    dpi : int
        Dots per inch for the saved figure.
    save_name : str
        Filename for the assembled figure (PNG).

    Returns
    -------
    Path
        Path to the saved overview figure.
    """
    outdir = Path(output)
    _ensure_outdir(outdir)

    if image_paths is None:
        image_paths = sorted(outdir.glob("*_process.png"))

    image_paths = list(image_paths)
    n = len(image_paths)

    if n == 0:
        print("⚠ No process PNGs found to assemble.")
        return outdir / save_name

    rows, cols = _grid_shape(n, max_cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    images = [_load_rgba(p) for p in image_paths]
    images = [im for im in images if im is not None]
    n_img = len(images)

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx < n_img:
            ax.imshow(images[idx])

    fig.suptitle("Processes", fontsize=16, weight="bold")
    out_path = outdir / save_name
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Process overview saved to: {out_path}")
    return out_path


def assemble_type_figure(
    output: str | Path = "out",
    image_paths: Optional[Sequence[Path]] = None,
    *,
    max_cols: int = 6,
    figsize: tuple[float, float] = (20, 10),
    dpi: int = 200,
    save_name: str = "type_overview.png",
) -> Path:
    """
    Assemble all type PNGs into a single overview figure.

    Parameters
    ----------
    output : str or Path
        Directory where the input PNGs live and where the overview figure is saved.
    image_paths : sequence[Path] or None
        Explicit list of type PNGs. If None, we search for '*_type.png' in `output`.
    max_cols : int
        Maximum number of columns in the grid.
    figsize : (float, float)
        Matplotlib figure size in inches.
    dpi : int
        Dots per inch for the saved figure.
    save_name : str
        Filename for the assembled figure (PNG).

    Returns
    -------
    Path
        Path to the saved overview figure.
    """
    outdir = Path(output)
    _ensure_outdir(outdir)

    if image_paths is None:
        image_paths = sorted(outdir.glob("*_type.png"))

    image_paths = list(image_paths)
    n = len(image_paths)

    if n == 0:
        print("⚠ No type PNGs found to assemble.")
        return outdir / save_name

    rows, cols = _grid_shape(n, max_cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    images = [_load_rgba(p) for p in image_paths]
    images = [im for im in images if im is not None]
    n_img = len(images)

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx < n_img:
            ax.imshow(images[idx])

    fig.suptitle("Types", fontsize=16, weight="bold")
    out_path = outdir / save_name
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Type overview saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# Optional: tiny CLI for quick manual use
# ---------------------------------------------------------------------

if __name__ == "__main__":
    out = Path("out")
    core = _build_core()

    print("Generating process figures...")
    proc_paths = plot_all_processes(out, core=core)
    print(f"Generated {len(proc_paths)} process figs in {out}")
    assemble_process_figure(out)

    print("Generating type figures...")
    type_paths = plot_all_types(out, core=core)
    print(f"Generated {len(type_paths)} type figs in {out}")
    assemble_type_figure(out)
