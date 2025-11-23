# spatio_flux/experiments/overview_fig.py

from typing import List, Optional, Sequence
import math
from PIL import Image, UnidentifiedImageError

from pathlib import Path
from typing import Optional, Sequence
from PIL import Image, ImageDraw, ImageFont
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
            collapse_redundant_processes=False,
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


SPATIO_FLUX_TYPE_EXAMPLES = {
    'simple_particle': {
        '_type': 'simple_particle',
        'id': 'simple_particle',
        'position': [0.0, 0.0],
        'mass': 1.0,
    },
    'complex_particle': {
        '_type': 'particle',
        'id': 'complex_particle',
        'position': [1.0, 1.0],
        'mass': 2.0,
        'velocity': [0.5, -0.5],
    },
    'fields': {
        '_type': 'fields',
        'mol_id': [[0.1, 0.2], [0.3, 0.4]],
    }
}


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
        particle_ids=['simple_particle', 'complex_particle']
    )
    plot_settings.update(
        dict(
            dpi="300",
            # show_values=True,
            show_types=True,
            collapse_redundant_processes=False,
            value_char_limit=20,
            type_char_limit=40,
        )
    )

    for type_name in sorted(SPATIO_FLUX_TYPE_EXAMPLES.keys()):
        type_state = SPATIO_FLUX_TYPE_EXAMPLES[type_name]
        # type_state = core.default(type_schema)
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

def assemble_image_grid(
    output: str | Path,
    image_paths: Sequence[Path],
    *,
    title: str = "",
    target_height: int = 300,        # height of each panel
    n_cols: Optional[int] = None,    # force N columns if set
    n_rows: Optional[int] = None,    # force N rows if set
    col_gap_px: int = 40,
    row_gap_px: int = 60,
    margin_px: int = 40,
    save_name: str = "overview.png",
) -> Path:
    """
    Universal image grid assembler:
      - Preserves aspect ratios
      - Resizes to target_height
      - Supports explicit rows/columns
      - Tight pixel-level spacing (no subplot whitespace)

    If n_cols and n_rows are both None, creates a single-row horizontal strip.
    """
    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)

    image_paths = list(image_paths)
    if not image_paths:
        print("⚠ No images found to assemble.")
        return outdir / save_name

    # ---- Load and resize images to common height ----
    images = []
    for p in image_paths:
        im = Image.open(p).convert("RGBA")
        w, h = im.size
        if h == 0:
            continue
        new_w = int(round(w * target_height / h))
        images.append(im.resize((new_w, target_height), Image.LANCZOS))

    if not images:
        print("⚠ Failed to read any images.")
        return outdir / save_name

    n = len(images)

    # ---- Determine rows & columns ----
    if n_cols is not None:
        n_cols = max(1, min(n_cols, n))
        n_rows_eff = (n + n_cols - 1) // n_cols
    elif n_rows is not None:
        n_rows_eff = max(1, min(n_rows, n))
        n_cols = (n + n_rows_eff - 1) // n_rows_eff
    else:
        # default: single horizontal strip
        n_rows_eff = 1
        n_cols = n

    n_rows = n_rows_eff

    # ---- Cell geometry (each cell width = max image width) ----
    cell_w = max(im.width for im in images)
    cell_h = target_height

    total_width = (
        margin_px * 2 +
        n_cols * cell_w +
        col_gap_px * (n_cols - 1)
    )

    title_height = int(target_height * 0.4) if title else 0
    grid_height = n_rows * cell_h + row_gap_px * (n_rows - 1)
    total_height = margin_px * 2 + title_height + grid_height

    # ---- Create the canvas ----
    canvas = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # ---- Optional title ----
    if title:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(title_height * 0.6))
        except OSError:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        title_x = (total_width - text_w) // 2
        title_y = margin_px + (title_height - text_h) // 2

        draw.text((title_x, title_y), title, font=font, fill=(0, 0, 0, 255))

    # ---- Paste images in the grid ----
    start_y = margin_px + title_height
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if idx >= n:
                break
            im = images[idx]

            cell_x = margin_px + c * (cell_w + col_gap_px)
            cell_y = start_y + r * (cell_h + row_gap_px)

            paste_x = cell_x + (cell_w - im.width) // 2
            paste_y = cell_y + (cell_h - im.height) // 2

            canvas.paste(im, (paste_x, paste_y), im)
            idx += 1

    out_path = outdir / save_name
    canvas.save(out_path)
    print(f"Saved image grid to: {out_path}")
    return out_path


def assemble_process_figure(
    output="out",
    *,
    target_height=350,
    n_cols=None,
    n_rows=None,
    col_gap_px=40,
    row_gap_px=60,
    save_name="process_overview.png",
):
    image_paths = sorted(Path(output).glob("*_process.png"))
    return assemble_image_grid(
        output,
        image_paths,
        title="Processes",
        target_height=target_height,
        n_cols=n_cols,
        n_rows=n_rows,
        col_gap_px=col_gap_px,
        row_gap_px=row_gap_px,
        save_name=save_name,
    )

def assemble_type_figure(
    output="out",
    *,
    target_height=300,
    n_cols=None,
    n_rows=None,
    col_gap_px=0,
    row_gap_px=20,
    save_name="type_overview.png",
):
    image_paths = sorted(Path(output).glob("*_type.png"))
    return assemble_image_grid(
        output,
        image_paths,
        title="Types",
        target_height=target_height,
        n_cols=n_cols,
        n_rows=n_rows,
        col_gap_px=col_gap_px,
        row_gap_px=row_gap_px,
        save_name=save_name,
    )


# ---------------------------------------------------------------------
# Optional: tiny CLI for quick manual use
# ---------------------------------------------------------------------

if __name__ == "__main__":
    out = Path("out")
    core = _build_core()

    print("Generating process figures...")
    proc_paths = plot_all_processes(out, core=core)
    print(f"Generated {len(proc_paths)} process figs in {out}")
    assemble_process_figure(out, n_rows=2)

    print("Generating type figures...")
    type_paths = plot_all_types(out, core=core)
    print(f"Generated {len(type_paths)} type figs in {out}")
    assemble_type_figure(out)
