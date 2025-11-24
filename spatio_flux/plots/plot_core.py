# spatio_flux/experiments/overview_fig.py

from typing import List
from pathlib import Path
from typing import Optional, Sequence
from PIL import Image, ImageDraw, ImageFont

from bigraph_viz import plot_bigraph

from spatio_flux import build_core
from spatio_flux.processes import PROCESS_DOCS
from spatio_flux.plots.colors import build_plot_settings


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ensure_outdir(outdir: Path) -> None:
    """Create the output directory if it does not exist."""
    outdir.mkdir(parents=True, exist_ok=True)


def _force_white_background(im: Image.Image) -> Image.Image:
    """
    Convert transparent/partially transparent pixels to pure white.
    Eliminates gray halos from alpha edges.
    """
    if im.mode != "RGBA":
        im = im.convert("RGBA")

    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.paste(im, mask=im.split()[3])  # paste using alpha channel as mask
    return bg.convert("RGB")  # flatten to opaque RGB

# ---------------------------------------------------------------------
# Public API: generate individual figs
# ---------------------------------------------------------------------


def plot_all_processes(
    outdir: str | Path = "out",
    core: Optional[object] = None,
) -> List[Path]:
    """
    Generate bigraph plots for all registered processes (PROCESS_DOCS).

    Parameters
    ----------
    outdir : str or Path
        Directory where the PNGs will be written.
    core : object or None
        Existing type core. If None, a new Vivarium core is created and the
        process + spatio-flux types are registered.

    Returns
    -------
    List[Path]
        List of paths to the generated PNG files.
    """
    outdir = Path(outdir)
    _ensure_outdir(outdir)

    if core is None:
        core = build_core()

    generated: List[Path] = []

    plot_settings = build_plot_settings(particle_ids=['particle', 'complex_particle'])
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
    # 'substrate': {
    #     '_type': 'concentration',
    #     '_value': 42.0,
    # },
    'substrate': {
        '_type': 'conc_counts_volume',
        'volume': 1,
        'counts': 1,
        'concentration': 1,
        # '_apply': apply_conc_counts_volume,
    },
    'fields': {
        '_type': 'fields',
        'substrate_id': [[0.1, 0.1, 0.1, 0.1]],
    },
    'particle': {
        '_type': 'particle',
        # 'id': 'particle_0',
        'position': [1.0, 1.0],
        'mass': 1.0,
    },
    'complex_particle': {
        '_type': 'complex_particle',
        # 'id': 'complex_particle_0',
        'position': [1.0, 1.0],
        'mass': 2.0,
        'velocity': [0.5, -0.5],
    },
}


def plot_all_types(
    outdir: str | Path = "out",
    core: Optional[object] = None,
) -> List[Path]:
    """
    Generate bigraph plots for all registered spatio-flux types (SPATIO_FLUX_TYPES).

    Parameters
    ----------
    outdir : str or Path
        Directory where the PNGs will be written.
    core : object or None
        Existing type core. If None, a new Vivarium core is created and the
        process + spatio-flux types are registered.

    Returns
    -------
    List[Path]
        List of paths to the generated PNG files.
    """
    outdir = Path(outdir)
    _ensure_outdir(outdir)

    if core is None:
        core = build_core()

    generated: List[Path] = []

    plot_settings = build_plot_settings(
        particle_ids=['particle', 'complex_particle'],
        conc_type_species=['conc_counts_volume', 'substrate'],
    )
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
    outdir: str | Path,
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
      - Tight spacing by using per-image widths instead of fixed cell width
      - Flattens all images onto a pure white background (no gray seams)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_paths = list(image_paths)
    if not image_paths:
        print("⚠ No images found to assemble.")
        return outdir / save_name

    def _flatten_to_white(im: Image.Image) -> Image.Image:
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.paste(im, (0, 0), im)
        return bg.convert("RGB")

    # ---- Load and resize images to common height ----
    images: list[Image.Image] = []
    for p in image_paths:
        im = Image.open(p)
        im = _flatten_to_white(im)

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

    cell_h = target_height

    # ---- First pass: compute row widths ----
    row_widths: list[int] = []
    idx = 0
    for r in range(n_rows):
        row_imgs = images[idx: idx + n_cols]
        if not row_imgs:
            break
        row_width = sum(im.width for im in row_imgs)
        if len(row_imgs) > 1:
            row_width += col_gap_px * (len(row_imgs) - 1)
        row_widths.append(row_width)
        idx += len(row_imgs)

    max_row_width = max(row_widths) if row_widths else 0

    title_height = int(target_height * 0.4) if title else 0
    grid_height = n_rows * cell_h + row_gap_px * (n_rows - 1)
    total_width = margin_px * 2 + max_row_width
    total_height = margin_px * 2 + title_height + grid_height

    # ---- Create canvas ----
    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
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

        draw.text((title_x, title_y), title, font=font, fill=(0, 0, 0))

    # ---- Paste images row by row ----
    start_y = margin_px + title_height
    idx = 0
    for r in range(n_rows):
        row_imgs = images[idx: idx + n_cols]
        if not row_imgs:
            break

        row_width = row_widths[r]
        # Center row horizontally within max_row_width
        row_x = margin_px + (max_row_width - row_width) // 2

        y = start_y + r * (cell_h + row_gap_px)
        x = row_x
        for im in row_imgs:
            paste_x = int(x)
            paste_y = int(y + (cell_h - im.height) // 2)
            canvas.paste(im, (paste_x, paste_y))
            x += im.width + col_gap_px

        idx += len(row_imgs)

    out_path = outdir / save_name
    canvas.save(out_path)
    print(f"Saved image grid to: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# High-level helpers: generate + assemble
# ---------------------------------------------------------------------

def assemble_process_figures(
    core,
    outdir = "out",
    *,
    target_height: int = 350,
    n_cols: Optional[int] = None,
    n_rows: Optional[int] = None,
    col_gap_px: int = 40,
    row_gap_px: int = 60,
    save_name: str = "process_overview.png",
) -> Path:
    """
    Generate all *_process.png bigraph images and assemble them
    into a single overview figure.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating process figures...")
    proc_paths = plot_all_processes(outdir, core=core)
    print(f"Generated {len(proc_paths)} process figs in {outdir}")

    return assemble_image_grid(
        outdir,
        proc_paths,
        # title="Processes",
        target_height=target_height,
        n_cols=n_cols,
        n_rows=n_rows,
        col_gap_px=col_gap_px,
        row_gap_px=row_gap_px,
        save_name=save_name,
    )


def assemble_type_figures(
    core,
    outdir="out",
    *,
    target_height: int = 1200,
    n_cols: Optional[int] = None,
    n_rows: Optional[int] = None,
    col_gap_px: int = 40,
    row_gap_px: int = 60,
    save_name: str = "type_overview.png",
) -> Path:
    """
    Generate all *_type.png bigraph images and assemble them
    into a single overview figure.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating type figures...")
    type_paths = plot_all_types(outdir, core=core)
    print(f"Generated {len(type_paths)} type figs in {outdir}")

    return assemble_image_grid(
        outdir,
        type_paths,
        target_height=target_height,
        n_cols=n_cols,
        n_rows=n_rows,
        col_gap_px=col_gap_px,
        row_gap_px=row_gap_px,
        margin_px=0,              # really tight outer margin
        save_name=save_name,
    )


# Optional CLI for manual testing
if __name__ == "__main__":
    out = Path("out")
    core = build_core()

    assemble_process_figures(core, outdir=out, n_rows=2)
    assemble_type_figures(core, outdir=out, n_rows=1)
