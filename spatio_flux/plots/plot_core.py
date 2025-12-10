# spatio_flux/experiments/overview_fig.py

from typing import List
from pathlib import Path
from typing import Optional, Sequence
from PIL import Image, ImageDraw, ImageFont

from bigraph_viz import plot_bigraph
from process_bigraph import allocate_core

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
        core = allocate_core()

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
    'fields': {
        'substrate_id': {
            '_type': 'count_concentration_volume',
            'volume': 1,
            'count': 1,
            'concentration': 1,
            # '_apply': apply_conc_count_volume,
        },
    },
    # 'fields': {
    #     '_type': 'fields',
    #     'substrate_id': [[0.1, 0.1, 0.1, 0.1]],
    # },
    # 'particles': {
    #     'particle': {
    #         '_type': 'particle',
    #         # 'id': 'particle_0',
    #         'position': [1.0, 1.0],
    #         'mass': 1.0,
    #     },
    # },
    'particles': {
        'particle_id': {
            '_type': 'complex_particle',
            # 'id': 'complex_particle_0',
            'position': [1.0, 1.0],
            'mass': 2.0,
            'velocity': [0.5, -0.5],
        },
    }
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
        core = allocate_core()

    generated: List[Path] = []

    plot_settings = build_plot_settings(
        particle_ids=['particle_id'],
        conc_type_species=['substrate_id'],
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
        target_height: int = 300,
        n_cols: Optional[int] = None,
        n_rows: Optional[int] = None,
        layout: Optional[Sequence[Sequence[Optional[int]]]] = None,
        col_gap_px: int = 40,
        row_gap_px: int = 60,
        margin_px: int = 40,
        save_name: str = "overview.png",
) -> Path:
    """
    Universal image grid assembler with optional explicit placement.

    Options:
      - Provide `layout` to explicitly control which image appears in which grid cell.
        Example:
            layout = [
                [0, 1, 2],
                [3, None, 4],
            ]
      - If layout=None, fall back to automatic n_rows / n_cols logic.

    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load all images ---
    flat_paths = list(image_paths)
    if not flat_paths:
        print("⚠ No images found to assemble.")
        return outdir / save_name

    def _flatten_to_white(im: Image.Image) -> Image.Image:
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.paste(im, (0, 0), im)
        return bg.convert("RGB")

    # Resize all images first to uniform height
    loaded_images = []
    for p in flat_paths:
        im = Image.open(p)
        im = _flatten_to_white(im)
        w, h = im.size
        new_w = int(round(w * target_height / h))
        loaded_images.append(im.resize((new_w, target_height), Image.LANCZOS))

    # ===============================================================
    # Manual layout if provided
    # ===============================================================
    if layout is not None:
        # Validate indices
        for row in layout:
            for cell in row:
                if cell is not None and not (0 <= cell < len(loaded_images)):
                    raise ValueError(f"Invalid layout index {cell}")

        n_rows_eff = len(layout)
        n_cols_eff = max(len(row) for row in layout)

        # Compute per-row widths
        row_widths = []
        for row in layout:
            row_imgs = [
                loaded_images[idx] for idx in row if idx is not None
            ]
            if row_imgs:
                w = sum(im.width for im in row_imgs)
                if len(row_imgs) > 1:
                    w += col_gap_px * (len(row_imgs) - 1)
            else:
                w = 0
            row_widths.append(w)

        max_row_width = max(row_widths)

    else:
        # ===========================================================
        # Old automatic layout logic
        # ===========================================================
        n = len(loaded_images)

        if n_cols is not None:
            n_cols_eff = max(1, min(n_cols, n))
            n_rows_eff = (n + n_cols_eff - 1) // n_cols_eff
        elif n_rows is not None:
            n_rows_eff = max(1, min(n_rows, n))
            n_cols_eff = (n + n_rows_eff - 1) // n_rows_eff
        else:
            n_rows_eff = 1
            n_cols_eff = n

        row_widths = []
        idx = 0
        for r in range(n_rows_eff):
            row_imgs = loaded_images[idx: idx + n_cols_eff]
            if row_imgs:
                w = sum(im.width for im in row_imgs)
                if len(row_imgs) > 1:
                    w += col_gap_px * (len(row_imgs) - 1)
            else:
                w = 0
            row_widths.append(w)
            idx += len(row_imgs)

        max_row_width = max(row_widths)

    # ===============================================================
    # Compute final canvas size
    # ===============================================================
    title_height = int(target_height * 0.4) if title else 0
    grid_height = n_rows_eff * target_height + row_gap_px * (n_rows_eff - 1)
    total_width = margin_px * 2 + max_row_width
    total_height = margin_px * 2 + title_height + grid_height

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Title (optional)
    if title:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(title_height * 0.6))
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            ((total_width - text_w) // 2, margin_px + (title_height - text_h) // 2),
            title,
            font=font,
            fill=(0, 0, 0),
        )

    # ===============================================================
    # Paste images using layout or automatic grid
    # ===============================================================
    start_y = margin_px + title_height

    if layout is not None:
        for r, row in enumerate(layout):
            y = start_y + r * (target_height + row_gap_px)
            # compute THIS row's left offset
            row_imgs = [idx for idx in row if idx is not None]
            row_content_width = (
                    sum(loaded_images[idx].width for idx in row_imgs)
                    + col_gap_px * (len(row_imgs) - 1)
            )
            row_x = margin_px + (max_row_width - row_content_width) // 2

            x = row_x
            for cell in row:
                if cell is not None:
                    im = loaded_images[cell]
                    canvas.paste(im, (int(x), int(y)))
                    x += im.width + col_gap_px
                else:
                    # blank cell: skip width
                    x += 0

    else:
        # Original automatic mode
        idx = 0
        for r in range(n_rows_eff):
            y = start_y + r * (target_height + row_gap_px)
            row_imgs = loaded_images[idx: idx + n_cols_eff]
            row_width = row_widths[r]
            row_x = margin_px + (max_row_width - row_width) // 2

            x = row_x
            for im in row_imgs:
                canvas.paste(im, (int(x), int(y)))
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

    layout = [[1,0]]  # switch the order
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
        layout=layout
    )


# Optional CLI for manual testing
if __name__ == "__main__":
    out = Path("out")
    core = allocate_core()

    assemble_process_figures(core, outdir=out, n_rows=2)
    assemble_type_figures(core, outdir=out, n_rows=1)
