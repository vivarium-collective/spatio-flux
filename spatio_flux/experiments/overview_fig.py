# spatio_flux/experiments/overview_fig.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from bigraph_viz import plot_bigraph
from spatio_flux.plots.colors import build_plot_settings


# ---------------------------------------------------------------------
# Types: example payloads to plot
# ---------------------------------------------------------------------

SPATIO_FLUX_TYPE_EXAMPLES = {
    "fields": {
        "substrate_id": {
            "_type": "count_concentration_volume",
            "volume": 1,
            "count": 1,
            "concentration": 1,
        },
    },
    "particles": {
        "particle_id": {
            "_type": "complex_particle",
            "position": [1.0, 1.0],
            "mass": 2.0,
            "velocity": [0.5, -0.5],
        },
    },
}


# ---------------------------------------------------------------------
# Plotting: processes (from core.link_registry)
# ---------------------------------------------------------------------

def plot_all_process_links(
    core,
    outdir: str | Path = "out",
    *,
    include_module_prefix: str = "spatio_flux.processes",
    exclude_names: Sequence[str] = ("edge",),
) -> List[Path]:
    """
    Plot each *process link* from core.link_registry as a single link node:

        state = { name: {"_type": "link", "address": f"local:{module.Class}"} }

    We prefer short-name registry keys (key == cls.__name__) to avoid plotting
    both the short-name and fully-qualified duplicates.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_settings = build_plot_settings(particle_ids=["particle", "complex_particle"])
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

    reg = getattr(core, "link_registry", None)
    if not isinstance(reg, dict):
        raise TypeError("Expected core.link_registry to be a dict[str, type]")

    # collect items in stable order
    items: List[Tuple[str, type]] = []
    for key, cls in reg.items():
        if not isinstance(key, str) or not isinstance(cls, type):
            continue
        if key in exclude_names:
            continue

        if include_module_prefix:
            mod = getattr(cls, "__module__", "")
            if not mod.startswith(include_module_prefix):
                continue

        # prefer short-name keys to avoid fq duplicates
        if key != getattr(cls, "__name__", ""):
            continue

        items.append((key, cls))

    items.sort(key=lambda kv: kv[0].lower())

    generated: List[Path] = []
    for name, cls in items:
        address = f"{cls.__module__}.{cls.__name__}"
        state = {
            name: {
                "_type": "link",
                "address": f"local:{address}",
                "config": core.default(cls.config_schema),
            }
        }
        fname = f"{name}_process"

        try:
            plot_bigraph(
                state=state,
                core=core,
                out_dir=str(outdir),
                filename=fname,
                **plot_settings,
            )
        except Exception as e:
            print(f"⚠ Skipping {name}: {e}")
            continue

        png = outdir / f"{fname}.png"
        if png.exists():
            generated.append(png)

    return generated


# ---------------------------------------------------------------------
# Plotting: types (simple examples)
# ---------------------------------------------------------------------

def plot_all_types(
    core,
    outdir: str | Path = "out",
) -> List[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_settings = build_plot_settings(
        particle_ids=["particle_id"],
        conc_type_species=["substrate_id"],
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

    generated: List[Path] = []
    for type_name in sorted(SPATIO_FLUX_TYPE_EXAMPLES.keys()):
        type_state = SPATIO_FLUX_TYPE_EXAMPLES[type_name]
        fname = f"{type_name}_type"

        plot_bigraph(
            state={type_name: type_state},
            core=core,
            out_dir=str(outdir),
            filename=fname,
            **plot_settings,
        )

        png = outdir / f"{fname}.png"
        if png.exists():
            generated.append(png)

    return generated


# ---------------------------------------------------------------------
# Image grid assembly
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
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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

    loaded_images: List[Image.Image] = []
    for p in flat_paths:
        im = Image.open(p)
        im = _flatten_to_white(im)
        w, h = im.size
        new_w = int(round(w * target_height / h))
        loaded_images.append(im.resize((new_w, target_height), Image.LANCZOS))

    # Determine layout
    if layout is not None:
        for row in layout:
            for cell in row:
                if cell is not None and not (0 <= cell < len(loaded_images)):
                    raise ValueError(f"Invalid layout index {cell}")

        n_rows_eff = len(layout)
        row_widths = []
        for row in layout:
            row_imgs = [loaded_images[idx] for idx in row if idx is not None]
            w = sum(im.width for im in row_imgs) + col_gap_px * max(0, len(row_imgs) - 1)
            row_widths.append(w)
        max_row_width = max(row_widths)

    else:
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
        for _ in range(n_rows_eff):
            row_imgs = loaded_images[idx : idx + n_cols_eff]
            w = sum(im.width for im in row_imgs) + col_gap_px * max(0, len(row_imgs) - 1)
            row_widths.append(w)
            idx += len(row_imgs)
        max_row_width = max(row_widths)

    title_height = int(target_height * 0.4) if title else 0
    grid_height = n_rows_eff * target_height + row_gap_px * (n_rows_eff - 1)
    total_width = margin_px * 2 + max_row_width
    total_height = margin_px * 2 + title_height + grid_height

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if title:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(title_height * 0.6))
        except Exception:
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

    start_y = margin_px + title_height

    if layout is not None:
        for r, row in enumerate(layout):
            y = start_y + r * (target_height + row_gap_px)

            row_idxs = [idx for idx in row if idx is not None]
            row_content_width = (
                sum(loaded_images[idx].width for idx in row_idxs)
                + col_gap_px * max(0, len(row_idxs) - 1)
            )
            x = margin_px + (max_row_width - row_content_width) // 2

            for cell in row:
                if cell is None:
                    continue
                im = loaded_images[cell]
                canvas.paste(im, (int(x), int(y)))
                x += im.width + col_gap_px

    else:
        idx = 0
        for r in range(n_rows_eff):
            y = start_y + r * (target_height + row_gap_px)

            row_imgs = loaded_images[idx : idx + n_cols_eff]
            row_content_width = row_widths[r]
            x = margin_px + (max_row_width - row_content_width) // 2

            for im in row_imgs:
                canvas.paste(im, (int(x), int(y)))
                x += im.width + col_gap_px

            idx += len(row_imgs)

    out_path = outdir / save_name
    canvas.save(out_path)
    print(f"Saved image grid to: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------

def assemble_process_figures(
    core,
    outdir: str | Path = "out",
    *,
    target_height: int = 350,
    n_cols: Optional[int] = None,
    n_rows: Optional[int] = None,
    col_gap_px: int = 40,
    row_gap_px: int = 60,
    save_name: str = "process_overview.png",
    include_module_prefix: str = "spatio_flux.processes",
) -> Path:
    """
    1) Plot all process links from core.link_registry into PNGs
    2) Assemble into one overview PNG
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating process figures...")
    proc_paths = plot_all_process_links(
        core=core,
        outdir=outdir,
        include_module_prefix=include_module_prefix,
    )
    print(f"Generated {len(proc_paths)} process figs in {outdir}")

    return assemble_image_grid(
        outdir,
        proc_paths,
        target_height=target_height,
        n_cols=n_cols,
        n_rows=n_rows,
        col_gap_px=col_gap_px,
        row_gap_px=row_gap_px,
        save_name=save_name,
    )


def assemble_type_figures(
    core,
    outdir: str | Path = "out",
    *,
    target_height: int = 1200,
    n_cols: Optional[int] = None,
    n_rows: Optional[int] = None,
    col_gap_px: int = 40,
    row_gap_px: int = 60,
    save_name: str = "type_overview.png",
) -> Path:
    print("Generating type figures...")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    type_paths = plot_all_types(core=core, outdir=outdir)
    print(f"Generated {len(type_paths)} type figs in {outdir}")

    layout = [[1, 0]]  # switch the order (fields then particles)
    return assemble_image_grid(
        outdir,
        type_paths,
        target_height=target_height,
        n_cols=n_cols,
        n_rows=n_rows,
        col_gap_px=col_gap_px,
        row_gap_px=row_gap_px,
        margin_px=0,
        save_name=save_name,
        layout=layout,
    )


if __name__ == "__main__":
    # Minimal CLI usage:
    #   python -m spatio_flux.experiments.overview_fig
    from process_bigraph import allocate_core

    out = Path("out")
    core = allocate_core()

    assemble_process_figures(core, outdir=out, n_rows=2)
    assemble_type_figures(core, outdir=out, n_rows=1)
