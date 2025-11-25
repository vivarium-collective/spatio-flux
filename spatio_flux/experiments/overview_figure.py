from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

from bigraph_viz import plot_bigraph
from process_bigraph import Composite

from spatio_flux.experiments.test_suite import get_newtonian_particle_comets_doc
from spatio_flux.plots.colors import build_plot_settings
from spatio_flux.plots.plot_core import assemble_process_figures, assemble_type_figures, assemble_image_grid
from spatio_flux import build_core




def composite_figure(
        core,
        outdir="out",
        filename="metacomposite_bigraph",
        config=None
):
    config = config or {'n_bins': (5, 10), 'bounds': (1.0, 2.0)}
    document = get_newtonian_particle_comets_doc(core, config)
    sim = Composite(document, core=core)

    # Visualize initial composition
    plot_state = {k: v for k, v in sim.state.items() if k not in ['global_time', 'emitter']}
    plot_schema = {k: v for k, v in sim.composition.items() if k not in ['global_time', 'emitter']}

    # get particles for coloring
    particle_ids = []
    if 'particles' in plot_state and plot_state['particles']:
        particle_ids = list(plot_state['particles'].keys())
    plot_settings = build_plot_settings(
        particle_ids=particle_ids,
        # conc_type_species=['conc_counts_volume', 'substrate'],
    )
    plot_settings.update(
        dict(
            dpi="300",
            show_values=True,
            show_types=False,
            value_char_limit=20,
            type_char_limit=40,
            collapse_redundant_processes={
                'exclude': [('particle_movement',), ('particle_division',)]
            },
        )
    )

    plot_bigraph(
        state=plot_state,
        schema=plot_schema,
        core=core,
        out_dir=outdir,
        filename=filename,
        **plot_settings
    )


def assemble_AB_overview(
    core,
    outdir="out",
    *,
    width_A=1800,
    width_B=1700,
    row_gap_px=40,
    margin_px=30,
    save_name="types_overview.png",
    label_font_size=50,
    dpi=300,
):
    """
    Build a 2-panel figure:
        A. Processes overview
        B. Types overview
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Panel A: processes ---
    print("Generating panel A (process overview)...")
    fig_A_path = assemble_process_figures(
        core,
        outdir=outdir,
        n_rows=2,
        save_name="panel_A_process.png",
    )

    # --- Panel B: types ---
    print("Generating panel B (type overview)...")
    fig_B_path = assemble_type_figures(
        core,
        outdir=outdir,
        n_rows=1,
        save_name="panel_B_types.png",
    )

    # --- Resize each to target width (downscale only) ---
    def resize_to_width(path, target_w):
        im = Image.open(path)
        w, h = im.size
        if w <= target_w:
            return im.copy()  # keep original resolution
        scale = target_w / w
        new_h = int(h * scale)
        return im.resize((target_w, new_h), Image.LANCZOS)

    imA = resize_to_width(fig_A_path, width_A)
    imB = resize_to_width(fig_B_path, width_B)

    # --- Combined canvas size ---
    total_w = max(imA.width, imB.width) + 2 * margin_px
    total_h = (
        margin_px +
        imA.height +
        row_gap_px +
        imB.height +
        margin_px
    )

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # --- Paste panel A ---
    y = margin_px
    canvas.paste(imA, (margin_px, y))
    y += imA.height + row_gap_px

    # --- Paste panel B ---
    canvas.paste(imB, (margin_px, y))

    # --- Font for labels ---
    label_font_path = fm.findfont("Arial", fallback_to_default=True)
    font = ImageFont.truetype(label_font_path, label_font_size)

    def draw_label(letter, x, y, pad=10):
        bbox = draw.textbbox((x, y), letter, font=font)
        x0, y0, x1, y1 = bbox
        x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
        draw.rectangle([x0, y0, x1, y1], fill="white")
        draw.text((x, y), letter, fill=(0, 0, 0), font=font)

    # --- Labels A and B ---
    draw_label("a.", margin_px + 5, margin_px + 5)
    draw_label("b.", margin_px + 5, imA.height + row_gap_px + margin_px + 5)

    # --- Save final figure ---
    outpath = outdir / save_name
    canvas.save(outpath, dpi=(dpi, dpi))

    print(f"Saved combined A+B figure: {outpath}")
    return outpath

# Optional CLI for manual testing
if __name__ == "__main__":
    out = Path("out")
    core = build_core()

    composite_figure(core, outdir=out)
    assemble_AB_overview(core)
