from pathlib import Path

from bigraph_viz import plot_bigraph
from process_bigraph import Composite
from PIL import Image, ImageDraw, ImageFont

from spatio_flux.experiments.test_suite import get_newtonian_particle_comets_doc
from spatio_flux.plots.colors import build_plot_settings
from spatio_flux.plots.plot_core import assemble_process_figures, assemble_type_figures, assemble_image_grid
from spatio_flux import build_core


def composite_figure(
        core,
        outdir="out",
        config=None
):
    config = config or {'n_bins': (2, 2), 'bounds': (1.0, 1.0)}
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
            show_types=True,
            collapse_redundant_processes=False,
            value_char_limit=20,
            type_char_limit=40,
        )
    )

    plot_bigraph(
        state=plot_state,
        schema=plot_schema,
        core=core,
        out_dir=outdir,
        filename=f"metacomposite_bigraph",
        **plot_settings
    )


def assemble_ABC_overview(
    core,
    outdir="out",
    *,
    width_A=1800,
    width_B=1800,
    width_C=1800,
    row_gap_px=40,
    margin_px=30,
    save_name="overview_ABC.png",
):
    """
    Build a 3-panel figure:
        A. Processes overview
        B. Types overview
        C. Composite example

    Each panel is resized to a *target width* instead of a target height.
    Heights adjust automatically to preserve aspect ratio.
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
        n_rows=2,
        save_name="panel_B_types.png",
    )

    # --- Panel C: composite ---
    print("Generating panel C (composite bigraph)...")
    composite_figure(core, outdir=outdir)
    fig_C_path = outdir / "metacomposite_bigraph.png"

    # --- Resize each to target width ---
    def resize_to_width(path, target_w):
        im = Image.open(path)
        w, h = im.size
        scale = target_w / w
        new_h = int(h * scale)
        return im.resize((target_w, new_h), Image.LANCZOS)

    imA = resize_to_width(fig_A_path, width_A)
    imB = resize_to_width(fig_B_path, width_B)
    imC = resize_to_width(fig_C_path, width_C)

    # --- Combined canvas size ---
    total_w = max(width_A, width_B, width_C) + 2 * margin_px
    total_h = (
        margin_px +
        imA.height +
        row_gap_px +
        imB.height +
        row_gap_px +
        imC.height +
        margin_px
    )

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # --- Paste images ---
    y = margin_px
    canvas.paste(imA, (margin_px, y))
    y += imA.height + row_gap_px

    canvas.paste(imB, (margin_px, y))
    y += imB.height + row_gap_px

    canvas.paste(imC, (margin_px, y))

    # --- Add labels A, B, C ---
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()

    draw.text((margin_px + 5, margin_px + 5), "A", fill=(0, 0, 0), font=font)
    draw.text((margin_px + 5, imA.height + row_gap_px + margin_px + 5),
              "B", fill=(0, 0, 0), font=font)
    draw.text((margin_px + 5,
               imA.height + imB.height + 2 * row_gap_px + margin_px + 5),
              "C", fill=(0, 0, 0), font=font)

    outpath = outdir / save_name
    canvas.save(outpath)
    print(f"Saved combined A+B+C figure: {outpath}")

    return outpath



# Optional CLI for manual testing
if __name__ == "__main__":
    out = Path("out")
    core = build_core()

    assemble_ABC_overview(core)
