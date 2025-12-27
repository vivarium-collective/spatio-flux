"""
Figure B: stack existing spatioflux figures vertically.

- Figure (a): {TEST_NAME}_viz.png
- Figure (b): *_snapshots.png, resized uniformly to match width of (a)

No distortion, no overlap, clean labels, tight PDF.
"""

import time
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from spatio_flux.experiments.test_suite import (
    SIMULATIONS,
    DEFAULT_RUNTIME_LONG,
    allocate_core,
    prepare_output_dir,
    run_composite_document,
)

# -------------------------
# Config
# -------------------------
TEST_NAME = "spatioflux_reference_demo"
OUT_DIR = Path("out")
OUT_FIGURE = OUT_DIR / "mega-composite.pdf"
DPI = 300

LABEL_MARGIN_PX = 70     # vertical space above each panel
LABEL_FONT_FRAC = 0.035 # label size relative to panel width


# -------------------------
# Run simulation + plots
# -------------------------
def run_reference_composite_and_plots():
    prepare_output_dir(str(OUT_DIR))
    core = allocate_core()

    if TEST_NAME not in SIMULATIONS:
        raise KeyError(f"No simulation named '{TEST_NAME}'")

    sim_info = SIMULATIONS[TEST_NAME]
    config = sim_info.get("config", {}) or {}

    print(f"\n🚀 Running test: {TEST_NAME}")
    doc = sim_info["doc_func"](core=core, config=config)

    runtime = sim_info.get("time", DEFAULT_RUNTIME_LONG)
    t0 = time.time()

    results = run_composite_document(
        doc,
        core=core,
        name=TEST_NAME,
        time=runtime,
        outdir=str(OUT_DIR),
        show_types=False,
        show_values=False,
    )

    print(f"✅ Completed in {time.time() - t0:.2f}s")

    plot_config = sim_info.get("plot_config", {}) or {}
    sim_info["plot_func"](results, doc.get("state", doc), config=plot_config)

    print("✅ Plots done.")


# -------------------------
# Helpers
# -------------------------
def trim_white(img):
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img.convert("RGB"), bg)
    bbox = diff.getbbox()
    return img.crop(bbox) if bbox else img


def find_first(out_dir, suffix):
    matches = sorted(out_dir.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No file ending with {suffix}")
    return matches[0]


def add_label_margin(img, label):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + LABEL_MARGIN_PX), (255, 255, 255))
    canvas.paste(img, (0, LABEL_MARGIN_PX))

    draw = ImageDraw.Draw(canvas)

    font_size = max(24, int(w * LABEL_FONT_FRAC))
    font_path = fm.findfont("DejaVu Sans", fallback_to_default=True)
    font = ImageFont.truetype(font_path, font_size)

    text = f"{label}."
    x = int(0.02 * w)
    y = (LABEL_MARGIN_PX - font_size) // 2

    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return canvas


def resize_to_width(img, target_w):
    if img.width == target_w:
        return img
    scale = target_w / img.width
    new_h = int(round(img.height * scale))
    return img.resize((target_w, new_h), Image.LANCZOS)


def stack_vertically(images):
    widths = [im.width for im in images]
    heights = [im.height for im in images]

    canvas = Image.new(
        "RGB",
        (max(widths), sum(heights)),
        (255, 255, 255),
    )

    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height

    return canvas


def save_tight_pdf(img, path):
    w_px, h_px = img.size
    fig = plt.figure(figsize=(w_px / DPI, h_px / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# -------------------------
# Assemble Figure B
# -------------------------
def assemble_figure_B():
    viz = Image.open(OUT_DIR / f"{TEST_NAME}_viz.png")
    snaps = Image.open(find_first(OUT_DIR, "_snapshots.png"))

    viz = trim_white(viz)
    snaps = trim_white(snaps)

    viz = add_label_margin(viz, "a")
    snaps = add_label_margin(snaps, "b")

    # 🔑 resize b to match width of a (aspect preserved)
    snaps = resize_to_width(snaps, viz.width)

    stacked = stack_vertically([viz, snaps])
    save_tight_pdf(stacked, OUT_FIGURE)

    print(f"\n📄 Saved: {OUT_FIGURE}")


# -------------------------
# Main
# -------------------------
def main():
    run_reference_composite_and_plots()
    assemble_figure_B()


if __name__ == "__main__":
    main()
