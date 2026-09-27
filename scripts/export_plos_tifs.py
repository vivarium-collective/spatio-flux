#!/usr/bin/env python
"""Export the paper figures as PLOS-ready 300-DPI TIFFs.

Rasterizes each study's composed figure SVG (studies/fig-0N/visualizations/
figure_N.svg) at ~2x the target size (via scripts/rasterize_svg.mjs, which
renders the embedded loom/sim panels), then downsamples to a 300-DPI LZW TIFF
sized to PLOS limits: 2250 px wide (7.5"), capped at 2625 px tall (8.75").
A figure taller than that at full width is fit to the height instead.

    python scripts/export_plos_tifs.py --out /path/to/figures_tif
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
RASTER = WS / "scripts" / "rasterize_svg.mjs"
FIGS = [1, 2, 3, 4, 5, 6, 7, 8]

TARGET_W = 2250      # 7.5 in at 300 dpi (PLOS double-column max width)
MAX_H = 2625         # 8.75 in at 300 dpi (PLOS max height)
DPI = 300


def _svg_size(svg: Path) -> tuple[float, float]:
    m = re.search(r'width="([0-9.]+)"[^>]*height="([0-9.]+)"', svg.read_text())
    if not m:
        raise SystemExit(f"no width/height in {svg}")
    return float(m.group(1)), float(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir for Fig<N>.tif")
    args = ap.parse_args()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in FIGS:
        svg = WS / "studies" / f"fig-0{n}" / "visualizations" / f"figure_{n}.svg"
        if not svg.is_file():
            print(f"[skip] fig{n}: missing {svg}")
            continue
        sw, sh = _svg_size(svg)
        # target box: full width, unless that would overflow the height cap.
        if sh / sw * TARGET_W > MAX_H:
            out_h = MAX_H
            out_w = round(MAX_H * sw / sh)
        else:
            out_w = TARGET_W
            out_h = round(TARGET_W * sh / sw)
        # rasterize at ~2x the output width for a crisp downsample.
        scale = max(2.0, (out_w * 2) / sw)
        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / f"fig{n}.png"
            subprocess.run(["node", str(RASTER), str(svg), str(png), f"{scale:.3f}"],
                           check=True, cwd=str(WS), capture_output=True)
            tif = out_dir / f"Fig{n}.tif"
            subprocess.run([
                "magick", str(png),
                "-resize", f"{out_w}x{out_h}",
                "-background", "white", "-flatten",
                "-units", "PixelsPerInch", "-density", str(DPI),
                "-compress", "lzw", str(tif),
            ], check=True)
        print(f"[ok] Fig{n}.tif  {out_w}x{out_h} @ {DPI}dpi")


if __name__ == "__main__":
    main()
