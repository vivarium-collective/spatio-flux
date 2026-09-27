#!/usr/bin/env python
"""Regenerate Figure 7 panels b / c / d as HIGH-RESOLUTION, LARGE-FONT plots.

The three simulation panels were originally emitted tiny (~489x360 px) with
~11pt titles and ~8pt legends, so in the composed figure they read blurry and
their legends are unreadable. This script re-runs the same three composites the
studies use and re-renders each panel crisp and large, keeping the SAME series,
colors, units, titles, and axis semantics as the originals.

Panel  Study            Title            Notes
  b    monod_kinetics   "Monod kinetics" glucose/acetate/biomass vs Time (min)
  c    ecoli_core_dfba  "dFBA"           glucose/acetate/biomass vs Time (min)
  d    community_dfba   "hybrid community" 7 organisms, log y, normalized

Data source: each composite is built via the study's baseline (composite ref +
params from studies/<slug>/study.yaml), run for 60 min through the same
``run_composite_document`` path the study runner uses, then the in-memory
trajectory is plotted. No fabricated data.

The plotting faithfully reproduces ``spatio_flux.plots.plot.plot_time_series``
(series labels with units, the "Value[ (normalized)][ (log)]" y-label, the
STANDARD_FIELD_COLORS-then-tab20 color assignment, legend outside right), only
scaled up: figsize (8, 5.7) @ dpi 200 (~1600x1140 px), title 22 / labels 18 /
ticks 15 / legend 15, linewidth 2.5, a light grid (alpha 0.3), no legend frame.

    python scripts/build_fig07bcd_plots.py
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from spatio_flux.core import build_core
from pbg_superpowers.composite_generator import build_generator, _REGISTRY
from spatio_flux.library.tools import run_composite_document
from spatio_flux.plots.plot import sort_results
from spatio_flux.analysis.flush_spec import STANDARD_FIELD_COLORS
from spatio_flux.runners.run_study import _resolve_runtime

WS = Path(__file__).resolve().parents[1]
VIZ = WS / "studies" / "fig-07" / "visualizations"

# ---- large-font / high-res style (consistent across all three panels) ----
FIGSIZE = (8.0, 5.7)   # aspect ~1.4 (w/h); at dpi 200 -> ~1600x1140 px
DPI = 200
TITLE_FS = 22
LABEL_FS = 18
TICK_FS = 15
LEGEND_FS = 15
LINEWIDTH = 2.5
GRID_ALPHA = 0.3
LEGEND_WIDTH_FRACTION = 0.30  # matches plot_time_series default

# Per-panel definitions. field_names="$all_fields"/"$community_species" resolve
# from the run's state exactly as the study runner resolves them.
CONC_UNITS = {"glucose": "mM", "acetate": "mM", "biomass": "gDW",
              "kinetic_biomass": "gDW"}
PANELS = [
    {
        "filename": "fig07b-monod-kinetics.png",
        "slug": "monod_kinetics",
        "title": "Monod kinetics",
        "field_names": "$all_fields",
        "field_units": {"glucose": "mM", "acetate": "mM", "biomass": "gDW"},
        "log_scale": False, "normalize": False,
    },
    {
        "filename": "fig07c-dfba.png",
        "slug": "ecoli_core_dfba",
        "title": "dFBA",
        "field_names": "$all_fields",
        "field_units": CONC_UNITS,
        "log_scale": False, "normalize": False,
    },
    {
        "filename": "fig07d-hybrid-community.png",
        "slug": "community_dfba",
        "title": "hybrid community",
        "field_names": "$community_species",
        "field_units": {},
        "log_scale": True, "normalize": True,
    },
]


def _baseline(slug):
    """(composite ref, params) from studies/<slug>/study.yaml — authoritative."""
    spec = yaml.safe_load((WS / "studies" / slug / "study.yaml").read_text())
    b = spec["baseline"][0]
    return b["composite"], b.get("params", {}) or {}


def _run(slug, time_min=60):
    """Build the study baseline, run it, and return (sorted_results, state)."""
    ref, params = _baseline(slug)
    core = build_core()
    doc = build_generator(_REGISTRY[ref], overrides=params, core=core)
    out = WS / "studies" / slug / "charts"
    out.mkdir(parents=True, exist_ok=True)
    results, _pt, _fw = run_composite_document(
        doc, core=core, name=slug, time=time_min, outdir=str(out),
        show_types=True, show_values=True)
    state = doc.get("state", doc)
    return sort_results(results), state


def _assign_colors(field_names, field_colors):
    """Reproduce plot_time_series: semantic colors first, then a tab20 cycle."""
    cycle = list(plt.get_cmap("tab20").colors)
    assigned = dict(field_colors)
    idx = 0
    for f in field_names:
        if f in assigned:
            continue
        if idx < len(cycle):
            assigned[f] = cycle[idx]
            idx += 1
    return assigned


def _series_label(name, units):
    u = units.get(name)
    return f"{name} ({u})" if u else name


def _render(panel, sorted_results, state):
    """Render one large, high-res panel; return output path + pixel size."""
    ctx = _resolve_runtime(state)
    fn = panel["field_names"]
    field_names = ctx.get(fn, fn) if isinstance(fn, str) and fn.startswith("$") else fn
    units = panel["field_units"]
    normalize = panel["normalize"]
    log_scale = panel["log_scale"]
    colors = _assign_colors(field_names, STANDARD_FIELD_COLORS)

    times = sorted_results["time"]
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for name in field_names:
        if name not in sorted_results["fields"]:
            print(f"  (skip: '{name}' not in results)")
            continue
        data = sorted_results["fields"][name]
        if normalize:
            init = data[0] if data[0] != 0 else 1e-12
            data = [v / init for v in data]
        ax.plot(times, data, label=_series_label(name, units),
                color=colors.get(name), linewidth=LINEWIDTH)

    if log_scale:
        ax.set_yscale("log")

    # y-label mirrors plot_time_series: base + optional units/normalized/log.
    unique_units = {units.get(f) for f in field_names if units.get(f)}
    units_suffix = f" ({list(unique_units)[0]})" if len(unique_units) == 1 else ""
    norm_suffix = " (normalized)" if normalize else ""
    scale_suffix = " (log)" if log_scale else ""
    ax.set_xlabel("Time (min)", fontsize=LABEL_FS)
    ax.set_ylabel(f"Value{units_suffix}{norm_suffix}{scale_suffix}", fontsize=LABEL_FS)
    ax.set_title(panel["title"], fontsize=TITLE_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FS * 0.9)
    ax.grid(True, alpha=GRID_ALPHA)

    # Legend outside to the right (no overlap), left-aligned, no frame.
    fig.subplots_adjust(right=1.0 - LEGEND_WIDTH_FRACTION)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0, fontsize=LEGEND_FS, frameon=False)
    try:
        leg._legend_box.align = "left"
    except Exception:
        pass

    VIZ.mkdir(parents=True, exist_ok=True)
    path = VIZ / panel["filename"]
    fig.savefig(path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    w, h = _png_size(path)
    return path, (w, h)


def _png_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def main():
    for panel in PANELS:
        print(f"[{panel['filename']}] running {panel['slug']} ...")
        sorted_results, state = _run(panel["slug"])
        path, (w, h) = _render(panel, sorted_results, state)
        print(f"  wrote {path.relative_to(WS)}  ({w}x{h} px)")


if __name__ == "__main__":
    main()
