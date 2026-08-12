#!/usr/bin/env python
"""Regenerate Figure 8 panel b — the spatio-flux REFERENCE MODEL over time.

Reproduces exactly how the test-suite report renders reference_demo_x2y2: run the
composite with its NATURAL dynamics via ``run_composite_document`` (the study
runner's path — which actually grows the per-particle dFBA so particles DIVIDE),
then render with ``plot_snapshots_grid`` (particles as pies of their ecoli_1 /
ecoli_2 sub-masses). No physics overrides — the lone particle grows and divides
into a tidy row, rather than being flooded by an artificial inflow.

    python scripts/build_fig08b_snapshots.py
"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import tempfile
from pathlib import Path

from process_bigraph.composite_generator import build_generator
from spatio_flux.composites import REGISTRY
from spatio_flux.core import build_core
from spatio_flux.library.tools import run_composite_document
from spatio_flux.plots.plot import plot_snapshots_grid

RUNTIME = 120.0
GENERATOR = "reference_demo_x2y2"                 # finer grid → richer division (test-suite scenario)
FIELDS = ["glucose", "acetate", "dissolved biomass"]
# The test-suite submass palette (flush_spec._SUBMASS_COLORS): ecoli_1 blue, ecoli_2 red.
SUBMASS_COLORS = {"ecoli_1": "#1f77b4", "ecoli_2": "#d62728"}
N_SNAP = 8


def _bounds(state) -> tuple:
    npc = (state.get("newtonian_particles") or {}).get("config") or {}
    b = npc.get("bounds") or (50.0, 50.0)
    return (float(b[0]), float(b[1]))


def main() -> None:
    core = build_core()
    entry = next(e for e in REGISTRY.values() if e.name == GENERATOR)
    doc = build_generator(entry, overrides={}, core=core)   # DEFAULT dynamics

    # Run through the study runner's path — this is what makes the per-particle
    # dFBA grow (and therefore divide); a bare Composite.run does not.
    with tempfile.TemporaryDirectory() as tmp:
        out = run_composite_document(doc, core=core, name="fig08b_reference",
                                     time=RUNTIME, outdir=tmp)
    results = out[0] if isinstance(out, tuple) else out
    bounds = _bounds(doc.get("state", doc))

    viz = Path("studies/fig-08/visualizations"); viz.mkdir(parents=True, exist_ok=True)
    plot_snapshots_grid(
        results,
        field_names=FIELDS,
        n_snapshots=N_SNAP,
        bounds=bounds,
        particles_row="separate",
        show_particle_submasses=True,
        submass_draw_legend=True,
        submass_color_map=SUBMASS_COLORS,
        submass_legend_fontsize=12,
        time_units="min",
        row_label_fontsize=16,
        time_label_fontsize=15,
        wspace=0.06, hspace=0.10,
        out_dir=str(viz),
        filename="fig08b-reference-snapshots.png",
    )
    print("wrote", viz / "fig08b-reference-snapshots.png")


if __name__ == "__main__":
    main()
