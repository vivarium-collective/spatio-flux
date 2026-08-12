#!/usr/bin/env python
"""Regenerate Figure 8 panel b — the spatio-flux REFERENCE MODEL over time.

Runs the `spatioflux_reference_demo` composite (fields + diffusion + a Monod-
kinetics array + Newtonian particles carrying per-particle dFBA that divide as
they grow) for RUNTIME minutes on the real 10x10 grid, then plots a
rows x cols(time) filmstrip: the glucose / acetate / dissolved-biomass fields as
heatmaps, plus a particles row scattering every particle at its position, colored
by dominant strain (ecoli_1 / ecoli_2) and sized by mass.

Stochastic (random particle ids + Brownian/Newtonian motion) — this writes a
fresh sample each run; it is not drift-checked.

    python scripts/build_fig08b_snapshots.py
"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from process_bigraph import Composite, gather_emitter_results, allocate_core
from process_bigraph.composite_generator import build_generator
from spatio_flux.composites import REGISTRY
from spatio_flux.library.tools import get_standard_emitter
from spatio_flux.plots.plot import plot_snapshots_grid

RUNTIME = 120.0            # longer run → more of the reference model's behaviour
BOUNDS = (50.0, 50.0)
N_PARTICLES = 3            # a few initial particles; the colony grows via inflow + division
# The reference composite ships with add_rate=0.0 (no inflow), so a lone particle
# just falls and the row looks empty. The ORIGINAL reference figure grew a settling
# colony because particles were seeded from the top boundary over time; re-enable
# that here (snapshot-sim only — the committed composite is untouched) so the
# particles row is the dense, dividing pile it used to be.
ADD_RATE = 0.3             # particle inflow event-rate (1/sec) at the top boundary
ADD_BOUNDARY = ["top"]
# The reference model's physics are very gentle (gravity=-1.0, jitter=1e-2), so
# particles drift and fill the box instead of dropping. Firm gravity + low jitter
# make them fall and settle into a growing bottom sediment.
GRAVITY = -9.81
JITTER = 1e-3
DAMPING = 0.90
COLLISION_R = 1.8          # domain-unit render radius (slightly > contact so disks touch, no gaps)
FIELDS = ["glucose", "acetate", "dissolved biomass"]
STRAINS = ["ecoli_1", "ecoli_2"]
N_SNAP = 6


def _strain_color(sub):
    """Blue = ecoli_1-dominant, red = ecoli_2-dominant, purple = 50/50."""
    e1 = float((sub or {}).get("ecoli_1", 0.0) or 0.0)
    e2 = float((sub or {}).get("ecoli_2", 0.0) or 0.0)
    tot = e1 + e2
    f = 0.5 if tot <= 0 else e1 / tot
    return plt.cm.coolwarm_r(f)


def _particle_circles(frame):
    """(circles, colors) drawn at the TRUE collision radius so a settled colony
    reads as touching disks — not scattered dots."""
    parts = frame.get("particles")
    if not isinstance(parts, dict):
        return [], []
    circles, colors = [], []
    for pid, p in parts.items():
        if pid.startswith("_") or not isinstance(p, dict) or "position" not in p:
            continue
        pos = p["position"]
        circles.append(Circle((float(pos[0]), float(pos[1])), COLLISION_R))
        colors.append(_strain_color(p.get("sub_masses")))
    return circles, colors


def main() -> None:
    core = allocate_core()
    entry = next(e for e in REGISTRY.values() if e.name == "spatioflux_reference_demo")
    doc = build_generator(entry, overrides={"n_particles": N_PARTICLES}, core=core)  # real 10x10 grid
    state = doc["state"] if isinstance(doc, dict) and "state" in doc else doc
    # Re-enable top-boundary particle inflow so a settling colony grows over time.
    eb = state.get("enforce_boundaries", {}).get("config")
    if isinstance(eb, dict):
        eb["add_rate"] = ADD_RATE
        eb["boundary_to_add"] = list(ADD_BOUNDARY)
    # Firm up the physics so particles drop and settle (see constants above).
    npc = state.get("newtonian_particles", {}).get("config")
    if isinstance(npc, dict):
        npc["gravity"] = GRAVITY
        npc["jitter_per_second"] = JITTER
        npc["damping_per_second"] = DAMPING
    state["emitter"] = get_standard_emitter(state_keys=list(state.keys()), subsample=1)
    sim = Composite({"state": state}, core=core)
    sim.run(RUNTIME)
    results = gather_emitter_results(sim)[("emitter",)]
    results = [r for r in results if isinstance(r.get("fields"), dict)]
    n = len(results)
    print(f"ran spatioflux_reference_demo — {n} emitted frames over {RUNTIME:.0f} min")

    # Give every particle a FIXED render radius = the pymunk contact radius, so the
    # settled colony reads as touching disks (the bogus mass-derived `radius` field
    # is way out of domain scale). Then render with the existing spatio-flux grid,
    # which draws each particle as a PIE of its sub_masses (ecoli_1 / ecoli_2).
    for r in results:
        for p in (r.get("particles") or {}).values():
            if isinstance(p, dict) and "position" in p:
                p["radius"] = COLLISION_R
    viz = Path("studies/fig-08/visualizations"); viz.mkdir(parents=True, exist_ok=True)
    plot_snapshots_grid(
        results,
        field_names=FIELDS,
        n_snapshots=N_SNAP,
        bounds=BOUNDS,
        particles_row="separate",
        particle_radius_key="radius",
        radius_fallback_from_mass=False,
        show_particle_submasses=True,
        submasses_key="sub_masses",
        submass_color_map={"ecoli_1": "#2563eb", "ecoli_2": "#dc2626"},  # blue / red
        submass_draw_legend=True,
        submass_legend_fontsize=12,
        particle_edgecolor="#1e293b",
        particle_linewidth=0.35,
        row_label_fontsize=16,
        time_label_fontsize=15,
        wspace=0.06, hspace=0.10,
        out_dir=str(viz),
        filename="fig08b-reference-snapshots.png",
    )
    print("wrote", viz / "fig08b-reference-snapshots.png")


if __name__ == "__main__":
    main()
