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
from process_bigraph import Composite, gather_emitter_results, allocate_core
from process_bigraph.composite_generator import build_generator
from spatio_flux.composites import REGISTRY
from spatio_flux.library.tools import get_standard_emitter

RUNTIME = 120.0            # longer run → more of the reference model's behaviour
BOUNDS = (50.0, 50.0)
N_PARTICLES = 1            # start with one, like the original reference figure
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
FIELDS = ["glucose", "acetate", "dissolved biomass"]
STRAINS = ["ecoli_1", "ecoli_2"]
STRAIN_COLORS = {"ecoli_1": "#2563eb", "ecoli_2": "#dc2626"}  # blue / red
N_SNAP = 6


def _particles(frame):
    """(xs, ys, colors, sizes) for every particle in an emitted frame."""
    parts = frame.get("particles")
    if not isinstance(parts, dict):
        return [], [], [], []
    xs, ys, cols, sizes = [], [], [], []
    for pid, p in parts.items():
        if pid.startswith("_") or not isinstance(p, dict) or "position" not in p:
            continue
        pos = p["position"]
        xs.append(float(pos[0])); ys.append(float(pos[1]))
        sub = p.get("sub_masses") or {}
        m = {s: float(sub.get(s, 0.0) or 0.0) for s in STRAINS}
        dom = max(m, key=m.get) if any(m.values()) else STRAINS[0]
        cols.append(STRAIN_COLORS[dom])
        sizes.append(12.0 + 70.0 * float(p.get("mass", 0.0) or 0.0))
    return xs, ys, cols, sizes


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

    idxs = np.linspace(0, n - 1, N_SNAP).astype(int)
    times = [r.get("global_time", 0.0) for r in results]
    labels = [f"t = {int(round(times[i]))} min" for i in idxs]

    rows = FIELDS + ["particles"]
    fig, axes = plt.subplots(
        len(rows), N_SNAP, figsize=(N_SNAP * 2.4, len(rows) * 2.4),
        gridspec_kw={"wspace": 0.06, "hspace": 0.10},
    )

    # Field rows — heatmaps, one shared vmax per field.
    for r, fname in enumerate(FIELDS):
        frames = [np.array(results[i]["fields"][fname], dtype=float) for i in idxs]
        vmax = max((f.max() for f in frames), default=1.0) or 1.0
        im = None
        for c, fr in enumerate(frames):
            ax = axes[r][c]
            im = ax.imshow(fr, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(fname, fontsize=16)
        cb = fig.colorbar(im, ax=list(axes[r]), fraction=0.025, pad=0.02)
        cb.ax.tick_params(labelsize=13)

    # Particles row — scatter positions colored by dominant strain, sized by mass.
    pr = len(FIELDS)
    for c, i in enumerate(idxs):
        ax = axes[pr][c]
        xs, ys, cols, sizes = _particles(results[i])
        if xs:
            ax.scatter(xs, ys, s=sizes, c=cols, alpha=0.85,
                       edgecolors="#33415580", linewidths=0.3)
        ax.set_xlim(0, BOUNDS[0]); ax.set_ylim(0, BOUNDS[1])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(labels[c], fontsize=15)
        if c == 0:
            ax.set_ylabel("particles", fontsize=16)
    # Strain legend on the particles row.
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=9,
                      markerfacecolor=STRAIN_COLORS[s], markeredgecolor="none", label=s)
               for s in STRAINS]
    axes[pr][-1].legend(handles=handles, loc="lower right", fontsize=11,
                        framealpha=0.85, handletextpad=0.3, borderpad=0.3)

    out = Path("studies/fig-08/visualizations/fig08b-reference-snapshots.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
