#!/usr/bin/env python
"""Regenerate Figure 7 panel f — COMETS field snapshots — from the big-grid
`comets_spatial_dfba` composite (a single vectorized SpatialDFBA process over the
whole lattice), so the snapshots come from a much larger simulation than the old
per-bin dFBA scenario.

Runs the composite on a 20x20 grid, then plots a rows(fields) x cols(time) grid of
the glucose / acetate / dissolved-biomass fields evolving over time.

    python scripts/build_fig07f_snapshots.py
"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from process_bigraph import Composite, gather_emitter_results, allocate_core
from process_bigraph.composite_generator import build_generator
from spatio_flux.composites import REGISTRY
from spatio_flux.library.tools import get_standard_emitter

N_BINS = [20, 20]
RUNTIME = 120.0   # longer run → more of COMETS' long-term spatial behaviour
FIELDS = ["glucose", "acetate", "dissolved biomass"]
N_SNAP = 5

core = allocate_core()
entry = next(e for e in REGISTRY.values() if e.name == "comets_spatial_dfba")
doc = build_generator(entry, overrides={"n_bins": N_BINS}, core=core)
state = doc["state"] if isinstance(doc, dict) and "state" in doc else doc
state["emitter"] = get_standard_emitter(state_keys=list(state.keys()), subsample=1)
sim = Composite({"state": state}, core=core)
sim.run(RUNTIME)
results = gather_emitter_results(sim)[("emitter",)]
results = [r for r in results if isinstance(r.get("fields"), dict)]
n = len(results)
print(f"ran comets_spatial_dfba on {N_BINS} grid — {n} emitted frames")

idxs = np.linspace(0, n - 1, N_SNAP).astype(int)
times = [r.get("global_time", 0.0) for r in results]
labels = [f"t = {int(round(times[i]))} min" for i in idxs]

fig, axes = plt.subplots(
    len(FIELDS), N_SNAP, figsize=(N_SNAP * 2.4, len(FIELDS) * 2.4),
    # Tighten the grid: near-zero gaps between the square field tiles so the
    # panel reads as a filmstrip, not scattered thumbnails.
    gridspec_kw={"wspace": 0.06, "hspace": 0.10},
)
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
        if r == len(FIELDS) - 1:
            ax.set_xlabel(labels[c], fontsize=15)
    cb = fig.colorbar(im, ax=list(axes[r]), fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=13)

out = Path("studies/fig-07/visualizations/fig07f-comets-snapshots.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
