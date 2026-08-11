#!/usr/bin/env python
"""Regenerate Figure 7 panel h — Brownian particle traces.

Runs the ``brownian_particles`` composite with many particles + a healthy
add-rate so the trajectories fill the domain (the prior panel had one particle
and read as near-empty), then renders the static traces figure to
``studies/fig-07/visualizations/fig07h-brownian-traces.png``.

    python scripts/build_fig07h_traces.py
"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
from pathlib import Path
from process_bigraph import Composite, gather_emitter_results, allocate_core
from process_bigraph.composite_generator import build_generator
from spatio_flux.composites import REGISTRY
from spatio_flux.composites._constants import SQUARE_BOUNDS
from spatio_flux.plots.plot import plot_particle_traces
from spatio_flux.library.tools import get_standard_emitter

core = allocate_core()
entry = next(e for e in REGISTRY.values() if e.name == "brownian_particles")
# A moderate population of DISTINCT particles exploring the domain — rich but
# legible. (Too many particles over too long a run + time-darkened traces turned
# the panel into a muddy blob; keep it colourful and readable.)
doc = build_generator(entry, overrides={"n_particles": 16, "add_rate": 0.0,
                                        "diffusion_rate": 1.0}, core=core)
state = doc["state"]
# subsample=2 → fewer frames (thinner traces); the walk still fills the domain.
state["emitter"] = get_standard_emitter(state_keys=list(state.keys()), subsample=2)
sim = Composite({"state": state}, core=core)
sim.run(80.0)
results = gather_emitter_results(sim)[("emitter",)]
history = [ {pid: dict(p) for pid, p in (fr.get("particles") or {}).items() if isinstance(p, dict)}
            for fr in results ]
history = [h for h in history if h]
print("frames:", len(history), "| max particles:", max((len(h) for h in history), default=0))
out = Path("studies/fig-07/visualizations")
plot_particle_traces(history=list(history), bounds=SQUARE_BOUNDS,
                     out_dir=str(out), filename="fig07h-brownian-traces.png",
                     # brighter (high min_brightness) so late trace points don't
                     # darken to mud; thin, semi-transparent traces stay distinct.
                     units="µm", legend=False, trace_alpha=0.28,
                     min_brightness=0.45, max_brightness=1.0, dpi=200)
print("wrote", out / "fig07h-brownian-traces.png")
