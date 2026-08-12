#!/usr/bin/env python
"""Run the fig-8 reference composite on its own and render a GIF of the particles
settling, so we can SEE the real dynamics (packing + per-strain composition).

- particles are drawn as circles at their TRUE collision radius (in domain units),
  so a settled sediment reads as touching disks — not scattered dots.
- each particle is colored by its ecoli_1 : ecoli_2 sub-mass ratio (blue =
  ecoli_1-dominant, red = ecoli_2-dominant, purple = 50/50), so ecoli_2 is
  visible when present.
- glucose is shown faintly behind the particles for spatial context.

Writes studies/fig-08/visualizations/fig08-reference.gif plus a static montage
fig08-reference-montage.png.

    python scripts/fig08_reference_gif.py
"""
import warnings; warnings.filterwarnings("ignore")
import io
import matplotlib; matplotlib.use("Agg")
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from process_bigraph import Composite, gather_emitter_results, allocate_core
from process_bigraph.composite_generator import build_generator
from spatio_flux.composites import REGISTRY
from spatio_flux.library.tools import get_standard_emitter

RUNTIME = 120.0
BOUNDS = (50.0, 50.0)
N_PARTICLES = 3            # a few initial particles
ADD_RATE = 0.3
COLLISION_R = 1.8          # domain-unit render radius (disks touch, no gaps)
GIF_FRAMES = 60
STRAIN_COLORS = {"ecoli_1": "#2563eb", "ecoli_2": "#dc2626"}  # blue / red


def _draw_pie(ax, x, y, r, sub):
    """Draw a particle as a pie of its sub_masses (a wedge per species) — the
    original spatio-flux particle look. Falls back to a grey disk if no submasses."""
    items = [(k, float(v)) for k, v in (sub or {}).items() if float(v or 0) > 0]
    tot = sum(v for _, v in items)
    if tot <= 0:
        ax.add_patch(Circle((x, y), r, facecolor="#9ca3af", edgecolor="#1e293b", linewidth=0.4))
        return
    a0 = 90.0
    for label, val in sorted(items):
        sweep = 360.0 * val / tot
        ax.add_patch(Wedge((x, y), r, a0, a0 + sweep,
                           facecolor=STRAIN_COLORS.get(label, "#9ca3af"),
                           edgecolor="#1e293b", linewidth=0.35))
        a0 += sweep


def _draw(ax, frame, glu, vmax):
    ax.clear()
    ax.imshow(glu, origin="lower", cmap="viridis", vmin=0, vmax=vmax,
              extent=[0, BOUNDS[0], 0, BOUNDS[1]], alpha=0.25, aspect="equal")
    parts = {k: v for k, v in frame.get("particles", {}).items()
             if not k.startswith("_") and isinstance(v, dict) and "position" in v}
    for p in parts.values():
        _draw_pie(ax, p["position"][0], p["position"][1], COLLISION_R, p.get("sub_masses"))
    ax.set_xlim(0, BOUNDS[0]); ax.set_ylim(0, BOUNDS[1])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    t = frame.get("global_time", 0.0)
    ax.set_title(f"t = {t:.0f} min   ·   {len(parts)} particles", fontsize=13)


def main() -> None:
    core = allocate_core()
    entry = next(e for e in REGISTRY.values() if e.name == "spatioflux_reference_demo")
    state = build_generator(entry, overrides={"n_particles": N_PARTICLES}, core=core)["state"]
    state["enforce_boundaries"]["config"].update(add_rate=ADD_RATE, boundary_to_add=["top"])
    state["newtonian_particles"]["config"].update(gravity=-9.81, jitter_per_second=1e-3,
                                                  damping_per_second=0.90)
    state["emitter"] = get_standard_emitter(state_keys=list(state.keys()), subsample=1)
    sim = Composite({"state": state}, core=core)
    sim.run(RUNTIME)
    res = [r for r in gather_emitter_results(sim)[("emitter",)] if isinstance(r.get("fields"), dict)]
    n = len(res)
    print(f"ran spatioflux_reference_demo — {n} frames")

    glu_max = max(np.array(r["fields"]["glucose"], dtype=float).max() for r in res) or 1.0
    viz = Path("studies/fig-08/visualizations"); viz.mkdir(parents=True, exist_ok=True)

    # GIF
    idxs = np.linspace(0, n - 1, GIF_FRAMES).astype(int)
    images = []
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for i in idxs:
        _draw(ax, res[i], np.array(res[i]["fields"]["glucose"], dtype=float), glu_max)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        buf.seek(0); images.append(imageio.imread(buf)); buf.close()
    plt.close(fig)
    gif = viz / "fig08-reference.gif"
    imageio.mimsave(gif, images, duration=0.1, loop=0)
    print("wrote", gif)

    # static montage (6 stills) so it can be viewed inline too
    stills = np.linspace(0, n - 1, 6).astype(int)
    fig, axes = plt.subplots(1, 6, figsize=(6 * 3.0, 3.0))
    for ax, i in zip(axes, stills):
        _draw(ax, res[i], np.array(res[i]["fields"]["glucose"], dtype=float), glu_max)
    fig.tight_layout()
    mont = viz / "fig08-reference-montage.png"
    fig.savefig(mont, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", mont)


if __name__ == "__main__":
    main()
