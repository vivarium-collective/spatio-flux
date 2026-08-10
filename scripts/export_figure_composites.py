#!/usr/bin/env python
"""Export the runnable spatio-flux figure composites (Fig 7.1–7.3 + Fig 8) as
fig-named, dashboard-discoverable specs so they're searchable as "fig…"
alongside the draft figures. Thin exports of the real generators (same state),
rendered/positioned/studied under one id per figure.

Run:  python scripts/export_figure_composites.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from process_bigraph import Composite, allocate_core
from process_bigraph.composite_generator import build_generator
from spatio_flux.composites import REGISTRY

OUT = Path(__file__).resolve().parents[1] / "spatio_flux" / "composites"

# generator name -> (spec stem, figure, description, overrides)
TARGETS = {
    "community_dfba": ("fig07-1-community-dfba", "Fig 7.1",
        "Hybrid microbial community: a shared `fields` store + a dynamic-FBA process per "
        "species (ecoli, ecoli core, cdiff, pputida, yeast, llactis) + Monod kinetics.", {}),
    "comets_diffusion": ("fig07-2-comets", "Fig 7.2",
        "COMETS-style spatial dFBA: fields {glucose, acetate, dissolved biomass} on a grid "
        "+ diffusion-advection + a dynamic-FBA process per grid bin.", {"n_bins": [4, 4]}),
    "brownian_particles": ("fig07-3-brownian-particles", "Fig 7.3",
        "Brownian particles: a `particles` store driven by brownian_movement with an "
        "enforce-boundaries step.", {}),
    "spatioflux_reference_demo": ("fig08-reference-model", "Fig 8",
        "spatio-flux reference model: fields + diffusion + a Monod-kinetics array with "
        "Newtonian particles carrying per-particle dFBA (ecoli_1, ecoli_2) aggregated into "
        "particle mass.",
        # Schematic grid: the model defaults to n_bins=[10,10] (100 kinetics cells).
        # A 4x4 grid depicts the same structure with far fewer nodes, matching the
        # COMETS figure. (Layout width is driven by the shared field stores, not the
        # grid count — a compact figure needs a hand-arranged view.)
        {"n_bins": [4, 4]}),
}


def _sanitize(node):
    import numpy as np
    if isinstance(node, np.ndarray):
        return node.tolist()
    if isinstance(node, np.generic):
        return node.item()
    if isinstance(node, dict):
        return {k: _sanitize(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_sanitize(v) for v in node]
    return node


def main() -> None:
    for gen_name, (stem, figure, desc, overrides) in TARGETS.items():
        core = allocate_core()
        try:
            entry = next(e for e in REGISTRY.values() if e.name == gen_name)
        except StopIteration:
            print(f"  SKIP {gen_name}: not in REGISTRY")
            continue
        try:
            doc = build_generator(entry, overrides=overrides, core=core)
            state = doc.get("state", doc) if isinstance(doc, dict) else doc
            schema = doc.get("schema") if isinstance(doc, dict) else None
            Composite({"state": state, **({"composition": schema} if schema else {})}, core=core)
            spec = {
                "name": stem,
                "description": f"{figure} — {desc}",
                "tags": ["paper-figure", figure.replace(" ", "-").lower()],
                "state": _sanitize(state),
            }
            (OUT / f"{stem}.composite.json").write_text(json.dumps(spec, indent=2, default=str))
            print(f"  OK  {gen_name} -> {stem}.composite.json  ({figure})")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {gen_name}: {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    main()
