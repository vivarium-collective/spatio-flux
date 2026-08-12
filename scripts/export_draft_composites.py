#!/usr/bin/env python
"""Write the paper's DRAFT-process composites (Fig 1a, 1b, 2, 3a, 3b) as
dashboard-discoverable specs in spatio_flux/composites/.

The wired composites (1b, 3a, 3b) are validated by constructing a Composite
against the workspace core; Fig 1a is four unwired cards (no stores) so it is
authored directly.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from process_bigraph import Composite
from spatio_flux.core import build_core
from spatio_flux import paper_figures as pf

OUT = Path(__file__).resolve().parents[1] / "spatio_flux" / "composites"

SPECS = [
    ("fig01a-draft-processes", "Fig 1a", pf.fig1a_processes_state, False,
     "Four draft-process cards — Gene Expression (ODE), Metabolism (FBA), Morphogen "
     "Gradient (PDE), Multicellular Interactions (ABM). Unwired: ports + contracts "
     "only, no stores."),
    ("fig01b-multiscale-composite", "Fig 1b", pf.fig1b_multiscale_state, True,
     "The multiscale draft composite: tissue ⊃ {fields, cell_population, cells ⊃ "
     "cell ⊃ molecules}. Molecular ODEs, FBA metabolism, structural packing, "
     "growth/division, and tissue-scale diffusion + ABM — all draft processes."),
    ("fig01c-study-workflow", "Fig 1c", pf.fig1c_study_workflow_state, False,
     "A study workflow: a draft Preprocess step feeds three parallel community-dFBA "
     "simulations (real, zoomable composites — the same real study template ×3); "
     "their outputs feed a draft Analysis + Visualization step. Pre/post are draft; "
     "the parallel simulations are a real composite."),
    # Fig 2b: the paper's process-bigraph diagram — abstract nodes n1..n6 (place
    # graph) with processes p1,p2,p3 replacing the Milner hyperedges.
    ("fig02-process-bigraph", "Fig 2", pf.fig02_bigraph_state, True,
     "Process bigraph (Fig 2b): place-graph nodes n1..n6 with processes p1, p2, p3 "
     "connecting them through typed ports — the process-graph replacement for the "
     "Milner link graph's hyperedges (Fig 2a)."),
    ("fig03a-store", "Fig 3a", pf.fig3a_store_state, False,
     "Store diagram (a): a single store `cell` holding an `ecoli` data type, shown "
     "in full detail — a store holds any data type and shows its name + type."),
    ("fig03b-place-graph", "Fig 3b", pf.fig3b_place_graph_state, False,
     "Store diagram (b): a place graph of nested stores — cell ⊃ {nuc, cyto ⊃ ribo, "
     "mucin}, with EPS a sibling of cell (outer stores connect to inner stores)."),
    ("fig04-process", "Fig 4", pf.fig04_process_state, False,
     "Process diagram: one process as a rectangle with typed ports on its boundary — "
     "inputs in_1 (species), in_2 (params) left; outputs out_1 (ss_species), out_2 "
     "(rates) right; config type steady_state; update (in_1, in_2) → (out_1, out_2)."),
    # Fig 5a: a process graph — processes wired to shared stores via typed ports.
    ("fig05a-process-graph", "Fig 5a", pf.fig05a_process_graph_state, False,
     "Process graph (Fig 5a): metabolism + gene_expression processes connected to "
     "shared stores (metab, enzymes, DNA) through typed ports — arrow directions "
     "show the input/output wiring."),
    # Fig 5b: a composite process — a `cell` with a real drillable inner bigraph.
    ("fig05b-composite-process", "Fig 5b", pf.fig05b_composite_process_state, False,
     "Composite process (Fig 5b): a `cell` process exposing external ports "
     "(nutrients, signals in; shape out) that bridge to an internal process bigraph "
     "(express / grow / transport over cyto / mem / nuc) — a drillable sub-composite."),
]


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
    core = build_core()
    # pbg 1.8.3: the discovery-path registration of `{_inherit: "float"}` scalar
    # types can be left unresolved on the core; a DIRECT register_type resolves
    # them. Force-register so Composite validation finds them. (The dashboard
    # renders the written static spec verbatim, so this only affects validation.)
    for _name in ("energy", "volume", "cell_count", "phase", "place_node", "ecoli",
                  "species", "params", "ss_species", "rates", "steady_state"):
        try:
            core.register_type(_name, {"_inherit": "float"})
        except Exception:
            pass
    for stem, figure, builder, validate, desc in SPECS:
        state = builder()
        if validate:
            try:
                Composite({"state": state}, core=core)
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {stem}: {type(e).__name__}: {str(e)[:160]}")
                continue
        spec = {
            "name": stem,
            "description": f"{figure} — {desc}",
            "tags": ["paper-figure", "draft", figure.replace(" ", "-").lower()],
            "state": _sanitize(state),
        }
        (OUT / f"{stem}.composite.json").write_text(json.dumps(spec, indent=2, default=str))
        nproc = sum(1 for _ in _walk(state))
        print(f"  OK  {stem}  ({figure}, {nproc} process nodes)")


def _walk(node):
    if isinstance(node, dict):
        if node.get("_type") in ("process", "step"):
            yield node
        for v in node.values():
            yield from _walk(v)


if __name__ == "__main__":
    main()
