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
    # Fig 2b (the process bigraph) IS the process graph — same generator as Fig 3a,
    # exported under its own fig02 name so the study links coherently by figure.
    ("fig02-process-bigraph", "Fig 2", pf.fig3a_process_graph_state, True,
     "Process bigraph (Fig 2b): the same metabolism + gene-expression process graph "
     "used in Fig 3a — a metabolism process and a gene-expression process over shared "
     "metab / enzymes / DNA stores — contrasted with a static Milner bigraph (Fig 2a)."),
    ("fig03a-process-graph", "Fig 3a", pf.fig3a_process_graph_state, True,
     "Process graph: a metabolism process (substrates + enzymes → products) and a "
     "gene-expression process (genes → enzymes) over shared metab / enzymes / DNA "
     "stores."),
    ("fig03b-composite-process", "Fig 3b", pf.fig3b_composite_process_state, True,
     "Composite process — the `cell`: cyto ⊃ {rib, nuc ⊃ DNA}, mem ⊃ chnl, with "
     "grow / express / transport processes and nutrient/signal inputs + shape output."),
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
    for _name in ("energy", "volume", "cell_count", "phase"):
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
