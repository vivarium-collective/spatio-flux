"""Registered generators for the paper figures that need to be built LIVE
(not just exported as static specs), so the workbench can resolve their inner
composites for the bigraph-loom drill-down + preview.

Fig 1c is the study-workflow composite: its three parallel simulations are
`local:Composite` PROCESS nodes (see ``paper_figures._community_dfba_sim``), so
each one's live instance IS a ``Composite``. Serving Fig 1c as a *generator*
(rather than the static ``fig01c-study-workflow.composite.json``) lets
``/api/composite-inner-state`` instantiate it and drill into each simulation —
which is what renders the inner-composite preview thumbnails on the sim nodes.
"""
from __future__ import annotations

from pbg_superpowers.composite_generator import composite_generator


@composite_generator(
    name="fig01c-study-workflow",
    description=(
        "Fig 1c — study workflow: a draft Preprocess step configures three parallel "
        "community-dFBA simulations, each a drillable sub-composite (a local:Composite "
        "process wrapping the real community-dFBA model). An Emitter captures the "
        "simulations into an emitter-data store, a LoadResults step reads it into a "
        "results table, and draft Analyses / Visualizations / Tests steps consume it."
    ),
)
def fig01c_study_workflow(core=None):
    # Lazy import: this module is imported while `spatio_flux.composites` is
    # initialising, and paper_figures pulls the workspace core — importing it at
    # module top risks a partial-module circular import (breaks the exporter).
    from spatio_flux import paper_figures as pf
    return {"state": pf.fig1c_study_workflow_state()}
