"""Draft-process composites for the Process Bigraph paper figures.

These are the *conceptual* figures (Fig 1 and Fig 3) — built from
``@draft_process`` placeholders: typed ports + a governing-equation contract,
but NO update dynamics yet. They render in bigraph-loom (topology + contracts)
and are badged DRAFT so they're never mistaken for the runnable spatio-flux
composites. The runnable figures (Fig 4–7) use spatio-flux's real processes.

Auto-discovery registers each ``DraftProcess`` subclass here at
``local:<ClassName>`` and calls this module's ``register_types`` — so the extra
biological/physical types the drafts use are available on every core.
"""
from __future__ import annotations

import json
from pathlib import Path

from process_bigraph import DraftProcess, draft_process


# Extra biological/physical scalar types (spatio-flux already provides
# concentration/mass/count). NOTE: process-bigraph 1.8.3 does not resolve a
# `_units` key on a scalar type (it works in 1.4.x); `{_inherit: "float"}` alone
# resolves, and the store's `_type` NAME still shows in the viewer. The
# place-graph compartments (molecules, cytoplasm, …) are kept as plain nested
# BRANCHES (record types also don't resolve here), so processes wire to leaves.
_EXTRA_TYPES = {
    "energy":     {"_inherit": "float"},
    "volume":     {"_inherit": "float"},
    "cell_count": {"_inherit": "float"},
    "phase":      {"_inherit": "float"},
    "place_node": {"_inherit": "float"},   # abstract bigraph node (Fig 2)
}


def register_types(core):
    """Register the draft figures' extra scalar types (idempotent). concentration
    and mass are provided by spatio-flux; register defensively if absent."""
    def _ensure(name, schema):
        try:
            if core.access(name) is None:
                core.register_type(name, schema)
        except Exception:
            try:
                core.register_type(name, schema)
            except Exception:
                pass
    _ensure("concentration", {"_inherit": "float"})
    _ensure("mass", {"_inherit": "float"})
    for name, schema in _EXTRA_TYPES.items():
        _ensure(name, schema)
    return core


# ── Fig 1b / multiscale draft processes (molecular → tissue) ─────────────────
@draft_process(
    name="Transcription",
    inputs={"dna": "concentration", "reg": "concentration"},
    outputs={"mrna": "concentration"},
    contract={
        "summary": "Gene expression — transcription (ODE)",
        "description": "RNA polymerase reads the DNA template to synthesize mRNA; the regulator gates the synthesis rate.",
        "math": [r"\frac{dr}{dt} = \alpha - \gamma_r\, r"],
        "symbols": {"r": "mRNA concentration", "α": "transcription rate", "γ_r": "mRNA degradation rate"},
        "ports": {"dna": "DNA template", "reg": "regulatory signal", "mrna": "transcribed mRNA"},
    },
)
class Transcription(DraftProcess):
    pass


@draft_process(
    name="Translation",
    inputs={"mrna": "concentration", "energy": "energy"},
    outputs={"protein": "concentration"},
    contract={
        "summary": "Gene expression — translation (ODE)",
        "description": "Ribosomes translate mRNA into protein, consuming energy cofactors (ATP / GTP).",
        "math": [r"\frac{dp}{dt} = \beta\, r - \gamma_p\, p"],
        "symbols": {"p": "protein concentration", "r": "mRNA concentration", "β": "translation rate", "γ_p": "protein degradation rate"},
        "ports": {"mrna": "mRNA template", "energy": "ATP / GTP energy", "protein": "translated protein"},
    },
)
class Translation(DraftProcess):
    pass


@draft_process(
    name="RNADegradation",
    inputs={"mrna": "concentration"}, outputs={"mrna": "concentration"},
    contract={
        "summary": "mRNA degradation — first-order turnover",
        "description": "Sets mRNA lifetime via first-order decay.",
        "math": [r"\frac{dr}{dt} = -\gamma_r\, r"],
        "symbols": {"r": "mRNA concentration", "γ_r": "mRNA degradation rate"},
        "ports": {"mrna": "mRNA (substrate and remaining pool)"},
    },
)
class RNADegradation(DraftProcess):
    pass


@draft_process(
    name="Metabolism",
    inputs={"enzymes": "concentration", "energy": "energy"},
    outputs={"metabolites": "concentration"},
    contract={
        "summary": "Metabolism — Flux Balance Analysis (FBA)",
        "description": "Constraint-based steady-state flux optimization: maximize a biomass / objective flux subject to mass balance and flux bounds. Enzyme levels and energy set the bounds.",
        "math": [r"\text{maximize}\quad Z = c^{\mathsf{T}} v", r"\text{s.t.}\quad S\,v = 0", r"v_{\min} \le v \le v_{\max}"],
        "symbols": {"Z": "objective (biomass flux)", "c": "objective coefficients", "v": "reaction flux vector", "S": "stoichiometric matrix", "v_min, v_max": "flux bounds"},
        "ports": {"enzymes": "enzyme (protein) levels — set flux bounds", "energy": "available energy — sets flux bounds", "metabolites": "produced metabolites"},
    },
)
class Metabolism(DraftProcess):
    pass


@draft_process(
    name="MolecularPacking",
    inputs={"molecules": "map[concentration]", "volume": "volume"},
    outputs={"structure": "mass"},
    contract={
        "summary": "Structural packing — cellPACK-style 3D packing (parsimony)",
        "description": "Places each molecular species at its true abundance into the cell geometry with an octree collision engine (the parsimony packer), filling to a target volume occupancy subject to non-overlap constraints.",
        "math": [r"\varphi = \frac{\sum_i n_i\, v_i}{V_{\text{cell}}}", r"\lVert x_i - x_j \rVert \ge r_i + r_j \quad \forall\, i \ne j"],
        "symbols": {"φ": "volume occupancy", "nᵢ": "copy number of species i", "vᵢ": "molecular volume", "V_cell": "cell volume", "xᵢ": "packed position", "rᵢ": "collision radius"},
        "ports": {"molecules": "molecular species (abundances)", "volume": "cell volume (geometry)", "structure": "packed 3D structure"},
    },
)
class MolecularPacking(DraftProcess):
    pass


@draft_process(
    name="Growth", inputs={"mass": "mass"}, outputs={"volume": "volume"},
    contract={
        "summary": "Cell growth — exponential",
        "description": "Biomass accumulation drives volume increase.",
        "math": [r"\frac{dV}{dt} = \mu\, V"],
        "symbols": {"V": "cell volume", "μ": "specific growth rate"},
        "ports": {"mass": "cell mass", "volume": "cell volume"},
    },
)
class Growth(DraftProcess):
    pass


@draft_process(
    name="Division", inputs={"volume": "volume", "phase": "phase"}, outputs={"phase": "phase"},
    contract={
        "summary": "Cell division — threshold-gated",
        "description": "Division triggers when volume and cell-cycle phase cross thresholds; volume halves and phase resets.",
        "math": [r"V \ge V_{\text{div}} \ \wedge\ \phi \ge \phi^\ast \;\Rightarrow\; V \to \tfrac{V}{2},\ \phi \to 0"],
        "symbols": {"V": "cell volume", "V_div": "division volume", "φ": "cell-cycle phase", "φ*": "phase threshold"},
        "ports": {"volume": "cell volume", "phase": "cell-cycle phase (reset on divide)"},
    },
)
class Division(DraftProcess):
    pass


@draft_process(
    name="Diffusion", inputs={"field": "concentration"}, outputs={"field": "concentration"},
    contract={
        "summary": "Morphogen gradient — reaction–diffusion PDE",
        "description": "A diffusing morphogen field with a local source and first-order decay sets up a spatial gradient across the tissue.",
        "math": [r"\frac{\partial C(x,t)}{\partial t} = D\,\frac{\partial^2 C}{\partial x^2} + S(x) - \lambda C"],
        "symbols": {"C": "morphogen concentration", "D": "diffusion coefficient", "S(x)": "local source", "λ": "decay rate"},
        "ports": {"field": "morphogen concentration field"},
    },
)
class Diffusion(DraftProcess):
    pass


@draft_process(
    name="ABM",
    inputs={"population": "cell_count", "field": "concentration"},
    outputs={"population": "cell_count"},
    contract={
        "summary": "Multicellular interactions — Agent-Based Model",
        "description": "Off-lattice agents move under interaction forces, chemotaxis up the morphogen gradient, and stochastic noise; contact-range interactions can remove cells.",
        "math": [r"\vec{x}_i(t{+}\Delta t) = \vec{x}_i + \mu\, f_{\text{int}} + \chi\, \nabla C + \eta\, \xi(t)", r"P_{\text{kill}} = \text{Prob(kill)} \cdot \mathbf{1}_{\lVert \vec{x}_i - \vec{x}_j \rVert < d}"],
        "symbols": {"x⃗ᵢ": "position of cell i", "μ": "mobility", "f_int": "interaction force", "χ": "chemotactic coefficient", "∇C": "morphogen gradient", "η·ξ(t)": "noise", "P_kill": "kill probability", "d": "interaction radius"},
        "ports": {"population": "cell population", "field": "local morphogen field"},
    },
)
class ABM(DraftProcess):
    pass


# ── Fig 1a: the four subsystem cards ─────────────────────────────────────────
# Metabolism (FBA), Diffusion (morphogen PDE) and ABM (multicellular) are reused
# above; gene expression as one aggregate ODE card is its own draft:
@draft_process(
    name="GeneExpression",
    inputs={"dna": "concentration", "energy": "energy"},
    outputs={"mrna": "concentration", "protein": "concentration"},
    contract={
        "summary": "Gene expression — ordinary differential equations",
        "description": "Transcription + translation as coupled ODEs: DNA templates mRNA; mRNA templates protein; each species turns over.",
        "math": [r"\frac{dr}{dt} = \alpha - \gamma_r\, r", r"\frac{dp}{dt} = \beta\, r - \gamma_p\, p"],
        "symbols": {"r": "mRNA", "p": "protein", "α": "transcription rate", "β": "translation rate", "γ_r": "mRNA decay", "γ_p": "protein decay"},
        "ports": {"dna": "DNA template", "energy": "ATP / GTP", "mrna": "mRNA", "protein": "protein"},
    },
)
class GeneExpression(DraftProcess):
    pass


# ── Fig 3a: process graph (metabolism + gene expression over shared stores) ──
@draft_process(
    name="MetabolismGraph",
    inputs={"substrates": "concentration", "enzymes": "concentration"},
    outputs={"products": "concentration"},
    contract={
        "summary": "Metabolism — substrates + enzymes → products",
        "description": "The process-graph view of metabolism: consumes substrates under enzyme catalysis to make products.",
        "math": [r"\frac{d[\text{products}]}{dt} = k_{\text{cat}}\,[\text{enzymes}]\,\frac{[\text{substrates}]}{K_m + [\text{substrates}]}"],
        "symbols": {"k_cat": "turnover number", "K_m": "Michaelis constant"},
        "ports": {"substrates": "substrate pool", "enzymes": "enzyme levels", "products": "product pool"},
    },
)
class MetabolismGraph(DraftProcess):
    pass


@draft_process(
    name="GeneExpressionGraph",
    inputs={"genes": "concentration"}, outputs={"enzymes": "concentration"},
    contract={
        "summary": "Gene expression — genes → enzymes",
        "description": "The process-graph view of gene expression: reads genes (DNA) and produces the enzymes that metabolism uses.",
        "math": [r"\frac{d[\text{enzymes}]}{dt} = k_{\text{expr}}\,[\text{genes}] - \gamma\,[\text{enzymes}]"],
        "symbols": {"k_expr": "expression rate", "γ": "enzyme turnover"},
        "ports": {"genes": "gene / DNA template", "enzymes": "produced enzymes"},
    },
)
class GeneExpressionGraph(DraftProcess):
    pass


# ── Fig 3b: composite process (the cell) ─────────────────────────────────────
@draft_process(
    name="Grow",
    inputs={"ribosomes": "concentration", "nutrients": "concentration", "signals": "concentration"},
    outputs={"membrane": "concentration"},
    contract={
        "summary": "Cell growth — ribosomes + nutrients + signals → membrane",
        "description": "Growth reads cytoplasmic ribosomes, external nutrients, and signals to expand the cell (membrane synthesis).",
        "math": [r"\frac{dm}{dt} = \mu\,[\text{rib}]\,\frac{[\text{nutrients}]}{K + [\text{nutrients}]}"],
        "symbols": {"m": "membrane / biomass", "μ": "growth rate", "[rib]": "ribosome level"},
        "ports": {"ribosomes": "cytoplasmic ribosomes", "nutrients": "external nutrients", "signals": "external signals", "membrane": "membrane channels / growth"},
    },
)
class Grow(DraftProcess):
    pass


@draft_process(
    name="Express",
    inputs={"genes": "concentration"}, outputs={"ribosomes": "concentration"},
    contract={
        "summary": "Gene expression → ribosomes",
        "description": "Reads the nuclear DNA and produces ribosomes into the cytoplasm.",
        "math": [r"\frac{d[\text{rib}]}{dt} = k\,[\text{DNA}] - \gamma\,[\text{rib}]"],
        "symbols": {"k": "expression rate", "γ": "ribosome turnover"},
        "ports": {"genes": "nuclear DNA", "ribosomes": "cytoplasmic ribosomes"},
    },
)
class Express(DraftProcess):
    pass


@draft_process(
    name="Transport",
    inputs={"channels": "concentration", "nutrients": "concentration"},
    outputs={"shape": "concentration"},
    contract={
        "summary": "Membrane transport → cell shape",
        "description": "Membrane channels import nutrients and set the cell's boundary shape.",
        "math": [r"J = P\,(c_{\text{out}} - c_{\text{in}})"],
        "symbols": {"J": "transport flux", "P": "channel permeability", "c": "concentration in/out"},
        "ports": {"channels": "membrane channels", "nutrients": "external nutrients", "shape": "cell shape (boundary output)"},
    },
)
class Transport(DraftProcess):
    pass


# ── helpers ──────────────────────────────────────────────────────────────────
@draft_process(
    name="BigraphLink",
    inputs={"in": "place_node"},
    outputs={"out": "place_node"},
    contract={
        "summary": "Process p — connects place-graph nodes via typed ports",
        "description": "A process in the process bigraph: it connects nodes of the "
                       "place graph through its typed ports, replacing a Milner "
                       "hyperedge in the link graph.",
        "status": "draft - no update",
        "ports": {"in": "a node this process reads", "out": "a node this process writes"},
    },
)
class BigraphLink(DraftProcess):
    pass


# ── Fig 1c: study-workflow steps (draft pre/post around real simulations) ─────
@draft_process(
    name="Preprocess",
    inputs={"raw": "concentration"},
    outputs={"conditions": "concentration"},
    contract={
        "summary": "Pre-processing — prepare the simulation conditions",
        "description": "The workflow's pre-step: reads the raw study inputs and "
                       "derives the initial conditions / parameters shared by the "
                       "parallel simulation ensemble.",
        "status": "draft - no update dynamics yet",
        "ports": {"raw": "raw study inputs", "conditions": "prepared simulation conditions"},
    },
)
class Preprocess(DraftProcess):
    pass


@draft_process(
    name="AnalysisViz",
    inputs={"runs": "concentration"},
    outputs={"figure": "concentration"},
    contract={
        "summary": "Analysis + visualization of the simulation ensemble",
        "description": "The workflow's post-step: aggregates the outputs of the "
                       "parallel simulations and renders the analysis + figure.",
        "status": "draft - no update dynamics yet",
        "ports": {"runs": "the parallel simulation outputs", "figure": "analysis + visualization"},
    },
)
class AnalysisViz(DraftProcess):
    pass


# ── Fig 4: the process schematic (typed ports + update function) ──────────────
@draft_process(
    name="ProcessSchematic",
    inputs={"in_1": "species", "in_2": "params"},
    outputs={"out_1": "ss_species", "out_2": "rates"},
    contract={
        # Punchy headline capturing the formal basis (not the notation — that's math).
        "summary": "A typed function signature whose update method emits a delta Δ",
        "description": (
            "Typed input ports (what it reads) and typed output ports (what it "
            "updates); the update method maps config + inputs to a delta Δ — the "
            "tree of changes to apply, branched by output port."),
        "status": "draft - no update dynamics yet",
        # Single-line contract: a typed process maps its typed inputs to a delta.
        "math": [
            r"p_{\text{proc}}[\text{config}{:}\tau_c]\ :\ "
            r"\text{in}_1^{\tau_1},\ \text{in}_2^{\tau_2}\ \longrightarrow\ \Delta",
        ],
        "ports": {"in_1": "species", "in_2": "params",
                  "out_1": "steady-state species", "out_2": "rates"},
    },
)
class ProcessSchematic(DraftProcess):
    pass


def _v(type_name: str, value: float) -> dict:
    return {"_type": type_name, "_value": float(value)}


def _proc(cls, inputs: dict, outputs: dict) -> dict:
    """Loom-friendly process node carrying the draft contract (math/symbols/port
    docs) + a DRAFT flag. Unwired ports are allowed (inputs/outputs may be {})."""
    status = str(cls.contract.get("status", ""))
    return {
        "_type": "process",
        "address": f"local:{cls.__name__}",
        "_draft": "draft" in status.lower(),
        "config": {"interval": 1.0, "summary": cls.contract.get("summary", ""), "contract": cls.contract},
        "_inputs": dict(cls.DRAFT_INPUTS),
        "_outputs": dict(cls.DRAFT_OUTPUTS),
        "_contract": {
            "summary": cls.contract.get("summary", ""),
            "description": cls.contract.get("description", ""),
            "status": status,
            "math": list(cls.contract.get("math", [])),
            "symbols": dict(cls.contract.get("symbols", {})),
            "inputs": {p: cls.contract.get("ports", {}).get(p, t) for p, t in cls.DRAFT_INPUTS.items()},
            "outputs": {p: cls.contract.get("ports", {}).get(p, t) for p, t in cls.DRAFT_OUTPUTS.items()},
        },
        "inputs": inputs,
        "outputs": outputs,
    }


# ── composite states ─────────────────────────────────────────────────────────
def fig1a_processes_state() -> dict:
    """Fig 1a: four unwired draft-process cards — no stores, no wiring."""
    return {
        "gene_expression": _proc(GeneExpression, {}, {}),
        "metabolism": _proc(Metabolism, {}, {}),
        "morphogen_gradient": _proc(Diffusion, {}, {}),
        "multicellular_interactions": _proc(ABM, {}, {}),
    }


def _cell() -> dict:
    return {
        # `molecules` is a plain nested BRANCH (place-graph container); processes
        # wire to its leaf species (record types don't resolve in pbg 1.8.3).
        "molecules": {"DNA": _v("concentration", 1.0), "mRNA": _v("concentration", 0.0),
                      "protein": _v("concentration", 0.0), "metabolites": _v("concentration", 0.0)},
        "reg_signals": _v("concentration", 1.0),
        "energy": _v("energy", 10.0),
        "structure": _v("mass", 0.0),
        "mass": _v("mass", 1.0), "volume": _v("volume", 1.0), "phase": _v("phase", 0.0),
        "transcription": _proc(Transcription, {"dna": ["molecules", "DNA"], "reg": ["reg_signals"]}, {"mrna": ["molecules", "mRNA"]}),
        "translation": _proc(Translation, {"mrna": ["molecules", "mRNA"], "energy": ["energy"]}, {"protein": ["molecules", "protein"]}),
        "rna_degradation": _proc(RNADegradation, {"mrna": ["molecules", "mRNA"]}, {"mrna": ["molecules", "mRNA"]}),
        "metabolism": _proc(Metabolism, {"enzymes": ["molecules", "protein"], "energy": ["energy"]}, {"metabolites": ["molecules", "metabolites"]}),
        "structural_packing": _proc(MolecularPacking, {"molecules": ["molecules"], "volume": ["volume"]}, {"structure": ["structure"]}),
        "growth": _proc(Growth, {"mass": ["mass"]}, {"volume": ["volume"]}),
        "division": _proc(Division, {"volume": ["volume"], "phase": ["phase"]}, {"phase": ["phase"]}),
    }


def fig1b_multiscale_state() -> dict:
    """Fig 1b: the multiscale draft composite (tissue ⊃ cells ⊃ cell ⊃ molecules)."""
    return {
        "tissue": {
            "fields": _v("concentration", 0.0),
            "cell_population": _v("cell_count", 1.0),
            "cells": {"cell": _cell()},
            "diffusion": _proc(Diffusion, {"field": ["fields"]}, {"field": ["fields"]}),
            "abm": _proc(ABM, {"population": ["cell_population"], "field": ["fields"]}, {"population": ["cell_population"]}),
        },
    }


def _community_dfba_sim() -> dict:
    """One parallel run as a genuine sub-composite: a `local:Composite` PROCESS
    whose inner document is the real community-dFBA composite (loaded from its
    committed spec, minus the top-level clock).

    Nesting it as a Composite process — rather than inlining the community-dFBA
    stores at the top level — is what makes each simulation a *drillable*
    sub-composite: its live instance IS a `Composite`, so the workbench flags it
    `is_composite_process` and the loom renders it as one collapsed node with an
    inner-composite preview + drill affordance (⤢). `is_composite_process` is set
    explicitly too so the STATIC export renders the same way without a live build."""
    spec = json.loads(
        (Path(__file__).resolve().parent / "composites"
         / "fig07-1-community-dfba.composite.json").read_text(encoding="utf-8"))
    inner = dict(spec.get("state") or {})
    inner.pop("global_time", None)
    return {
        "_type": "process",
        "address": "local:Composite",
        "is_composite_process": True,
        "config": {"state": inner},
        "inputs": {},
        "outputs": {},
    }


def fig1c_study_workflow_state() -> dict:
    """Fig 1c: a study workflow. A draft Preprocess step feeds three PARALLEL
    community-dFBA simulations (real, zoomable composites — the same real study
    template ×3), whose outputs feed a draft Analysis + Visualization step. The
    pre/post steps are draft; the parallel simulations are a real composite."""
    return {
        "raw_data": _v("concentration", 0.0),
        "preprocess": _proc(Preprocess, {"raw": ["raw_data"]}, {"conditions": ["simulations"]}),
        # Three parallel runs of the SAME real composite (an ensemble); each is a
        # full community-dFBA composite you can zoom into.
        "simulations": {
            "sim_0": _community_dfba_sim(),
            "sim_1": _community_dfba_sim(),
            "sim_2": _community_dfba_sim(),
        },
        "results": _v("concentration", 0.0),
        "analysis": _proc(AnalysisViz, {"runs": ["simulations"]}, {"figure": ["results"]}),
    }


def fig02_bigraph_state() -> dict:
    """Fig 2b: the process bigraph of the paper's composition-framework diagram.

    Place graph (solid nesting): n1 ⊃ {n3, n4}, n4 ⊃ {n6}, n2 ⊃ {n5}.
    Processes p1, p2, p3 replace the Milner link graph's hyperedges, connecting
    the nodes through their typed ports (dashed wires in the figure).
    """
    return {
        # Place graph: n1/n2/n4 are BRANCH nodes (contain children); n3/n5/n6 are leaves.
        "n1": {"n3": _v("place_node", 0.0), "n4": {"n6": _v("place_node", 0.0)}},
        "n2": {"n5": _v("place_node", 0.0)},
        # Processes wired across the place graph (paths into the nesting). p1 and
        # p3 also link to n2 (extra hyperedge spoke / process wire).
        "p1": _proc(BigraphLink, {"in": ["n1"]},          {"out": ["n1", "n3"], "out_b": ["n2"]}),
        "p2": _proc(BigraphLink, {"in": ["n1", "n3"]},    {"out": ["n1", "n4", "n6"]}),
        "p3": _proc(BigraphLink, {"in": ["n2", "n5"]},    {"out": ["n1", "n4", "n6"], "out_b": ["n2"]}),
    }


def fig3a_store_state() -> dict:
    """Fig 3a (store diagram, panel a): a SINGLE store shown in full detail, with
    its three parts labelled literally so the diagram is self-documenting — the
    store's NAME is ``name``, its VALUE is ``value``, and its TYPE is ``type``.
    (The loom reads a typed store's display value from ``_default``.)"""
    return {
        "name": {"_type": "type", "_default": "value"},
    }


def fig3b_place_graph_state() -> dict:
    """Fig 3b (store diagram, panel b): a PLACE GRAPH of nested stores — a cell
    and its compartments, each holding a biologically meaningful quantity with a
    DISTINCT data type, so the place graph reads as real cell biology:

        cell ⊃ { nucleus  ⊃ {chromatin: genome,   mRNA: transcript},
                 cytoplasm ⊃ {ribosome: count,     ATP:  energy},
                 membrane  ⊃ {receptor: count} }
        + medium (extracellular sibling): concentration

    Outer stores connect to inner stores by the place-graph nesting."""
    return {
        "cell": {
            "nucleus": {
                "chromatin": _v("genome", 0.0),
                "mRNA": _v("transcript", 0.0),
            },
            "cytoplasm": {
                "ribosome": _v("count", 0.0),
                "ATP": _v("energy", 0.0),
            },
            "membrane": {"receptor": _v("count", 0.0)},
        },
        "medium": _v("concentration", 0.0),  # extracellular sibling
    }


def fig04_process_state() -> dict:
    """Fig 4: a single process schematic — a rectangle with typed input ports
    (in_1: species, in_2: params) and output ports (out_1: ss_species, out_2:
    rates), config type steady_state, and update function (in_1,in_2)→(out_1,out_2)."""
    return {
        "process": _proc(ProcessSchematic, {}, {}),
    }
