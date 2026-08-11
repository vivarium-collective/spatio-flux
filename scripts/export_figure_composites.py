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
    "comets_spatial_dfba": ("fig07-2-comets", "Fig 7.2",
        "COMETS-style spatial dFBA: fields {glucose, acetate, dissolved biomass} on a grid "
        "+ advection-diffusion, driven by a SINGLE vectorized SpatialDFBA process that updates "
        "every lattice site as one batched structured state — so it scales to much larger grids "
        "than a dFBA process per bin.", {"n_bins": [20, 20]}),
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

def _class_contracts() -> dict:
    """Map ``local:<ClassName>`` -> the structured ``.contract`` declared ON the
    process class (single source of truth, see spatio_flux/processes/*.py). Read
    from every process class that advertises a contract — so ANY figure using a
    contract-bearing process picks it up automatically, and adding a contract to a
    new process needs no change here."""
    import importlib
    import inspect

    out: dict = {}
    # Scan the process package's submodules; guard optional-dep imports.
    import spatio_flux.processes as _procs
    modnames = getattr(_procs, "__all__", None) or [
        "dfba", "monod_kinetics", "diffusion_advection", "particles", "pymunk_particles",
    ]
    for modname in modnames:
        try:
            mod = importlib.import_module(f"spatio_flux.processes.{modname}")
        except Exception:
            continue
        for _name, cls in inspect.getmembers(mod, inspect.isclass):
            c = getattr(cls, "contract", None)
            if isinstance(c, dict) and c.get("summary"):
                out[f"local:{cls.__name__}"] = c
    return out


def _inject_contracts(state: dict, contracts: dict | None = None) -> dict:
    """Attach `_contract` (from the process class's `.contract`) to every
    process/step node, so the loom card shows the process's own governing
    equations. Recurses into nested stores (fields, particles, …)."""
    if contracts is None:
        contracts = _class_contracts()
    for v in state.values():
        if not isinstance(v, dict):
            continue
        if v.get("_type") in ("process", "step"):
            c = contracts.get(str(v.get("address", "")))
            if c and "_contract" not in v:
                v["_contract"] = {
                    "summary": c["summary"],
                    "description": c.get("description", c["summary"]),
                    "status": "",
                    "math": list(c.get("math", [])),
                    "symbols": dict(c.get("symbols", {})),
                    "inputs": {}, "outputs": {},
                }
        else:
            _inject_contracts(v, contracts)
    return state


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


def _resolve_defaults(node):
    """Unwrap `make_default(type, value)` → value. Composition schemas wrap every
    process field as `{"_type": t, "_default": v}`; the loom (and the fig07 exports)
    expect the resolved plain values (address="local:X", inputs={...}). A dict whose
    keys are EXACTLY {_type, _default} is such a wrapper; anything else (e.g. a
    process node `{_type: process, address, ...}`) is real state — recurse into it."""
    if isinstance(node, dict):
        if set(node.keys()) == {"_type", "_default"}:
            return _resolve_defaults(node["_default"])
        return {k: _resolve_defaults(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_defaults(v) for v in node]
    return node


def _materialize_map_compositions(state, schema):
    """Inject a composition schema's per-entry processes into the exported state so
    they render as explicit loom nodes.

    A composition like `get_community_dfba_particle_composition` declares
    `{"particles": {"_type": "map", "_value": {<per-particle processes>}}}` — the
    per-submass dFBA processes + aggregate_mass step live in the particle *type*
    (`_value`), not in the state tree, so a state-only snapshot drops them. Here we
    copy that `_value` composition into every entry of `state[store]` (resolving the
    make_default wrappers), which is exactly the structure each entry is given at
    runtime — keeping Fig 8a's schematic identical to the model that runs 8b."""
    import copy
    if not isinstance(schema, dict):
        return state
    for store, sch in schema.items():
        if not (isinstance(sch, dict) and sch.get("_type") == "map"):
            continue
        value_t = _resolve_defaults(sch.get("_value"))
        entries = state.get(store)
        if not (isinstance(value_t, dict) and isinstance(entries, dict)):
            continue
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            for key, node in value_t.items():
                if key not in entry:  # never clobber instance data (e.g. sub_masses values)
                    entry[key] = copy.deepcopy(node)
            _backfill_wire_targets(entry, value_t)
    return state


def _backfill_wire_targets(entry, processes):
    """Ensure sibling stores referenced by the injected processes exist, so the
    materialized dFBA/aggregate nodes don't render with dangling edges. A wire path
    like `["local"]` or `["exchange"]` names a store the composition would provide at
    runtime (empty substrate stores); add any missing single-hop target as `{}`."""
    def wire_heads(wires):
        heads = []
        if isinstance(wires, dict):
            for w in wires.values():
                heads += wire_heads(w)
        elif isinstance(wires, list) and wires and all(isinstance(x, str) for x in wires):
            heads.append(wires[0])
        return heads
    for node in processes.values():
        if not (isinstance(node, dict) and node.get("_type") in ("process", "step")):
            continue
        for head in wire_heads(node.get("inputs")) + wire_heads(node.get("outputs")):
            if head not in entry:
                entry[head] = {}


# Composites whose exported state is genuinely non-deterministic (random
# Newtonian-particle ids) — kept as a committed sample, not regenerated or
# drift-checked. Making these reproducible (a fixed-seed generator) is a follow-up.
STOCHASTIC = {"fig08-reference-model"}


def _rename_everywhere(obj, old, new):
    """Recursively rename a dict KEY / string VALUE `old` -> `new` (used to keep a
    renamed field consistent across the fields branch AND the process wiring)."""
    if isinstance(obj, dict):
        return {(new if k == old else k): _rename_everywhere(v, old, new) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):  # wiring paths arrive as tuples
        return [_rename_everywhere(x, old, new) for x in obj]
    if isinstance(obj, str):
        return new if obj == old else obj
    return obj  # numpy field arrays / numbers — never `== old`-compare (ambiguous)


def _figure_transforms(stem: str, state: dict) -> dict:
    """Figure-only PRESENTATION transforms, DECLARED here so a re-export is
    deterministic and can never clobber a hand-edit. These shape the figure's
    picture (a named field branch, typed fields, a clean particle) WITHOUT
    touching the real model the generator produces (which keeps its efficient
    map-of-arrays + stochastic particles). Idempotent + deterministic — `--check`
    verifies committed == fresh export."""
    if stem == "fig07-2-comets":
        # Rename "dissolved biomass" -> "biomass" EVERYWHERE (the field child AND
        # the process wiring that targets it) so the branch + wires stay consistent.
        state = _rename_everywhere(state, "dissolved biomass", "biomass")
        # fields: opaque map-of-arrays -> an explicit branch of named, typed child
        # stores so the loom shows the field structure (glucose/acetate/biomass).
        f = state.get("fields", {})
        shape = (f.get("_value") or {}).get("_shape", [20, 20]) if isinstance(f, dict) else [20, 20]
        arr = lambda: {"_type": "array", "_shape": list(shape), "_data": "concentration"}
        state["fields"] = {"glucose": arr(), "acetate": arr(), "biomass": arr()}
    elif stem == "fig07-1-community-dfba":
        # scalar field children -> typed `concentration` stores (not bare `number`).
        f = state.get("fields")
        if isinstance(f, dict):
            for k, v in list(f.items()):
                if not k.startswith("_") and isinstance(v, (int, float)):
                    f[k] = {"_type": "concentration", "_value": float(v)}
    elif stem == "fig07-3-brownian-particles":
        # the ensemble's ONE stochastic particle -> a fixed, clean representative
        # (deterministic id/position/mass) so the schematic + its content hash are
        # stable across exports.
        parts = state.get("particles")
        if isinstance(parts, dict):
            for k in [k for k in parts if not k.startswith("_")]:
                parts.pop(k)
            parts["particle"] = {"id": "particle", "position": [7.9, 38.8], "mass": 0.059}
    return state


def _build_spec(gen_name, stem, figure, desc, overrides) -> str | None:
    """Export one figure composite to its JSON string (or None on failure)."""
    core = allocate_core()
    try:
        entry = next(e for e in REGISTRY.values() if e.name == gen_name)
    except StopIteration:
        print(f"  SKIP {gen_name}: not in REGISTRY")
        return None
    try:
        doc = build_generator(entry, overrides=overrides, core=core)
        state = doc.get("state", doc) if isinstance(doc, dict) else doc
        schema = doc.get("schema") if isinstance(doc, dict) else None
        Composite({"state": state, **({"composition": schema} if schema else {})}, core=core)
        state = _materialize_map_compositions(state, schema)
        state = _inject_contracts(state)
        state = _figure_transforms(stem, state)
        spec = {
            "name": stem,
            "description": f"{figure} — {desc}",
            "tags": ["paper-figure", figure.replace(" ", "-").lower()],
            "state": _sanitize(state),
        }
        return json.dumps(spec, indent=2, default=str)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {gen_name}: {type(e).__name__}: {str(e)[:150]}")
        return None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Export the figure composites (or --check for drift).")
    ap.add_argument("--check", action="store_true",
                    help="verify each committed spec matches a fresh export; exit 1 on drift (CI gate)")
    args = ap.parse_args()

    drift = []
    for gen_name, (stem, figure, desc, overrides) in TARGETS.items():
        target = OUT / f"{stem}.composite.json"
        # Stochastic composites (random Newtonian-particle ids) can't be
        # byte-reproduced, so they're neither drift-checked nor regenerated over an
        # existing committed sample (that would churn the figure for no reason).
        if stem in STOCHASTIC:
            if args.check:
                print(f"  skip  {stem} (stochastic particles — not drift-checked)")
            elif target.exists():
                print(f"  keep  {stem} (stochastic — kept committed sample)")
            else:
                c = _build_spec(gen_name, stem, figure, desc, overrides)
                if c:
                    target.write_text(c)
                    print(f"  OK  {gen_name} -> {stem}.composite.json ({figure})")
            continue
        content = _build_spec(gen_name, stem, figure, desc, overrides)
        if content is None:
            if args.check:
                drift.append(stem)
            continue
        if args.check:
            cur = target.read_text() if target.exists() else ""
            if cur.rstrip() == content.rstrip():
                print(f"  ok    {stem}")
            else:
                drift.append(stem)
                print(f"  DRIFT {stem}: committed spec differs from a fresh export")
        else:
            target.write_text(content)
            print(f"  OK  {gen_name} -> {stem}.composite.json  ({figure})")
    if args.check and drift:
        raise SystemExit(f"[export --check] {len(drift)} composite(s) drifted: {', '.join(drift)} "
                         "— re-run `export_figure_composites.py` and commit.")


if __name__ == "__main__":
    main()
