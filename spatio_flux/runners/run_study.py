#!/usr/bin/env python
"""Bespoke study runner (canonical_runs entry). From the workspace root::

    python spatio_flux/runners/run_study.py <slug> [<time>]

Builds the study's baseline composite, runs it (writing the structure artifacts
``<slug>_viz.png`` / ``_schema.json`` / ``_state.json`` and returning the in-memory
trajectory), then runs the post-sim analysis flush into ``studies/<slug>/charts/``.

Faithful figures come from replaying the trajectory through the same plot.py
primitives the old test suite used (see spatio_flux.analysis.flush_spec).

NOTE: this does not yet persist a SQLite ``runs.db`` (dashboard run history). The
dashboard still surfaces the figures via each study's ``image:charts/*`` viz
entries, which read the files directly. runs.db persistence is a follow-up.
"""
import json
import os
import sys

import yaml

from spatio_flux.core import build_core
from pbg_superpowers.composite_generator import build_generator, _REGISTRY
from spatio_flux.analysis.flush import run_analysis_flush
from spatio_flux.analysis.flush_spec import FLUSH_SPEC, STANDARD_FIELD_COLORS
from spatio_flux.library.tools import run_composite_document
from spatio_flux.runners.run_store import write_run_zarr


def _run_id(slug):
    # Unique per study — list_simulations dedups rows by run_id.
    return f"{slug}-reproduce"


def _load_baseline(slug):
    """Read (composite ref, params) from studies/<slug>/study.yaml (authoritative)."""
    path = os.path.join("studies", slug, "study.yaml")
    if not os.path.exists(path):
        raise SystemExit(f"no study.yaml for slug: {slug}")
    with open(path) as f:
        spec = yaml.safe_load(f)
    baseline = spec["baseline"][0]
    return baseline["composite"], baseline.get("params", {})


def _first_config(state, *names):
    """Return the config dict of the first present process among ``names``."""
    for n in names:
        node = state.get(n)
        if isinstance(node, dict) and isinstance(node.get("config"), dict):
            return node["config"]
    return {}


def _resolve_runtime(state):
    """Compute the $-placeholder context from the composite state."""
    field_cfg = _first_config(state, "diffusion", "brownian_movement",
                              "particle_exchange", "newtonian_particles")
    bounds = field_cfg.get("bounds")
    n_bins = field_cfg.get("n_bins")
    ctx = {"$colors": STANDARD_FIELD_COLORS, "$bounds": bounds, "$n_bins": n_bins}
    if isinstance(state.get("fields"), dict):
        ctx["$all_fields"] = list(state["fields"].keys())
    # community dFBA biomass species ids (mirrors plot_community_dfba)
    ctx["$community_species"] = [
        entry["inputs"]["biomass"][-1]
        for entry in state.values()
        if isinstance(entry, dict) and isinstance(entry.get("inputs"), dict)
        and "biomass" in entry["inputs"]
    ]
    if n_bins:
        nx, ny = n_bins
        x0 = nx // 2
        ctx["$coordinates_corners"] = [[0, 0], [nx - 1, ny - 1]]
        ctx["$coordinates_comets"] = [[0, x0], [x0, x0], [nx - 1, x0]]
    nt = state.get("newtonian_particles")
    if isinstance(nt, dict):
        ctx["$pymunk_config"] = nt.get("config", {})
    sd = state.get("spatial_dFBA")
    if isinstance(sd, dict):
        ctx["$model_grid"] = sd.get("config", {}).get("model_grid")
    return ctx


def _apply(config, ctx):
    """Replace $-placeholder values in one step's config from the context."""
    out = {}
    for k, v in config.items():
        out[k] = ctx.get(v, v) if isinstance(v, str) and v.startswith("$") else v
    return out


def main(slug, runtime=None):
    if slug not in FLUSH_SPEC:
        raise SystemExit(f"unknown study slug: {slug}")
    ref, params = _load_baseline(slug)
    runtime = int(runtime) if runtime else 60

    core = build_core()
    doc = build_generator(_REGISTRY[ref], overrides=params, core=core)

    study_dir = os.path.join("studies", slug)
    charts = os.path.join(study_dir, "charts")
    os.makedirs(charts, exist_ok=True)

    # Run + write structure artifacts into charts/; keep the trajectory in memory.
    results, proc_t, fw_t = run_composite_document(
        doc, core=core, name=slug, time=runtime, outdir=charts,
        show_types=True, show_values=True)

    state = doc.get("state", doc)
    ctx = _resolve_runtime(state)
    specs = [{"step": s["step"], "config": _apply(s["config"], ctx)}
             for s in FLUSH_SPEC[slug]]
    written = run_analysis_flush(slug, results, state, charts, specs)

    # Persist the trajectory to the per-study zarr run-store so the run is
    # discoverable + tagged in the SimulationsDB (via the study.yaml runs: entry).
    run_id = _run_id(slug)
    store = write_run_zarr(study_dir, run_id, results,
                           provenance={"composite": ref, "config": params, "run_id": run_id})

    with open(os.path.join(study_dir, f"{slug}_timing.json"), "w") as f:
        json.dump({"process_time": proc_t, "framework_time": fw_t,
                   "elapsed": proc_t + fw_t}, f)
    print(f"✅ {slug}: wrote {len(written)} figures to {charts}; "
          f"zarr -> {store} ({proc_t + fw_t:.2f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
