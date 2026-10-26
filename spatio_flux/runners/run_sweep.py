#!/usr/bin/env python
"""Parameter-sweep runner for the discovery-engine phase-diagram studies.

    python spatio_flux/runners/run_sweep.py <slug> [<steps>]

Reads ``studies/<slug>/study.yaml`` ``conditions.sweep``:

    conditions:
      sweep:
        composite: <registered composite ref>
        fixed: {param: value, ...}       # params held constant
        axis: {param: [v1, v2, ...]}      # the swept axis (one param)
        metric: {name: <id>, kind: n_particles|centroid_y, field: <field>}
        steps: <int>

For each axis value it builds + runs the composite, computes the scalar metric,
and writes the metric-vs-axis series to the study's zarr run-store (coord
``time`` = the axis values, data_var = the metric name) so the existing
``observable_monotone`` evaluator can assert the regime boundary. Also writes a
phase-diagram figure + a sweep_summary.json.
"""
import json
import os
import sys

import numpy as np
import yaml

from spatio_flux.core import build_core
from pbg_superpowers.composite_generator import build_generator, _REGISTRY
from spatio_flux.library.tools import run_composite_document
from spatio_flux.runners.run_store import write_run_zarr


def _metric(results, spec):
    kind = spec["kind"]
    if kind == "n_particles":
        return float(len(results[-1].get("particles") or {}))
    if kind == "centroid_y":
        a = np.asarray((results[-1].get("fields") or {}).get(spec["field"]), dtype=float)
        tot = a.sum()
        if tot <= 0:
            return 0.0
        ys = np.arange(a.shape[0])
        return float((a.sum(axis=1) * ys).sum() / tot)
    raise ValueError(f"unknown sweep metric kind: {kind}")


def main(slug, steps=None):
    spec = yaml.safe_load(open(os.path.join("studies", slug, "study.yaml")))
    sw = (spec.get("conditions") or {}).get("sweep") or {}
    comp, fixed, metric = sw["composite"], sw.get("fixed", {}), sw["metric"]
    steps = int(steps or sw.get("steps", 40))
    # Two forms: a simple scalar `axis: {param: [values]}`, or explicit `points`
    # (each {value: <scalar>, overrides: {...}}) for vector-valued params.
    if sw.get("points"):
        pname = sw.get("axis_label", "axis")
        cells = [(float(p["value"]), {**fixed, **p["overrides"]}) for p in sw["points"]]
    else:
        (pname, pvals), = sw["axis"].items()
        cells = [(float(v), {**fixed, pname: v}) for v in pvals]
    core = build_core()
    key = comp if comp in _REGISTRY else next(k for k in _REGISTRY if k.endswith("." + comp) or k == comp)
    charts = os.path.join("studies", slug, "charts")
    os.makedirs(charts, exist_ok=True)
    seeds = sw.get("seeds") or [0]  # stochastic composites: average the metric over seeds
    xs, ms = [], []
    for v, ov in cells:
        per_seed = []
        for sd in seeds:
            np.random.seed(int(sd))
            doc = build_generator(_REGISTRY[key], overrides=ov, core=core)
            results, _, _ = run_composite_document(doc, core=core, name=f"{slug}-{v}-s{sd}", time=steps, outdir=charts)
            per_seed.append(_metric(results, metric))
        xs.append(v)
        ms.append(float(np.mean(per_seed)))
        print(f"  {pname}={v} -> {metric['name']}={ms[-1]:.4g} (seeds {seeds}: {[round(x,2) for x in per_seed]})")
    mname = metric.get("name", "metric")
    traj = [{"global_time": x, "fields": {mname: m}} for x, m in zip(xs, ms)]
    store = write_run_zarr(os.path.join("studies", slug), f"{slug}-reproduce", traj,
                           provenance={"composite": comp, "sweep": sw})
    json.dump({"axis": pname, "values": xs, "metric": mname, "metric_values": ms},
              open(os.path.join(charts, f"{slug}_sweep.json"), "w"), indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(xs, ms, "o-", lw=2)
    ax.set_xlabel(pname)
    ax.set_ylabel(mname)
    ax.set_title(f"{slug}: {mname} vs {pname} (phase diagram)")
    fig.tight_layout()
    fig.savefig(os.path.join(charts, f"{slug}_phase.png"), dpi=110)
    plt.close(fig)
    print(f"✅ {slug}: swept {pname} -> zarr {store}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
