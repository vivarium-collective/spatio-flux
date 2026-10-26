#!/usr/bin/env python
"""Generic runner for the discovery-engine studies (no FLUSH_SPEC needed).

    python spatio_flux/runners/run_new_study.py <slug> [<time>]

Reads the study's baseline composite + params from ``studies/<slug>/study.yaml``
(``conditions.baseline``), runs it, persists the trajectory to the per-study zarr
run-store, and writes a focused figure of the observables under ``charts/``:

  * scalar fields     -> a trajectory plot (total vs time), one line per field;
  * 2D fields         -> a trajectory plot of each field's spatial total vs time
                         plus a first/last snapshot of the dominant biomass field;
  * particle runs     -> adds n_particles and total particle mass vs time.

Unlike ``run_study.py`` this does not replay the legacy FLUSH_SPEC recipes — the
new studies declare their own quantitative observables, evaluated by
``spatio_flux.evaluators`` against the zarr.
"""
import json
import os
import sys

import yaml

from spatio_flux.core import build_core
from pbg_superpowers.composite_generator import build_generator, _REGISTRY
from spatio_flux.library.tools import run_composite_document
from spatio_flux.runners.run_store import write_run_zarr


def _baseline(slug, which=None):
    spec = yaml.safe_load(open(os.path.join("studies", slug, "study.yaml")))
    cond = (spec.get("conditions") or {})
    base = cond.get("baseline")
    if isinstance(base, list):  # multiple baselines; pick by name or first
        base = next((b for b in base if b.get("name") == which), base[0]) if which else base[0]
    base = base or (spec.get("baseline") or [{}])[0]
    return base["composite"], base.get("params", {}), spec


def _figure(slug, results, charts, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    store = os.path.join("studies", slug, f"runs.{slug}{tag}-reproduce.zarr")
    ds = xr.open_zarr(store)
    t = np.asarray(ds["time"].values, dtype=float)
    scal, twod = [], []
    for v in ds.data_vars:
        dims = ds[v].dims
        if v in ("particle_x", "particle_y"):
            continue
        if dims == ("time",):
            scal.append(v)
        elif len(dims) == 3:
            twod.append(v)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for v in scal:
        ax.plot(t, np.asarray(ds[v].values, dtype=float), label=v)
    for v in twod:
        tot = np.asarray(ds[v].sum(dim=[d for d in ds[v].dims if d != "time"]).values, dtype=float)
        ax.plot(t, tot, label=f"Σ {v}")
    if "particle_mass" in ds.data_vars:
        pm = np.asarray(ds["particle_mass"].sum(dim="particle").values, dtype=float)
        ax.plot(t, pm, "--", label="Σ particle mass")
    ax.set_xlabel("time")
    ax.set_ylabel("quantity (field units / count)")
    ax.set_title(f"{slug}{tag}: observable trajectories")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = os.path.join(charts, f"{slug}{tag}_trajectory.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(slug, runtime=None, which=None, tag=""):
    import numpy as _np; _np.random.seed(0)  # deterministic runs so stochastic studies reproduce
    ref, params, _ = _baseline(slug, which)
    runtime = int(runtime) if runtime else 60
    core = build_core()
    key = ref if ref in _REGISTRY else next((k for k in _REGISTRY if k.endswith("." + ref) or k == ref), ref)
    doc = build_generator(_REGISTRY[key], overrides=params, core=core)
    study_dir = os.path.join("studies", slug)
    charts = os.path.join(study_dir, "charts")
    os.makedirs(charts, exist_ok=True)
    results, proc_t, fw_t = run_composite_document(
        doc, core=core, name=f"{slug}{tag}", time=runtime, outdir=charts,
        show_types=False, show_values=False)
    run_id = f"{slug}{tag}-reproduce"
    store = write_run_zarr(study_dir, run_id, results,
                           provenance={"composite": ref, "config": params, "run_id": run_id})
    fig = _figure(slug, results, charts, tag)
    # interactive Plotly viz (self-contained HTML; gitignored, regenerated at publish)
    try:
        import xarray as xr
        from spatio_flux.analysis.plotly_viz import field_heatmap_movie, particle_movie
        vizdir = os.path.join(study_dir, "viz")
        ds = xr.open_zarr(store)
        twod = [v for v in ds.data_vars if ds[v].ndim == 3]
        if twod:
            field_heatmap_movie(ds, twod[:3], os.path.join(vizdir, f"{slug}{tag}_fields.html"),
                                title=f"{slug}{tag}")
        if "particle_x" in ds.data_vars:
            cb = "strain" if "particle_strain" in ds.data_vars else "mass"
            particle_movie(ds, os.path.join(vizdir, f"{slug}{tag}_particles.html"),
                           color_by=cb, title=f"{slug}{tag}")
    except Exception as e:  # noqa: BLE001 — viz is best-effort
        print(f"plotly viz skipped for {slug}{tag}: {e}")
    print(f"✅ {slug}{tag}: zarr -> {store}; figure -> {fig} ({proc_t + fw_t:.2f}s)")


if __name__ == "__main__":
    args = sys.argv[1:]
    slug = args[0]
    runtime = args[1] if len(args) > 1 else None
    which = args[2] if len(args) > 2 else None
    tag = args[3] if len(args) > 3 else ""
    main(slug, runtime, which, tag)
