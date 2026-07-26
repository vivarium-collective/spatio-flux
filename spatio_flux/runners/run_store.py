"""Write a run's trajectory to a per-study zarr store.

The workbench's SimulationsDB / run-history uses the zarr store format
``studies/<slug>/runs.<run_id>.zarr`` (the interchangeable XArray/zarr emitter
convention — NOT SQLite). The canonical ``pbg_emitters.XArrayEmitter`` is shaped
for v2ecoli's ``{time, agents:{id:{listeners}}}`` whole-cell state; spatio-flux's
state is ``{global_time, fields, particles}``, so we write an equivalent
self-describing zarr directly with xarray (same store format + provenance root
attrs the dashboard's ``_read_zarr_provenance`` reads back).
"""
from __future__ import annotations

import json
import os

import numpy as np


def zarr_store_path(study_dir: str, run_id: str) -> str:
    """Canonical per-study zarr path: studies/<slug>/runs.<run_id>.zarr."""
    return os.path.join(study_dir, f"runs.{run_id}.zarr")


def _field_dataarrays(trajectory):
    """Build {name: (dims, ndarray)} from per-tick field values.

    Scalar fields -> ('time',); 2D-array fields -> ('time','y','x'). Fields with
    inconsistent per-tick shapes are skipped (best-effort capture).
    """
    import xarray as xr  # local import: only the runner needs xarray

    if not trajectory:
        return {}, []
    times = [float(step.get("global_time", i)) for i, step in enumerate(trajectory)]
    field_names = set()
    for step in trajectory:
        field_names.update((step.get("fields") or {}).keys())

    data_vars = {}
    for name in sorted(field_names):
        series = [((step.get("fields") or {}).get(name)) for step in trajectory]
        if any(v is None for v in series):
            continue
        arr0 = np.asarray(series[0], dtype=float)
        try:
            stacked = np.stack([np.asarray(v, dtype=float) for v in series])
        except Exception:
            continue
        if arr0.ndim == 0:
            data_vars[name] = xr.DataArray(stacked, dims=("time",))
        elif arr0.ndim == 2:
            data_vars[name] = xr.DataArray(stacked, dims=("time", "y", "x"))
        # higher-dim fields skipped
    # particle count series (cheap, always encodable)
    if any("particles" in (step or {}) for step in trajectory):
        counts = [len((step.get("particles") or {})) for step in trajectory]
        data_vars["n_particles"] = xr.DataArray(np.asarray(counts), dims=("time",))
    return data_vars, times


def write_run_zarr(study_dir, run_id, trajectory, provenance):
    """Write the trajectory to studies/<slug>/runs.<run_id>.zarr; return the path."""
    import xarray as xr

    store = zarr_store_path(study_dir, run_id)
    data_vars, times = _field_dataarrays(trajectory)
    ds = xr.Dataset(
        data_vars,
        coords={"time": np.asarray(times)},
        attrs={
            "run_id": run_id,
            "composite": str(provenance.get("composite", "")),
            "provenance": json.dumps(provenance, default=str, sort_keys=True),
        },
    )
    # fresh store each run
    if os.path.isdir(store):
        import shutil
        shutil.rmtree(store)
    ds.to_zarr(store, mode="w")
    return store
