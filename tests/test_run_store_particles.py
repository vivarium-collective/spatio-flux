"""Per-particle persistence in the run-store + the MSD evaluator.

The run-store previously persisted only an ``n_particles`` count. For a real
Brownian-MSD contract we need per-particle positions over time. Particle counts
change across ticks (division / seeding), so positions are stored as
``(time, particle)`` arrays NaN-padded to the max count, slots assigned by the
order a particle id first appears.
"""
from __future__ import annotations

import math
import os

import pytest

pytest.importorskip("xarray")  # modern-stack only; the locked unit-tests CI lacks it
pytest.importorskip("zarr")

import numpy as np
import xarray as xr

from spatio_flux.runners.run_store import write_run_zarr, zarr_store_path
from spatio_flux import evaluators as EV


def _write_particle_run(ws, slug, steps):
    study_dir = os.path.join(ws, "studies", slug)
    os.makedirs(study_dir, exist_ok=True)
    traj = [{"global_time": float(i), "particles": parts} for i, parts in enumerate(steps)]
    write_run_zarr(study_dir, f"{slug}-reproduce", traj,
                   provenance={"composite": "test", "run_id": f"{slug}-reproduce"})
    return xr.open_zarr(zarr_store_path(study_dir, f"{slug}-reproduce"))


def test_particles_persisted_padded(tmp_path):
    ws = str(tmp_path)
    steps = [
        {"A": {"position": (0.0, 0.0), "mass": 1.0}},
        {"A": {"position": (1.0, 1.0), "mass": 1.0}, "B": {"position": (5.0, 5.0), "mass": 2.0}},
        {"A": {"position": (2.0, 2.0), "mass": 1.0}, "B": {"position": (6.0, 6.0), "mass": 2.0}},
    ]
    ds = _write_particle_run(ws, "pp", steps)
    assert ds["particle_x"].dims == ("time", "particle")
    assert ds["particle_x"].shape == (3, 2)
    px = ds["particle_x"].values
    # slot 0 = A (present from t0), slot 1 = B (born at t1 -> NaN at t0)
    assert np.isnan(px[0, 1])
    assert px[0, 0] == 0.0 and px[1, 1] == 5.0
    assert ds["particle_mass"].values[2, 1] == 2.0


def test_msd_slope_linear_for_brownian(tmp_path):
    ws = str(tmp_path)
    # one particle whose squared displacement from t0 is exactly 2*t -> MSD slope 2
    steps = [{"P": {"position": (math.sqrt(2.0 * i), 0.0), "mass": 1.0}} for i in range(6)]
    ds = _write_particle_run(ws, "msd", steps)
    slope = EV.msd_slope(ds)
    assert abs(slope - 2.0) < 1e-6


def test_observable_msd_linear_hook(tmp_path):
    ws = str(tmp_path)
    steps = [{"P": {"position": (math.sqrt(2.0 * i), 0.0), "mass": 1.0}} for i in range(6)]
    _write_particle_run(ws, "msdh", steps)
    reg: dict = {}
    EV.register_evaluators(reg)
    test = {"name": "msd", "measure": {"kind": "observable_msd_linear", "slug": "msdh"},
            "pass_if": {"op": "msd_linear", "band": [1.5, 2.5], "min_r2": 0.95}}
    out = reg["observable_msd_linear"](test, None, ws)
    assert out["result"] == "PASS"
    assert out["evaluated_by"] == "code"


def test_particle_persistence_is_additive(tmp_path):
    """Adding particle arrays must not drop the existing n_particles count."""
    ws = str(tmp_path)
    steps = [{"A": {"position": (0.0, 0.0), "mass": 1.0}},
             {"A": {"position": (1.0, 0.0), "mass": 1.0}}]
    ds = _write_particle_run(ws, "add", steps)
    assert "n_particles" in ds.data_vars
    assert list(ds["n_particles"].values) == [1, 1]


def test_strain_particle_type_has_strain_and_simple_is_unchanged():
    # dedicated strain type carries the tag; the shared type is untouched so existing
    # particle studies' committed schemas don't drift.
    from spatio_flux import SIMPLE_PARTICLE_TYPE, STRAIN_PARTICLE_TYPE
    assert "strain" in STRAIN_PARTICLE_TYPE
    assert "strain" not in SIMPLE_PARTICLE_TYPE


def test_particle_strain_persisted(tmp_path):
    ws = str(tmp_path)
    steps = [
        {"A": {"position": (0.0, 0.0), "mass": 1.0, "strain": "red"}},
        {"A": {"position": (1.0, 1.0), "mass": 1.0, "strain": "red"},
         "B": {"position": (5.0, 5.0), "mass": 2.0, "strain": "blue"}},
    ]
    ds = _write_particle_run(ws, "ps_strain", steps)
    assert "particle_strain" in ds.data_vars
    assert ds["particle_strain"].dims == ("time", "particle")
    vals = ds["particle_strain"].values
    # slot 0 = A (red from t0), slot 1 = B (blue, appears at t1)
    assert str(vals[0, 0]) == "red"
    assert str(vals[1, 1]) == "blue"
