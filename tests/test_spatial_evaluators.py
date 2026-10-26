"""Derived-field spatial measure kinds: front_advances, centroid_shift, spread_grows.

These power the emergent-spatial-pattern studies (growth-front propagation,
advection plume, diffusion phase diagram). Each reads a 2D ('time','y','x')
field from the run zarr and reduces it to a derived spatial scalar series.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("xarray")  # modern-stack only; the locked unit-tests CI lacks it
pytest.importorskip("zarr")

import numpy as np

from spatio_flux.runners.run_store import write_run_zarr
from spatio_flux import evaluators as EV


def _write_field(ws, slug, frames):
    study_dir = os.path.join(ws, "studies", slug)
    os.makedirs(study_dir, exist_ok=True)
    traj = [{"global_time": float(i), "fields": {"dissolved biomass": f}} for i, f in enumerate(frames)]
    write_run_zarr(study_dir, f"{slug}-reproduce", traj,
                   provenance={"composite": "t", "run_id": f"{slug}-reproduce"})


def _reg():
    r = {}; EV.register_evaluators(r); return r


def test_front_advances_passes_when_front_moves_down(tmp_path):
    ws = str(tmp_path)
    # a biomass front that fills progressively more rows each tick
    frames = []
    for t in range(5):
        f = np.zeros((6, 4)); f[: t + 1, :] = 1.0; frames.append(f)
    _write_field(ws, "front", frames)
    out = _reg()["front_advances"]({"name": "F", "measure":
        {"kind": "front_advances", "slug": "front", "observable": "dissolved biomass", "threshold": 0.5},
        "pass_if": {"op": "front_advances", "tolerance": 0.0}}, None, ws)
    assert out["result"] == "PASS" and out["evaluated_by"] == "code"


def test_front_advances_fails_static_front(tmp_path):
    ws = str(tmp_path)
    f0 = np.zeros((6, 4)); f0[:2, :] = 1.0
    _write_field(ws, "static", [f0, f0.copy(), f0.copy()])
    out = _reg()["front_advances"]({"name": "F", "measure":
        {"kind": "front_advances", "slug": "static", "observable": "dissolved biomass", "threshold": 0.5},
        "pass_if": {"op": "front_advances", "tolerance": 0.0}}, None, ws)
    assert out["result"] == "FAIL"


def test_centroid_shift_detects_directed_migration(tmp_path):
    ws = str(tmp_path)
    # biomass blob moves down the y axis each tick
    frames = []
    for t in range(5):
        f = np.zeros((8, 4)); f[t, 1] = 1.0; frames.append(f)
    _write_field(ws, "mig", frames)
    out = _reg()["centroid_shift"]({"name": "C", "measure":
        {"kind": "centroid_shift", "slug": "mig", "observable": "dissolved biomass", "axis": "y", "min_shift": 2.0},
        "pass_if": {"op": "centroid_shift"}}, None, ws)
    assert out["result"] == "PASS"
    # a stationary blob -> FAIL
    f = np.zeros((8, 4)); f[0, 1] = 1.0
    _write_field(ws, "stay", [f, f.copy(), f.copy()])
    out2 = _reg()["centroid_shift"]({"name": "C", "measure":
        {"kind": "centroid_shift", "slug": "stay", "observable": "dissolved biomass", "axis": "y", "min_shift": 2.0},
        "pass_if": {"op": "centroid_shift"}}, None, ws)
    assert out2["result"] == "FAIL"


def test_spread_grows_passes_when_rg_increases(tmp_path):
    ws = str(tmp_path)
    # biomass spreading out from the center -> radius of gyration grows
    frames = []
    for t in range(4):
        f = np.zeros((9, 9)); c = 4
        f[c, c] = 1.0
        if t >= 1:
            f[c - t:c + t + 1, c - t:c + t + 1] = 1.0
        frames.append(f)
    _write_field(ws, "spread", frames)
    out = _reg()["spread_grows"]({"name": "S", "measure":
        {"kind": "spread_grows", "slug": "spread", "observable": "dissolved biomass"},
        "pass_if": {"op": "spread_grows", "tolerance": 0.0}}, None, ws)
    assert out["result"] == "PASS"
