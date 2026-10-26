"""Tests for the workspace's quantitative, code-evaluated measure kinds.

These evaluators read a run's zarr directly (the same direct-read pattern the
existing ``artifacts_present`` evaluator uses) and apply an op — so every new
behavior test is ``evaluated_by: code``. Fixtures build tiny real zarr stores via
the workspace run-store writer, then call the registered hooks exactly as
``viva_superpowers.study_evaluator`` would: ``hook(test, reader=None, ws_root)``.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("xarray")  # modern-stack only; the locked unit-tests CI lacks it
pytest.importorskip("zarr")

import numpy as np

from spatio_flux.runners.run_store import write_run_zarr
from spatio_flux import evaluators as EV


def _write(ws_root, slug, fields_by_tick, run_id=None, times=None):
    """Write studies/<slug>/runs.<run_id>.zarr from a list of per-tick field dicts."""
    run_id = run_id or f"{slug}-reproduce"
    study_dir = os.path.join(ws_root, "studies", slug)
    os.makedirs(study_dir, exist_ok=True)
    traj = []
    for i, fields in enumerate(fields_by_tick):
        t = times[i] if times is not None else float(i)
        traj.append({"global_time": t, "fields": fields})
    write_run_zarr(study_dir, run_id, traj, provenance={"composite": "test", "run_id": run_id})
    return run_id


def _registry():
    reg: dict = {}
    EV.register_evaluators(reg)
    return reg


def _run(kind, measure, pass_if, ws_root):
    reg = _registry()
    hook = reg[kind]
    test = {"name": f"T-{kind}", "measure": {"kind": kind, **measure}, "pass_if": pass_if}
    return hook(test, None, ws_root)


def test_observable_monotone_increasing_passes(tmp_path):
    ws = str(tmp_path)
    _write(ws, "mono_up", [{"biomass": v} for v in (0.1, 0.3, 0.6, 0.9)])
    out = _run("observable_monotone",
               {"slug": "mono_up", "observable": "biomass", "direction": "increasing"},
               {"op": "monotone", "tolerance": 0.0}, ws)
    assert out["result"] == "PASS"
    assert out["evaluated_by"] == "code"


def test_observable_monotone_flags_nonmonotone(tmp_path):
    ws = str(tmp_path)
    _write(ws, "mono_dip", [{"biomass": v} for v in (0.1, 0.6, 0.4, 0.9)])
    out = _run("observable_monotone",
               {"slug": "mono_dip", "observable": "biomass", "direction": "increasing"},
               {"op": "monotone", "tolerance": 0.0}, ws)
    assert out["result"] == "FAIL"


def test_observable_monotone_flat_fails_increasing(tmp_path):
    # a series that never accumulates must NOT pass "increasing" (net-change gate)
    ws = str(tmp_path)
    _write(ws, "flat_up", [{"biomass": v} for v in (5.0, 5.0, 5.0, 5.0)])
    out = _run("observable_monotone",
               {"slug": "flat_up", "observable": "biomass", "direction": "increasing"},
               {"op": "monotone", "tolerance": 0.01}, ws)
    assert out["result"] == "FAIL"


def test_observable_monotone_net_reversing_fails(tmp_path):
    # tiny within-tolerance steps that net DOWN must not pass "increasing"
    ws = str(tmp_path)
    _write(ws, "creepdown", [{"biomass": v} for v in (1.0, 0.99, 0.985, 0.98)])
    out = _run("observable_monotone",
               {"slug": "creepdown", "observable": "biomass", "direction": "increasing"},
               {"op": "monotone", "tolerance": 0.05}, ws)
    assert out["result"] == "FAIL"


def test_mass_conserved_within_tol(tmp_path):
    ws = str(tmp_path)
    # a 2D field whose TOTAL is conserved tick to tick (mass just moves around)
    a = np.array([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([[0.5, 0.5], [0.0, 0.0]])
    c = np.array([[0.25, 0.25], [0.25, 0.25]])
    _write(ws, "cons", [{"glucose": a}, {"glucose": b}, {"glucose": c}])
    out = _run("mass_conserved", {"slug": "cons", "observable": "glucose"},
               {"op": "mass_conserved", "tolerance": 1e-6}, ws)
    assert out["result"] == "PASS"
    # a leak beyond tol -> FAIL
    _write(ws, "leak", [{"glucose": a}, {"glucose": b * 0.5}])
    out2 = _run("mass_conserved", {"slug": "leak", "observable": "glucose"},
                {"op": "mass_conserved", "tolerance": 1e-6}, ws)
    assert out2["result"] == "FAIL"


def test_peak_then_decline_detects_overflow(tmp_path):
    ws = str(tmp_path)
    _write(ws, "ace", [{"acetate": v} for v in (0.0, 0.5, 1.2, 0.8, 0.2)])
    out = _run("peak_then_decline",
               {"slug": "ace", "observable": "acetate", "peak_index_band": [1, 3]},
               {"op": "peak_then_decline", "tolerance": 0.1}, ws)
    assert out["result"] == "PASS"
    # a monotone rise (no decline) -> FAIL
    _write(ws, "ace_rise", [{"acetate": v} for v in (0.0, 0.5, 1.2, 1.5, 1.8)])
    out2 = _run("peak_then_decline",
                {"slug": "ace_rise", "observable": "acetate", "peak_index_band": [1, 3]},
                {"op": "peak_then_decline", "tolerance": 0.1}, ws)
    assert out2["result"] == "FAIL"


def test_observable_band_final_value(tmp_path):
    ws = str(tmp_path)
    _write(ws, "band", [{"biomass": v} for v in (0.1, 0.5, 0.95)])
    ok = _run("observable_band", {"slug": "band", "observable": "biomass", "expected": [0.8, 1.1]},
              {"op": "band"}, ws)
    assert ok["result"] == "PASS"
    bad = _run("observable_band", {"slug": "band", "observable": "biomass", "expected": [0.0, 0.5]},
               {"op": "band"}, ws)
    assert bad["result"] == "FAIL"


def test_observable_flat_negative_control(tmp_path):
    ws = str(tmp_path)
    _write(ws, "starve", [{"biomass": v} for v in (0.1, 0.1001, 0.0999, 0.1)])
    out = _run("observable_flat", {"slug": "starve", "observable": "biomass"},
               {"op": "flat", "tolerance": 0.01}, ws)
    assert out["result"] == "PASS"
    _write(ws, "grew", [{"biomass": v} for v in (0.1, 0.4, 0.9)])
    out2 = _run("observable_flat", {"slug": "grew", "observable": "biomass"},
                {"op": "flat", "tolerance": 0.01}, ws)
    assert out2["result"] == "FAIL"


def test_missing_field_fails_cleanly(tmp_path):
    ws = str(tmp_path)
    _write(ws, "miss", [{"biomass": v} for v in (0.1, 0.5)])
    out = _run("observable_monotone",
               {"slug": "miss", "observable": "not_a_field", "direction": "increasing"},
               {"op": "monotone"}, ws)
    assert out["result"] == "FAIL"
    assert "not_a_field" in out["detail"]


def test_single_tick_trajectory_no_raise(tmp_path):
    ws = str(tmp_path)
    _write(ws, "one", [{"biomass": 0.1}])
    out = _run("observable_monotone",
               {"slug": "one", "observable": "biomass", "direction": "increasing"},
               {"op": "monotone"}, ws)
    assert out.get("result") == "FAIL" or out.get("evaluated_by") == "needs_rerun"


def test_series_total_sums_spatial_dims(tmp_path):
    ws = str(tmp_path)
    f0 = np.array([[1.0, 2.0], [3.0, 4.0]])  # total 10
    f1 = np.array([[2.0, 2.0], [3.0, 4.0]])  # total 11
    rid = _write(ws, "spat", [{"dissolved biomass": f0}, {"dissolved biomass": f1}])
    ds = EV.open_run(ws, "spat", rid)
    tot = EV.series_total(ds, "dissolved biomass")
    assert tot.shape == (2,)
    assert np.isclose(tot[0], 10.0) and np.isclose(tot[1], 11.0)
