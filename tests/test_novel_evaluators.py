"""Qualitative evaluators for the novel studies: spatial_overlap, demixing_index,
threshold_transition, gradient_correlation. Each must PASS the real signature and
FAIL the null (so a study can't regress to a monotone-knob check)."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

import numpy as np

from spatio_flux.runners.run_store import write_run_zarr
from spatio_flux import evaluators as EV


def _write(ws, slug, traj):
    sd = os.path.join(ws, "studies", slug)
    os.makedirs(sd, exist_ok=True)
    write_run_zarr(sd, f"{slug}-reproduce", traj, provenance={"composite": "t", "run_id": f"{slug}-reproduce"})


def _reg():
    r = {}; EV.register_evaluators(r); return r


def _run(kind, measure, pass_if, ws):
    t = {"name": kind, "measure": {"kind": kind, **measure}, "pass_if": pass_if}
    return _reg()[kind](t, None, ws)


# ---- spatial_overlap: B tracks acetate more than glucose (cross-feeding bridge) ----
def test_spatial_overlap_b_prefers_acetate(tmp_path):
    ws = str(tmp_path)
    ace = np.zeros((6, 6)); ace[:, 4:] = 1.0          # acetate on the right
    glc = np.zeros((6, 6)); glc[:, :2] = 1.0          # glucose on the left
    b = np.zeros((6, 6)); b[:, 4:] = 1.0              # B biomass co-located with acetate
    _write(ws, "xf", [{"fields": {"biomass_B": b, "acetate": ace, "glucose": glc}}])
    out = _run("spatial_overlap",
               {"slug": "xf", "observable": "biomass_B", "prefers": "acetate", "over": "glucose"},
               {"op": "spatial_overlap"}, ws)
    assert out["result"] == "PASS"
    # null: B co-located with glucose instead -> FAIL
    b2 = np.zeros((6, 6)); b2[:, :2] = 1.0
    _write(ws, "xf2", [{"fields": {"biomass_B": b2, "acetate": ace, "glucose": glc}}])
    out2 = _run("spatial_overlap",
                {"slug": "xf2", "observable": "biomass_B", "prefers": "acetate", "over": "glucose"},
                {"op": "spatial_overlap"}, ws)
    assert out2["result"] == "FAIL"


# ---- demixing_index: segregated strains PASS, interleaved FAIL ----
def _particles(coords_strains):
    return {f"p{i}": {"position": (x, y), "mass": 1.0, "strain": s}
            for i, (x, y, s) in enumerate(coords_strains)}


def test_demixing_index_segregated_vs_mixed(tmp_path):
    ws = str(tmp_path)
    # segregated: A on the left, B on the right (two clusters)
    seg = [(0, 0, "A"), (0.5, 0, "A"), (0, 0.5, "A"), (10, 10, "B"), (10.5, 10, "B"), (10, 10.5, "B")]
    _write(ws, "seg", [{"particles": _particles(seg)}, {"particles": _particles(seg)}])
    out = _run("demixing_index", {"slug": "seg", "k": 2}, {"op": "demixing_index", "min_purity": 0.8}, ws)
    assert out["result"] == "PASS"
    # mixed: A/B interleaved on a line
    mix = [(0, 0, "A"), (1, 0, "B"), (2, 0, "A"), (3, 0, "B"), (4, 0, "A"), (5, 0, "B")]
    _write(ws, "mix", [{"particles": _particles(mix)}, {"particles": _particles(mix)}])
    out2 = _run("demixing_index", {"slug": "mix", "k": 2}, {"op": "demixing_index", "min_purity": 0.8}, ws)
    assert out2["result"] == "FAIL"


# ---- threshold_transition: sharp cliff PASS, monotone FAIL ----
def test_threshold_transition_cliff_vs_monotone(tmp_path):
    ws = str(tmp_path)
    cliff = [5.0, 5.0, 5.0, 0.02, 0.01]   # sustained then collapse
    _write(ws, "cliff", [{"fields": {"m": v}} for v in cliff])
    out = _run("threshold_transition", {"slug": "cliff", "observable": "m"},
               {"op": "threshold_transition"}, ws)
    assert out["result"] == "PASS"
    grad = [5.0, 4.0, 3.0, 2.0, 1.0]      # gradual monotone -> no cliff
    _write(ws, "grad", [{"fields": {"m": v}} for v in grad])
    out2 = _run("threshold_transition", {"slug": "grad", "observable": "m"},
                {"op": "threshold_transition"}, ws)
    assert out2["result"] == "FAIL"


# ---- gradient_correlation: mass correlates with depth (y) PASS, flat FAIL ----
def test_gradient_correlation_mass_vs_depth(tmp_path):
    ws = str(tmp_path)
    corr = {f"p{i}": {"position": (0.0, float(i)), "mass": float(i) + 0.5} for i in range(6)}
    _write(ws, "corr", [{"particles": corr}])
    out = _run("gradient_correlation", {"slug": "corr", "axis": "y"},
               {"op": "gradient_correlation", "min_abs_corr": 0.7}, ws)
    assert out["result"] == "PASS"
    flat = {f"p{i}": {"position": (0.0, float(i)), "mass": 1.0} for i in range(6)}
    _write(ws, "flat", [{"particles": flat}])
    out2 = _run("gradient_correlation", {"slug": "flat", "axis": "y"},
                {"op": "gradient_correlation", "min_abs_corr": 0.7}, ws)
    assert out2["result"] == "FAIL"


def test_spatial_overlap_requires_margin(tmp_path):
    ws = str(tmp_path)
    import numpy as np
    a = np.ones((4, 4)); b = a + 1e-9  # B marginally more like 'a' than 'g' but within margin
    g = np.ones((4, 4))
    _write(ws, "tie", [{"fields": {"biomass_B": b, "acetate": a, "glucose": g}}])
    out = _run("spatial_overlap", {"slug": "tie", "observable": "biomass_B", "prefers": "acetate", "over": "glucose"},
               {"op": "spatial_overlap", "margin": 0.05}, ws)
    assert out["result"] == "FAIL"  # no real discriminator -> must not pass on a tie


def test_particle_settling_detects_gravity(tmp_path):
    ws = str(tmp_path)
    steps = [{"particles": {f"p{i}": {"position": (0.0, 10.0 - t * 2), "mass": 1.0} for i in range(3)}} for t in range(4)]
    _write(ws, "settle", steps)
    out = _run("particle_settling", {"slug": "settle"}, {"op": "particle_settling", "min_drop": 1.0}, ws)
    assert out["result"] == "PASS"
    flat = [{"particles": {"p0": {"position": (0.0, 5.0), "mass": 1.0}}} for _ in range(3)]
    _write(ws, "nofall", flat)
    out2 = _run("particle_settling", {"slug": "nofall"}, {"op": "particle_settling", "min_drop": 1.0}, ws)
    assert out2["result"] == "FAIL"
