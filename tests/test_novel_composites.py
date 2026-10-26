"""The novel composites build + run + show their phenomenon direction (real processes)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xarray")

from spatio_flux.core import build_core
from pbg_superpowers.composite_generator import build_generator, _REGISTRY
from spatio_flux.library.tools import run_composite_document


def _run(name, overrides, t):
    core = build_core()
    key = next(k for k in _REGISTRY if k.endswith("." + name))
    doc = build_generator(_REGISTRY[key], overrides=overrides, core=core)
    return run_composite_document(doc, core=core, time=t, outdir="/tmp/nvc")[0]


def _tot(res, field):
    return np.array([float(np.sum((s.get("fields") or {}).get(field, 0))) for s in res])


def test_spatial_cross_feeding_bridge_and_control():
    # with strain A: acetate is produced and strain B grows on it
    r = _run("spatial_cross_feeding", {"a_absent": False}, 40)
    bA, bB, ace = _tot(r, "biomass_A"), _tot(r, "biomass_B"), _tot(r, "acetate")
    assert bA[-1] > bA[0]            # A grows on glucose
    assert ace.max() > 0.1           # A secretes acetate (the bridge)
    assert bB[-1] >= bB[0]           # B does not shrink
    # A-absent control: no acetate, B does not grow
    r0 = _run("spatial_cross_feeding", {"a_absent": True}, 40)
    bB0, ace0 = _tot(r0, "biomass_B"), _tot(r0, "acetate")
    assert ace0.max() < 1e-6         # no acetate without A
    assert abs(bB0[-1] - bB0[0]) < 1e-3   # B flat without A's bridge
