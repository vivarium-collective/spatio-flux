"""Workspace evaluator hook for the spatio-flux-test-suite investigation.

Backs the ``artifacts_present`` measure kind (op ``all_exist_and_match``) that
every test-suite study declares. The framework
(``viva_superpowers.study_evaluator``) ships **no** evaluator for this kind, so
without this hook a study's gate resolves to the *agent bucket* — i.e. it is
never adjudicated by code. The studies were nonetheless scaffolded with a
hard-coded ``result: PASS``. This module makes the pass *earned*: it is computed.

``viva_superpowers.study_evaluator.evaluate_test`` consults
``<workspace_pkg>.evaluators.register_evaluators(registry)`` for any measure
kind that is not a native run-data kind, so registering ``artifacts_present``
here is all that is required for the dashboard to evaluate these gates by code.

What "reproduced" means here, stated honestly:

* every expected artifact must exist in ``studies/<slug>/charts/`` — the run
  actually produced its figures + structure files; AND
* the composite's structural schema (``<slug>_schema.json``) must match the
  committed reference in ``studies/<slug>/reference/`` within tolerance.

The schema is deterministic *up to* the ordering of ``|``-delimited port sets:
the emitter's ``_inputs`` port order varies run-to-run in field+particle
composites (unordered store iteration), which is semantically irrelevant, so
port specs are compared order-independently (see ``_port_set``). All other
structure carries no RNG (verified: identical across repeated runs).
The initial state (``<slug>_state.json``), the composite document
(``<slug>.json``) and the figures/gifs embed random particle ids/positions or
non-deterministic renders, so they are existence-checked, not value-matched.

The reference is baselined from the *current* canonical code. The old ``out0/``
snapshot is stale (it predates the current bigraph-schema representation) and
uncommitted, so it is **not** used as the reference.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

SCHEMA_SUFFIX = "_schema.json"


def _slug_from_expected(expected: list[str]) -> str | None:
    """Recover the study slug from the ``<slug>_schema.json`` expected entry."""
    for f in expected:
        if f.endswith(SCHEMA_SUFFIX):
            return f[: -len(SCHEMA_SUFFIX)]
    return None


def _numbers_match(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    fa, fb = float(a), float(b)
    if math.isnan(fa) and math.isnan(fb):
        return True
    diff = abs(fa - fb)
    scale = max(abs(fa), abs(fb), 1.0)
    return diff <= tol * scale


def _port_set(s: str):
    """A bigraph port spec (``a:t|b:t|...``) is an unordered *set* of ports, so
    ``|``-delimited strings compare order-independently. Returns the sorted token
    tuple for a port spec, else None (not a port spec)."""
    if isinstance(s, str) and "|" in s:
        return tuple(sorted(tok.strip() for tok in s.split("|")))
    return None


def _deep_match(a: Any, b: Any, tol: float, path: str = "") -> list[str]:
    """Structural comparison. Returns a list of mismatch descriptions (empty == match)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: absent in produced")
            elif k not in b:
                out.append(f"{path}/{k}: absent in reference")
            else:
                out += _deep_match(a[k], b[k], tol, f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path or '/'}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _deep_match(x, y, tol, f"{path}[{i}]")
        return out
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return [] if _numbers_match(a, b, tol) else [f"{path or '/'}: {a} != {b}"]
    pa, pb = _port_set(a), _port_set(b)
    if pa is not None and pb is not None:
        return [] if pa == pb else [f"{path or '/'}: ports {a!r} != {b!r}"]
    return [] if a == b else [f"{path or '/'}: {a!r} != {b!r}"]


def evaluate_artifacts_present(expected, charts_dir, reference_dir, tolerance=0.0):
    """Pure core: all expected artifacts exist + structural schema matches reference.

    Returns ``(result, detail)`` where ``result`` is ``"PASS"`` or ``"FAIL"``.
    """
    n = len(expected)
    missing = [f for f in expected if not os.path.isfile(os.path.join(charts_dir, f))]
    if missing:
        return "FAIL", f"missing {len(missing)}/{n} artifact(s): {', '.join(missing)}"

    schema = next((f for f in expected if f.endswith(SCHEMA_SUFFIX)), None)
    if schema is None:
        return "PASS", f"{n}/{n} artifacts present (no schema artifact to structurally match)"

    ref_path = os.path.join(reference_dir, schema)
    if not os.path.isfile(ref_path):
        return "PASS", (f"{n}/{n} artifacts present; no committed structural reference "
                        f"({schema}) — existence-only")
    with open(os.path.join(charts_dir, schema)) as fh:
        produced = json.load(fh)
    with open(ref_path) as fh:
        reference = json.load(fh)
    mismatches = _deep_match(produced, reference, tolerance)
    if mismatches:
        head = "; ".join(mismatches[:3])
        more = f" (+{len(mismatches) - 3} more)" if len(mismatches) > 3 else ""
        return "FAIL", f"schema drift vs reference: {head}{more}"
    return "PASS", (f"{n}/{n} artifacts present; {schema} matches reference "
                    f"(structural, tol={tolerance})")


def _artifacts_present_hook(test: dict, reader: Any, ws_root: Any) -> dict:
    """Framework evaluator: signature ``(test, reader, ws_root) -> outcome dict``."""
    measure = test.get("measure") or {}
    expected = list(measure.get("expected") or [])
    pass_if = test.get("pass_if") or {}
    tol = float(pass_if.get("tolerance", 0.0) or 0.0)
    slug = _slug_from_expected(expected)
    if slug is None or ws_root is None:
        return {"evaluated_by": "agent",
                "reason": "artifacts_present: cannot resolve study slug / ws_root"}
    sdir = os.path.join(str(ws_root), "studies", slug)
    result, detail = evaluate_artifacts_present(
        expected,
        os.path.join(sdir, "charts"),
        os.path.join(sdir, "reference"),
        tol,
    )
    return {
        "result": result,
        "measured_value": f"{len(expected)} artifacts",
        "evaluated_by": "code",
        "operator": "artifacts_present/all_exist_and_match",
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Quantitative, code-evaluated measure kinds (read the run's zarr directly).
#
# Each run is persisted at studies/<slug>/runs.<run_id>.zarr by
# spatio_flux.runners.run_store.write_run_zarr: scalar fields as ('time',),
# 2D fields as ('time','y','x'), plus an 'n_particles' ('time',) count. These
# evaluators open that store directly (same direct-read approach as
# artifacts_present) and apply an op, so every test is evaluated_by: code.
#
# Measure contract: measure={kind, slug, observable, [run_id], [direction],
# [peak_index_band], [expected]}, pass_if={op, [tolerance], [band]}.
# ---------------------------------------------------------------------------

def open_run(ws_root: Any, slug: str, run_id: str | None = None):
    """Open studies/<slug>/runs.<run_id>.zarr (run_id defaults to <slug>-reproduce)."""
    import xarray as xr
    from spatio_flux.runners.run_store import zarr_store_path

    run_id = run_id or f"{slug}-reproduce"
    store = zarr_store_path(os.path.join(str(ws_root), "studies", slug), run_id)
    return xr.open_zarr(store)


def series_total(ds, field: str) -> np.ndarray:
    """Per-tick total of a field: sum over spatial dims if 2D, else the scalar series."""
    if field not in ds.data_vars:
        raise KeyError(field)
    da = ds[field]
    spatial = [d for d in da.dims if d != "time"]
    if spatial:
        da = da.sum(dim=spatial)
    return np.asarray(da.values, dtype=float)


def front_position(ds, field: str, threshold: float) -> np.ndarray:
    """Per-tick front position along y: the largest row index with any cell >= threshold."""
    arr = np.asarray(ds[field].values, dtype=float)  # (time, y, x)
    out = []
    for t in range(arr.shape[0]):
        rows = np.where(np.any(arr[t] >= threshold, axis=1))[0]
        out.append(float(rows.max()) if rows.size else 0.0)
    return np.asarray(out)


def front_velocity(ds, field: str, threshold: float) -> float:
    """Least-squares slope of front_position vs the time coordinate."""
    y = front_position(ds, field, threshold)
    t = np.asarray(ds["time"].values, dtype=float)
    if len(t) < 2:
        return 0.0
    return float(np.polyfit(t, y, 1)[0])


def centroid(ds, field: str) -> np.ndarray:
    """Mass-weighted (y, x) centroid of a 2D field per tick; NaN where the field is empty."""
    arr = np.asarray(ds[field].values, dtype=float)  # (time, y, x)
    T = arr.shape[0]
    ys = np.arange(arr.shape[1])
    xs = np.arange(arr.shape[2])
    out = np.full((T, 2), np.nan)
    for t in range(T):
        m = arr[t]
        s = m.sum()
        if s <= 0:
            continue
        out[t, 0] = (m.sum(axis=1) * ys).sum() / s
        out[t, 1] = (m.sum(axis=0) * xs).sum() / s
    return out


def radius_of_gyration(ds, field: str) -> np.ndarray:
    """Mass-weighted RMS distance from the centroid per tick (spatial spread)."""
    arr = np.asarray(ds[field].values, dtype=float)
    c = centroid(ds, field)
    T = arr.shape[0]
    ys = np.arange(arr.shape[1])
    xs = np.arange(arr.shape[2])
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    out = np.full(T, np.nan)
    for t in range(T):
        m = arr[t]
        s = m.sum()
        if s <= 0:
            continue
        r2 = (YY - c[t, 0]) ** 2 + (XX - c[t, 1]) ** 2
        out[t] = math.sqrt((m * r2).sum() / s)
    return out


def delta_scalar(ds, field: str) -> float:
    """Final minus initial of the field's per-tick total."""
    v = series_total(ds, field)
    return float(v[-1] - v[0])


def _out(result: str, detail: str, operator: str, measured: Any) -> dict:
    return {"result": result, "measured_value": measured, "evaluated_by": "code",
            "operator": operator, "detail": detail}


def _open_series(test: dict, ws_root: Any):
    """Resolve (ds, field) from the test; return (ds, field, None) or (None, None, outcome)."""
    m = test.get("measure") or {}
    slug, field = m.get("slug"), m.get("observable")
    if not slug or ws_root is None:
        return None, None, {"evaluated_by": "agent",
                            "reason": "quantitative evaluator: measure.slug / ws_root missing"}
    try:
        ds = open_run(ws_root, slug, m.get("run_id"))
    except Exception as e:  # noqa: BLE001 — surface as a clean FAIL, never raise
        return None, None, _out("FAIL", f"cannot open run for {slug!r}: {e}",
                                f"{m.get('kind')}/open", None)
    if field not in ds.data_vars:
        return ds, None, _out("FAIL",
                              f"observable {field!r} not in run (have {list(ds.data_vars)})",
                              f"{m.get('kind')}/resolve", None)
    return ds, field, None


def _monotone_hook(test, reader, ws_root):
    m, pi = test.get("measure") or {}, test.get("pass_if") or {}
    tol = float(pi.get("tolerance", 0.0) or 0.0)
    direction = m.get("direction", "increasing")
    op = f"observable_monotone/{direction}"
    ds, field, err = _open_series(test, ws_root)
    if err:
        return err
    vals = series_total(ds, field)
    if len(vals) < 2:
        return _out("FAIL", "insufficient ticks (<2) to establish a trend", op, len(vals))
    diffs = np.diff(vals)
    scale = max(float(np.max(np.abs(vals))), 1.0)
    # non-reversing within tolerance AND a real net change in the claimed
    # direction (so a flat or net-reversing series cannot pass "grows"/"depletes").
    if direction == "increasing":
        ok = bool(np.all(diffs >= -tol * scale)) and vals[-1] > vals[0]
    else:
        ok = bool(np.all(diffs <= tol * scale)) and vals[-1] < vals[0]
    detail = (f"{direction}; steps [{float(diffs.min()):.4g},{float(diffs.max()):.4g}]; "
              f"net {float(vals[-1] - vals[0]):+.4g}; start {vals[0]:.4g} end {vals[-1]:.4g}; tol {tol}")
    return _out("PASS" if ok else "FAIL", detail, op, float(vals[-1]))


def _band_hook(test, reader, ws_root):
    m, pi = test.get("measure") or {}, test.get("pass_if") or {}
    band = m.get("expected") or pi.get("band") or [None, None]
    if band[0] is None or band[1] is None:
        return _out("FAIL", "no acceptance band declared (measure.expected / pass_if.band)",
                    "observable_band", None)
    lo, hi = float(band[0]), float(band[1])
    ds, field, err = _open_series(test, ws_root)
    if err:
        return err
    vals = series_total(ds, field)
    if len(vals) < 1:
        return _out("FAIL", "empty run", "observable_band", None)
    final = float(vals[-1])
    ok = lo <= final <= hi
    return _out("PASS" if ok else "FAIL",
                f"final {final:.4g} vs band [{lo:.4g},{hi:.4g}]", "observable_band", final)


def _mass_conserved_hook(test, reader, ws_root):
    m, pi = test.get("measure") or {}, test.get("pass_if") or {}
    tol = float(pi.get("tolerance", 1e-6) or 1e-6)
    ds, field, err = _open_series(test, ws_root)
    if err:
        return err
    vals = series_total(ds, field)
    if len(vals) < 2:
        return _out("FAIL", "insufficient ticks (<2)", "mass_conserved", len(vals))
    base = vals[0]
    scale = max(abs(base), 1.0)
    dev = float(np.max(np.abs(vals - base)))
    ok = dev <= tol * scale
    return _out("PASS" if ok else "FAIL",
                f"max deviation {dev:.4g} vs tol {tol * scale:.4g} (base {base:.4g})",
                "mass_conserved", dev)


def _peak_then_decline_hook(test, reader, ws_root):
    m, pi = test.get("measure") or {}, test.get("pass_if") or {}
    tol = float(pi.get("tolerance", 0.1) or 0.1)
    band = m.get("peak_index_band") or pi.get("peak_index_band")
    ds, field, err = _open_series(test, ws_root)
    if err:
        return err
    vals = series_total(ds, field)
    if len(vals) < 3:
        return _out("FAIL", "insufficient ticks (<3) for a peak", "peak_then_decline", len(vals))
    pk = int(np.argmax(vals))
    peak, end = float(vals[pk]), float(vals[-1])
    interior = 0 < pk < len(vals) - 1
    in_band = (band[0] <= pk <= band[1]) if band else True
    rose = peak > vals[0]
    declined = end < (1.0 - tol) * peak
    ok = interior and in_band and rose and declined
    return _out("PASS" if ok else "FAIL",
                f"peak {peak:.4g} at idx {pk} (band {band}); end {end:.4g}; "
                f"interior={interior} rose={rose} declined={declined}",
                "peak_then_decline", pk)


def _first_finite(col: np.ndarray) -> float:
    finite = np.where(np.isfinite(col))[0]
    return float(col[finite[0]]) if finite.size else float("nan")


def mean_squared_displacement(ds):
    """(times, MSD) where MSD(t) = mean over particles of |pos(t) − pos(t0_particle)|²,
    each particle referenced to its own first finite position; NaN slots ignored."""
    if "particle_x" not in ds.data_vars:
        raise KeyError("particle_x")
    x = np.asarray(ds["particle_x"].values, dtype=float)  # (time, particle)
    y = np.asarray(ds["particle_y"].values, dtype=float)
    t = np.asarray(ds["time"].values, dtype=float)
    n = x.shape[1]
    x0 = np.array([_first_finite(x[:, j]) for j in range(n)])
    y0 = np.array([_first_finite(y[:, j]) for j in range(n)])
    msd = np.full(x.shape[0], np.nan)
    for ti in range(x.shape[0]):
        d2 = (x[ti] - x0) ** 2 + (y[ti] - y0) ** 2
        vals = d2[np.isfinite(d2)]
        if vals.size:
            msd[ti] = float(np.mean(vals))
    return t, msd


def msd_slope(ds) -> float:
    """Least-squares slope of mean-squared-displacement vs time (diffusive scaling)."""
    t, msd = mean_squared_displacement(ds)
    ok = np.isfinite(msd)
    if ok.sum() < 2:
        return 0.0
    return float(np.polyfit(t[ok], msd[ok], 1)[0])


def _msd_linear_hook(test, reader, ws_root):
    m, pi = test.get("measure") or {}, test.get("pass_if") or {}
    band = m.get("expected") or pi.get("band") or [None, None]
    min_r2 = float(pi.get("min_r2", 0.0) or 0.0)
    slug = m.get("slug")
    if not slug or ws_root is None:
        return {"evaluated_by": "agent", "reason": "observable_msd_linear: measure.slug/ws_root missing"}
    try:
        ds = open_run(ws_root, slug, m.get("run_id"))
    except Exception as e:  # noqa: BLE001
        return _out("FAIL", f"cannot open run for {slug!r}: {e}", "observable_msd_linear/open", None)
    if "particle_x" not in ds.data_vars:
        return _out("FAIL", "run has no per-particle positions", "observable_msd_linear/resolve", None)
    t, msd = mean_squared_displacement(ds)
    ok = np.isfinite(msd)
    if ok.sum() < 2:
        return _out("FAIL", "insufficient MSD points (<2)", "observable_msd_linear", int(ok.sum()))
    slope, intercept = np.polyfit(t[ok], msd[ok], 1)
    pred = slope * t[ok] + intercept
    ss_res = float(np.sum((msd[ok] - pred) ** 2))
    ss_tot = float(np.sum((msd[ok] - np.mean(msd[ok])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    lo, hi = band
    in_band = (lo is None or slope >= lo) and (hi is None or slope <= hi)
    res = "PASS" if (in_band and r2 >= min_r2) else "FAIL"
    return _out(res, f"MSD slope {slope:.4g} (band {band}); R² {r2:.4g} (min {min_r2})",
                "observable_msd_linear", float(slope))


def _flat_hook(test, reader, ws_root):
    m, pi = test.get("measure") or {}, test.get("pass_if") or {}
    tol = float(pi.get("tolerance", 0.01) or 0.01)  # absolute amplitude tolerance
    ds, field, err = _open_series(test, ws_root)
    if err:
        return err
    vals = series_total(ds, field)
    if len(vals) < 2:
        return _out("FAIL", "insufficient ticks (<2)", "observable_flat", len(vals))
    amp = float(np.max(vals) - np.min(vals))
    ok = amp <= tol
    return _out("PASS" if ok else "FAIL",
                f"amplitude {amp:.4g} vs tol {tol} (start {vals[0]:.4g} end {vals[-1]:.4g})",
                "observable_flat", amp)


def _front_advances_hook(test, reader, ws_root):
    m, pi = test.get("measure") or {}, test.get("pass_if") or {}
    tol = float(pi.get("tolerance", 0.0) or 0.0)
    ds, field, err = _open_series(test, ws_root)
    if err:
        return err
    arr = np.asarray(ds[field].values, dtype=float)
    if arr.ndim != 3:
        return _out("FAIL", f"{field!r} is not a 2D ('time','y','x') field", "front_advances", None)
    thr = m.get("threshold")
    thr = float(thr) if thr is not None else 0.5 * float(np.nanmax(arr))
    fp = front_position(ds, field, thr)
    if len(fp) < 2:
        return _out("FAIL", "insufficient ticks", "front_advances", len(fp))
    diffs = np.diff(fp)
    scale = max(float(np.max(np.abs(fp))), 1.0)
    ok = bool(np.all(diffs >= -tol * scale)) and fp[-1] > fp[0]
    return _out("PASS" if ok else "FAIL",
                f"front {fp[0]:.3g}->{fp[-1]:.3g} (thr {thr:.3g}); non-decreasing within tol {tol}",
                "front_advances", float(fp[-1]))


def _centroid_shift_hook(test, reader, ws_root):
    m, pi = test.get("measure") or {}, test.get("pass_if") or {}
    axis = 0 if (m.get("axis", "y") == "y") else 1
    min_shift = float(m.get("min_shift", pi.get("min_shift", 1.0)) or 1.0)
    ds, field, err = _open_series(test, ws_root)
    if err:
        return err
    if np.asarray(ds[field].values).ndim != 3:
        return _out("FAIL", f"{field!r} is not a 2D field", "centroid_shift", None)
    c = centroid(ds, field)[:, axis]
    c = c[np.isfinite(c)]
    if len(c) < 2:
        return _out("FAIL", "insufficient centroid points", "centroid_shift", len(c))
    shift = abs(float(c[-1] - c[0]))
    ok = shift >= min_shift
    return _out("PASS" if ok else "FAIL",
                f"centroid {m.get('axis','y')} shift {shift:.3g} vs min {min_shift}",
                "centroid_shift", shift)


def _spread_grows_hook(test, reader, ws_root):
    m, pi = test.get("measure") or {}, test.get("pass_if") or {}
    tol = float(pi.get("tolerance", 0.0) or 0.0)
    ds, field, err = _open_series(test, ws_root)
    if err:
        return err
    if np.asarray(ds[field].values).ndim != 3:
        return _out("FAIL", f"{field!r} is not a 2D field", "spread_grows", None)
    rg = radius_of_gyration(ds, field)
    ok_mask = np.isfinite(rg)
    rg = rg[ok_mask]
    if len(rg) < 2:
        return _out("FAIL", "insufficient Rg points", "spread_grows", len(rg))
    diffs = np.diff(rg)
    scale = max(float(np.max(np.abs(rg))), 1.0)
    ok = bool(np.all(diffs >= -tol * scale)) and rg[-1] > rg[0]
    return _out("PASS" if ok else "FAIL",
                f"radius of gyration {rg[0]:.3g}->{rg[-1]:.3g} (grows within tol {tol})",
                "spread_grows", float(rg[-1]))


def register_evaluators(registry: dict) -> None:
    """Hook consumed by ``viva_superpowers.study_evaluator.load_workspace_evaluators``."""
    registry["artifacts_present"] = _artifacts_present_hook
    registry["front_advances"] = _front_advances_hook
    registry["centroid_shift"] = _centroid_shift_hook
    registry["spread_grows"] = _spread_grows_hook
    registry["observable_band"] = _band_hook
    registry["observable_monotone"] = _monotone_hook
    registry["mass_conserved"] = _mass_conserved_hook
    registry["peak_then_decline"] = _peak_then_decline_hook
    registry["observable_flat"] = _flat_hook
    registry["observable_msd_linear"] = _msd_linear_hook
    registry["spatial_overlap"] = _spatial_overlap_hook
    registry["demixing_index"] = _demixing_index_hook
    registry["threshold_transition"] = _threshold_transition_hook
    registry["gradient_correlation"] = _gradient_correlation_hook


# ---------------------------------------------------------------------------
# Qualitative evaluators for the novel studies (emergent signatures, not knobs).
# ---------------------------------------------------------------------------

def _final_2d(ds, name):
    if name not in ds.data_vars:
        return None
    a = np.asarray(ds[name].values, dtype=float)
    return a[-1] if a.ndim == 3 else None


def _corr(x, y):
    xf, yf = np.ravel(x).astype(float), np.ravel(y).astype(float)
    if xf.std() == 0 or yf.std() == 0:
        return 0.0
    return float(np.corrcoef(xf, yf)[0, 1])


def _spatial_overlap_hook(test, reader, ws_root):
    m = test.get("measure") or {}
    slug, obs = m.get("slug"), m.get("observable")
    prefers, over = m.get("prefers"), m.get("over")
    if not slug or ws_root is None:
        return {"evaluated_by": "agent", "reason": "spatial_overlap: measure.slug/ws_root missing"}
    try:
        ds = open_run(ws_root, slug, m.get("run_id"))
    except Exception as e:  # noqa: BLE001
        return _out("FAIL", f"cannot open run: {e}", "spatial_overlap/open", None)
    B, A, G = _final_2d(ds, obs), _final_2d(ds, prefers), _final_2d(ds, over)
    if B is None or A is None or G is None:
        return _out("FAIL", f"need three 2D fields {obs!r},{prefers!r},{over!r}", "spatial_overlap", None)
    ca, cg = _corr(B, A), _corr(B, G)
    return _out("PASS" if ca > cg else "FAIL",
                f"corr({obs},{prefers})={ca:.3g} > corr({obs},{over})={cg:.3g} ? ", "spatial_overlap", ca - cg)


def _demixing_index_hook(test, reader, ws_root):
    m, pi = test.get("measure") or {}, test.get("pass_if") or {}
    slug, k, min_p = m.get("slug"), int(m.get("k", 6)), float(pi.get("min_purity", 0.7))
    if not slug or ws_root is None:
        return {"evaluated_by": "agent", "reason": "demixing_index: measure.slug/ws_root missing"}
    try:
        ds = open_run(ws_root, slug, m.get("run_id"))
    except Exception as e:  # noqa: BLE001
        return _out("FAIL", f"cannot open run: {e}", "demixing_index/open", None)
    if "particle_strain" not in ds.data_vars:
        return _out("FAIL", "run has no particle_strain", "demixing_index/resolve", None)
    x = np.asarray(ds["particle_x"].values[-1], dtype=float)
    y = np.asarray(ds["particle_y"].values[-1], dtype=float)
    s = ds["particle_strain"].values[-1].astype(str)
    mask = np.isfinite(x) & np.isfinite(y) & (s != "")
    xs, ys, ss = x[mask], y[mask], s[mask]
    if len(xs) < 2 or len(set(ss)) < 2:
        return _out("FAIL", "insufficient strain-tagged particles / single strain", "demixing_index", len(xs))
    pts = np.column_stack([xs, ys])
    pur = []
    for i in range(len(xs)):
        d = np.sum((pts - pts[i]) ** 2, axis=1)
        nbrs = np.argsort(d)[1:k + 1]
        if len(nbrs):
            pur.append(float(np.mean(ss[nbrs] == ss[i])))
    avg = float(np.mean(pur)) if pur else 0.0
    return _out("PASS" if avg >= min_p else "FAIL",
                f"mean local strain purity {avg:.3g} vs min {min_p} (demixed when high)", "demixing_index", avg)


def _threshold_transition_hook(test, reader, ws_root):
    m = test.get("measure") or {}
    ds, field, err = _open_series(test, ws_root)
    if err:
        return err
    vals = series_total(ds, field)
    if len(vals) < 3:
        return _out("FAIL", "insufficient points (<3) for a transition", "threshold_transition", len(vals))
    diffs = np.abs(np.diff(vals))
    maxd = float(diffs.max())
    med = float(np.median(diffs)) if len(diffs) else 0.0
    rng = float(vals.max())
    dominant = maxd > 3 * max(med, 1e-9) and maxd > 0.5 * max(rng, 1e-9)
    collapsed = vals[-1] < 0.25 * max(rng, 1e-9)
    ok = bool(dominant and collapsed)
    return _out("PASS" if ok else "FAIL",
                f"max step {maxd:.3g} vs median {med:.3g}; end {vals[-1]:.3g} vs peak {rng:.3g}; "
                f"dominant={dominant} collapsed={collapsed}", "threshold_transition", maxd)


def _gradient_correlation_hook(test, reader, ws_root):
    m, pi = test.get("measure") or {}, test.get("pass_if") or {}
    slug = m.get("slug")
    axis = "particle_y" if m.get("axis", "y") == "y" else "particle_x"
    min_c = float(pi.get("min_abs_corr", 0.5))
    if not slug or ws_root is None:
        return {"evaluated_by": "agent", "reason": "gradient_correlation: measure.slug/ws_root missing"}
    try:
        ds = open_run(ws_root, slug, m.get("run_id"))
    except Exception as e:  # noqa: BLE001
        return _out("FAIL", f"cannot open run: {e}", "gradient_correlation/open", None)
    if "particle_mass" not in ds.data_vars or axis not in ds.data_vars:
        return _out("FAIL", "run lacks particle_mass / positions", "gradient_correlation/resolve", None)
    mass = np.asarray(ds["particle_mass"].values[-1], dtype=float)
    pos = np.asarray(ds[axis].values[-1], dtype=float)
    mask = np.isfinite(mass) & np.isfinite(pos)
    mm, pp = mass[mask], pos[mask]
    if len(mm) < 3 or mm.std() == 0 or pp.std() == 0:
        return _out("FAIL", "no gradient (flat mass/position or too few particles)", "gradient_correlation", 0.0)
    c = abs(_corr(pp, mm))
    return _out("PASS" if c >= min_c else "FAIL",
                f"|corr(mass, {m.get('axis','y')})| = {c:.3g} vs min {min_c}", "gradient_correlation", c)
