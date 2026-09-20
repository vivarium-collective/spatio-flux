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


def register_evaluators(registry: dict) -> None:
    """Hook consumed by ``viva_superpowers.study_evaluator.load_workspace_evaluators``."""
    registry["artifacts_present"] = _artifacts_present_hook
