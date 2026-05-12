"""Normalize a process-bigraph doc into a deterministic JSON-encodable
shape so snapshot tests can compare structurally.

Decisions:
- numpy arrays serialize as ``{"__numpy__": true, "dtype", "shape", "values"}``
  with float values rounded to 12 significant figures. Round-trip stability
  across platforms is more important than full float64 precision; the
  snapshots are for *structural* regression, not numerical analysis.
- Unencodable custom objects fall back to a ``__repr__`` envelope so the
  snapshot still reflects their presence and class.
- Dicts, lists, tuples, and JSON scalars recurse / pass through.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

_FLOAT_SIG_FIGS = 12


def _round_float(x: float) -> float:
    """Round to 12 significant figures via decimal string formatting."""
    if not isinstance(x, float):
        x = float(x)
    if x == 0.0 or not (x == x and x not in (float("inf"), float("-inf"))):
        return x
    return float(f"{x:.{_FLOAT_SIG_FIGS}g}")


def normalize_doc(node: Any) -> Any:
    """Recursively normalize ``node`` into a JSON-comparable structure."""
    if isinstance(node, dict):
        return {k: normalize_doc(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [normalize_doc(v) for v in node]
    if isinstance(node, np.ndarray):
        return {
            "__numpy__": True,
            "dtype": str(node.dtype),
            "shape": list(node.shape),
            "values": _array_to_lists(node),
        }
    if isinstance(node, (np.integer,)):
        return int(node)
    if isinstance(node, (np.floating,)):
        return _round_float(float(node))
    if isinstance(node, float):
        return _round_float(node)
    # Pass through everything else that's JSON-encodable as-is.
    try:
        json.dumps(node)
        return node
    except TypeError:
        return {"__repr__": type(node).__name__, "value": repr(node)}


def _array_to_lists(arr: np.ndarray) -> list:
    """Convert an ndarray to nested lists, rounding floats."""
    if arr.dtype.kind == "f":
        return [_round_float(x) for x in arr.flatten().tolist()] if arr.ndim == 1 \
            else [_array_to_lists(sub) for sub in arr]
    return arr.tolist()
