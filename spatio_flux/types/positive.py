"""
spatio_flux.types.positive

Custom numeric types used by spatio_flux on top of bigraph-schema.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from bigraph_schema.schema import Array, Float, Number
from bigraph_schema.methods import apply, render, resolve


# ---------------------------------------------------------------------
# Type definitions (thin semantic wrappers)
# ---------------------------------------------------------------------

@dataclass(kw_only=True)
class SetFloat(Float):
    """A float that is replaced by its update (no accumulation)."""


@dataclass(kw_only=True)
class PositiveFloat(Float):
    """A float that accumulates updates and is clamped to be non-negative."""


@dataclass(kw_only=True)
class Concentration(PositiveFloat):
    """Non-negative accumulator representing an environmental concentration."""


@dataclass(kw_only=True)
class Mass(PositiveFloat):
    """Non-negative accumulator representing the mass of a species."""


@dataclass(kw_only=True)
class Count(Float):
    """accumulator representing a count"""


@dataclass(kw_only=True)
class PositiveArray(Array):
    """An array whose updates are accumulated and clamped elementwise to be non-negative."""

@dataclass(kw_only=True)
class Delta(Float):
    pass

# ---------------------------------------------------------------------
# Render methods: dataclass schema -> registry name
# ---------------------------------------------------------------------

@render.dispatch
def render(schema: PositiveFloat, defaults: bool = False):
    return "positive_float"


@render.dispatch
def render(schema: Mass, defaults: bool = False):
    return "mass"


@render.dispatch
def render(schema: PositiveArray, defaults: bool = False):
    return "positive_array"


@render.dispatch
def render(schema: Concentration, defaults: bool = False):
    return "concentration"


@render.dispatch
def render(schema: Count, defaults: bool = False):
    return "count"


@render.dispatch
def render(schema: SetFloat, defaults: bool = False):
    return "set_float"


# ---------------------------------------------------------------------
# Resolve methods: merge across numeric schema updates
# ---------------------------------------------------------------------

@resolve.dispatch
def resolve(current: Concentration, update: Concentration, path=()):
    # If current has a default and update doesn't, preserve current's default.
    if current._default and not update._default:
        return replace(update, _default=current._default)
    return update


@resolve.dispatch
def resolve(current: Float, update: Concentration, path=()):
    # Concentration can replace a generic Number schema; preserve defaults.
    if current._default and not update._default:
        return replace(update, _default=current._default)
    return update


@resolve.dispatch
def resolve(current: Concentration, update: Float, path=()):
    # If update is generic numeric but provides a default, keep it.
    if update._default and not current._default:
        return replace(current, _default=update._default)
    return current


# ---------------------------------------------------------------------
# Apply methods: state update semantics
# ---------------------------------------------------------------------

@apply.dispatch
def apply(schema: SetFloat, state, update, path):
    # Replacement semantics.
    return update, []


@apply.dispatch
def apply(schema: PositiveFloat, state, update, path):
    # Accumulate with non-negativity clamp.
    if update is None:
        return state, []
    return max(0, state + update), []


@apply.dispatch
def apply(schema: PositiveArray, current, update, path):
    """
    Apply an update to a PositiveArray.

    Supported update formats:
    - dense: update is array-like; applied elementwise and clamped at zero
    - sparse: update is a nested dict of indices -> delta values
        Example (2D): {i: {j: delta}}  => current[i, j] += delta, clamped at 0

    Returns:
        (new_array, [])
    """

    # Scalar fallback (rare): treat as PositiveFloat semantics
    if not isinstance(current, np.ndarray):
        if isinstance(update, dict):
            raise ValueError("Cannot apply dict update to scalar current value.")
        return np.maximum(0, current + update), []

    # Dense update
    if isinstance(update, np.ndarray):
        return np.maximum(0, current + update), []

    # Sparse update (nested dict)
    result = np.array(current, copy=True)

    def _apply_sparse(delta, idx=()):
        if isinstance(delta, dict):
            for k, v in delta.items():
                _apply_sparse(v, idx + (k,))
            return

        result[idx] = np.maximum(0, result[idx] + delta)

    _apply_sparse(update)
    return result, []


# ---------------------------------------------------------------------
# Convenience: registry name -> schema class mapping
# ---------------------------------------------------------------------

positive_types = {
    "positive_float": PositiveFloat,
    "positive_array": PositiveArray,
    "count": Count,
    "mass": Mass,
    "concentration": Concentration,
    "set_float": SetFloat,
    "delta_conc": Delta,
}
