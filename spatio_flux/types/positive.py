"""
spatio_flux.types.positive

Custom numeric types used by spatio_flux on top of bigraph-schema.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, field

import numpy as np

from bigraph_schema.schema import Array, Float, Number, Tuple
from bigraph_schema.methods import apply, render, resolve


# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

@dataclass(kw_only=True)
class Position(Tuple):
    """Track a particle's position (x, y)."""
    _values: list = field(
        default_factory=lambda: [
            SetFloat(),  # x
            SetFloat(),  # y
        ]
    )

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


@dataclass(kw_only=True)
class ConcentrationDelta(Float):
    """A signed change in concentration (mM), emitted by a process to a
    Concentration store. Biologically a *concentration change*, not an absolute.

    Deliberately a plain ``Float`` subclass with NO custom ``resolve``/``apply``:
    the store (``concentration``) owns the accumulate-and-clamp semantics, and a
    render-only delta type resolves as cheaply as ``float``/``count``. Typing the
    output as ``concentration`` instead triggers the custom Concentration
    ``resolve`` dispatch per-particle-per-tick under division (embedded dFBA),
    which balloons ``access()``/deepcopy calls (~100x slowdown). This keeps the
    biology explicit in the loom while staying fast."""


@dataclass(kw_only=True)
class MassDelta(Float):
    """A signed change in mass (gDW), emitted to a Mass store. Render-only Float
    subclass — see :class:`ConcentrationDelta` for the performance rationale."""

# ---------------------------------------------------------------------
# Render methods: dataclass schema -> registry name
# ---------------------------------------------------------------------

@render.dispatch
def render(schema: PositiveFloat, defaults: bool = False):
    return "float"


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


@render.dispatch
def render(schema: ConcentrationDelta, defaults: bool = False):
    return "concentration_delta"


@render.dispatch
def render(schema: MassDelta, defaults: bool = False):
    return "mass_delta"


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


@resolve.dispatch
def resolve(current: Concentration, update: ConcentrationDelta, path=()):
    # A delta emitted into a Concentration store: the store type wins. Trivial,
    # allocation-free — no resolve_subclass recursion (delta and store are Float
    # siblings, which would otherwise fail to resolve).
    return current


@resolve.dispatch
def resolve(current: Mass, update: MassDelta, path=()):
    # A delta emitted into a Mass store: the store type wins.
    return current


@resolve.dispatch
def resolve(current: Count, update: ConcentrationDelta, path=()):
    # dFBA/Monod substrate deltas are polymorphic: they flow into a Count store
    # (particle exchange = molecule counts) as well as Concentration fields. The
    # store type wins in every case.
    return current


@resolve.dispatch
def resolve(current: Count, update: MassDelta, path=()):
    return current


# ---------------------------------------------------------------------
# Apply methods: state update semantics
# ---------------------------------------------------------------------

# @apply.dispatch
# def apply(schema: Position, state, update, path):
#     if state is None:
#         state = (0.0, 0.0)
#
#     dx, dy = update
#     x, y = state
#     x = float(x) + float(dx)
#     y = float(y) + float(dy)
#
#     # If schema has bounds like schema.bounds = (xmax, ymax)
#     if getattr(schema, "bounds", None) is not None:
#         xmax, ymax = schema.bounds
#         x = min(max(x, 0.0), float(xmax))
#         y = min(max(y, 0.0), float(ymax))
#
#     return (x, y), []


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
        (array, [])
    """

    # Scalar fallback (rare): treat as PositiveFloat semantics
    if not isinstance(current, np.ndarray):
        if isinstance(update, dict):
            raise ValueError("Cannot apply dict update to scalar current value.")
        return np.maximum(0, current + update), []

    # Dense update — clamp in place to avoid an extra allocation.
    if isinstance(update, np.ndarray):
        np.add(current, update, out=current)
        np.maximum(current, 0, out=current)
        return current, []

    # Sparse update (nested dict). Walk only the keys present in the
    # update — touching cells that received deltas, not the whole grid.
    def _apply_sparse(delta, idx=()):
        if isinstance(delta, dict):
            for k, v in delta.items():
                _apply_sparse(v, idx + (k,))
            return
        new_val = current[idx] + delta
        current[idx] = new_val if new_val > 0 else 0

    _apply_sparse(update)
    return current, []


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
    "concentration_delta": ConcentrationDelta,
    "mass_delta": MassDelta,
}
