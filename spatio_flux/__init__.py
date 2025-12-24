"""
spatio_flux.types_registry

Central place for registering spatio-flux type definitions into a bigraph-schema core.

Why this exists
---------------
spatio-flux relies on a shared vocabulary of types (particles, fields, bounds, reactions, etc.)
so that independently defined processes can interoperate through typed stores.

This module defines those types in one place and exposes a single entry point:

    register_types(core) -> core
"""

from bigraph_schema import make_default

from spatio_flux.types import CountConcentrationVolume, positive_types


# -----------------------------------------------------------------------------
# Core type specs
# -----------------------------------------------------------------------------

# Generic bounds structure used for reaction constraints, etc.
BOUNDS_TYPE = {
    "lower": "maybe[float]",
    "upper": "maybe[float]",
}

# Basic particle: minimal state used by several particle processes
SIMPLE_PARTICLE_TYPE = {
    "id": "string",
    "position": "position",
    "mass": "mass{1.0}",
    "local": "map[concentration]",
    "exchange": "map[count]",
    "sub_masses": "map[mass{0.0}]",
}

# Extended particle: physics-focused attributes
COMPLEX_PARTICLE_TYPE = {
    "_inherit": "particle",
    "shape": "enum[circle,segment]",
    "velocity": "tuple[set_float,set_float]",
    "inertia": "set_float",
    "friction": "set_float",
    "elasticity": "set_float",
    "radius": "set_float",
}

# Cardinal boundary names (used for boundary processes / constraints)
BOUNDARY_SIDE_TYPE = "enum[left,right,top,bottom]"

# A simple reaction specification used by kinetics-like processes
REACTION_TYPE = {
    "reactant": "string",          # substrate id OR "mass"
    "product": "string",           # substrate id OR "mass"
    "km": "float",
    "vmax": "float",
    "yield": "float{1.0}",
}

# Spatial fields: map[molecule_id -> array]
FIELDS_TYPE = {
    "_type": "map",
    "_value": {
        "_type": "array",
        "_data": "float64",
    },
}


# -----------------------------------------------------------------------------
# Registry dictionaries
# -----------------------------------------------------------------------------

SPATIO_FLUX_TYPES = {
    **positive_types,
    "position": "tuple[float,float]",
    "particle": SIMPLE_PARTICLE_TYPE,
    "complex_particle": COMPLEX_PARTICLE_TYPE,
    "bounds": BOUNDS_TYPE,
    "fields": FIELDS_TYPE,
    "count_concentration_volume": CountConcentrationVolume,
}

TYPES_DICT = {
    **SPATIO_FLUX_TYPES,
    "boundary_side": BOUNDARY_SIDE_TYPE,
    "reaction": REACTION_TYPE,
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def register_types(core):
    core.register_types(TYPES_DICT)
    return core
