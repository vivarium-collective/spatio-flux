"""Shared constants and helpers used by composite generators.

These lived in test_suite.py before the migration. They're extracted here
so generator modules don't have to import from test_suite.py — which would
be a circular import, because test_suite.py imports from this package to
look up generators.

test_suite.py re-exports these names for backward compatibility with any
external caller that imported them from there.
"""
from __future__ import annotations

# Standard small lattice used by particle / kinetics composites
SQUARE_BOUNDS = (50.0, 50.0)
SQUARE_BINS = (10, 10)

# Standard non-square lattice used by diffusion composites
DEFAULT_BOUNDS = (40.0, 80.0)
DEFAULT_BINS = (10, 20)
DEFAULT_BINS_SMALL = (2, 4)

DEFAULT_ADVECTION = (0.0, 0.2)
DEFAULT_DIFFUSION = 0.5
DEFAULT_ADD_RATE = 0.1
DEFAULT_ADD_BOUNDARY = ['top', 'left', 'right']
DEFAULT_REMOVE_BOUNDARY = ['left', 'right']
DEFAULT_INITIAL_MIN_MAX = {
    'glucose': (10, 10),
    'acetate': (0, 0),
    'dissolved biomass': (0, 0.1),
}


def get_newtonian_particles_process(config=None):
    """Helper that wraps a Pymunk newtonian-particles process address."""
    return {
        '_type': 'process',
        'address': 'local:PymunkParticleMovement',
        'config': config,
        'inputs':  {'particles': ['particles']},
        'outputs': {'particles': ['particles']},
    }
