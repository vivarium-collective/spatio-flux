"""
TODO: import all processes here and add to core
TODO -- make a "register_types" function that takes a core, registers all types and returns the core.
"""
import numpy as np

from bigraph_schema import default

from spatio_flux.processes import PROCESS_DICT
from spatio_flux.processes.configs import build_path
from spatio_flux.processes.particles import BrownianMovement
from spatio_flux.plots.plot import plot_species_distributions_with_particles_to_gif
from spatio_flux.types import SetFloat, PositiveFloat, PositiveArray, Concentration, CountConcentrationVolume, positive_types

bounds_type = {
    'lower': 'maybe[float]',
    'upper': 'maybe[float]'}

simple_particle_type = {
    'id': 'string',
    'position': 'position',
    'mass': 'concentration{1.0}',
    'local': 'map[concentration]',
    'exchange': 'map[count]',
}

particle_type = {
    '_inherit': 'particle',
    'shape': 'enum[circle,segment]',
    'velocity': 'tuple[set_float,set_float]',
    'inertia': 'set_float',
    'friction': 'set_float',
    'elasticity': 'set_float',
    'radius': 'set_float',
}

boundary_side = 'enum[left,right,top,bottom]'

# substrate_role_type = 'enum[reactant,product,enzyme]'
reaction_type = {
    # wiring / meaning
    'reactant': 'string',         # substrate id OR "mass"
    'product': 'string',          # substrate id OR "mass"
    # 'role': 'substrate_role',     # optional semantic tag
    # Monod params (reaction-scoped)
    'km': 'float',
    'vmax': 'float',
    'yield': 'float{1.0}',
}

fields_type =  {
    '_type': 'map',
    '_value': {
        '_type': 'array',
        '_data': 'float64'
    },
}


SPATIO_FLUX_TYPES = {
    **positive_types,
    'position': 'tuple[float,float]',
    'particle': simple_particle_type,
    'complex_particle': particle_type,
    'bounds': bounds_type,
    'fields': fields_type,
    'count_concentration_volume': CountConcentrationVolume,
    # TODO fields, concentrations, fluxes, etc.
}

TYPES_DICT = {
    **SPATIO_FLUX_TYPES,
    'boundary_side': boundary_side,
    'reaction': reaction_type,
}


def register_types(core):
    core.register_types(TYPES_DICT)

    return core
