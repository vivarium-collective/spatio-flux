"""
TODO: import all processes here and add to core
TODO -- make a "register_types" function that takes a core, registers all types and returns the core.
"""
import numpy as np

from bigraph_schema import default
from process_bigraph import ProcessTypes
from spatio_flux.processes import PROCESS_DICT
from spatio_flux.processes.configs import build_path
from spatio_flux.processes.particles import BrownianMovement
from spatio_flux.viz.plot import plot_species_distributions_with_particles_to_gif


def apply_non_negative(schema, current, update, top_schema, top_state, path, core):
    new_value = current + update
    return max(0, new_value)

def apply_non_negative_array(schema, current, update, top_schema, top_state, path, core):
    def recursive_update(result_array, current_array, update_dict, index_path=()):
        if isinstance(update_dict, dict):
            for key, val in update_dict.items():
                recursive_update(result_array, current_array, val, index_path + (key,))
        else:
            if isinstance(current_array, np.ndarray):
                current_value = current_array[index_path]
                result_array[index_path] = np.maximum(0, current_value + update_dict)
            else:
                # Scalar fallback
                return np.maximum(0, current_array + update_dict)

    if not isinstance(current, np.ndarray):
        if isinstance(update, dict):
            raise ValueError("Cannot apply dict update to scalar current")
        return np.maximum(0, current + update)

    result = np.copy(current)
    recursive_update(result, current, update)
    return result


positive_float = {
    '_inherit': 'float',
    '_apply': apply_non_negative
}

positive_array = {
    '_inherit': 'array',
    '_apply': apply_non_negative_array,
}

bounds_type = {
    'lower': 'maybe[float]',
    'upper': 'maybe[float]'}

set_float = {
    '_type': 'float',
    '_apply': 'set'}

simple_particle_type = {
    'id': 'string',
    'position': 'position',
    'mass': default('concentration', 1.0),
    'local': 'map[concentration]',
    'exchange': 'map[delta]',  # TODO is this counts?
}

particle_type = {
    '_inherit': 'simple_particle',
    'shape': 'enum[circle,segment]',
    'velocity': 'tuple[set_float,set_float]',
    'inertia': 'set_float',
    'friction': 'set_float',
    'elasticity': 'set_float',
    'radius': 'set_float',
}

boundary_side = 'enum[left,right,top,bottom]'

substrate_role_type = 'enum[reactant,product,enzyme]'
kinetics_type = {
    'vmax': 'float',
    'kcat': 'float',
    'role': 'substrate_role'}
reaction_type = 'map[kinetics]'

fields_type =  {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    # '_shape': self.config['n_bins'],
                    '_data': 'concentration'
                },
        }


SPATIO_FLUX_TYPES = {
    'position': 'tuple[float,float]',
    'delta': 'float',
    'concentration': positive_float,
    'set_float': set_float,
    'simple_particle': simple_particle_type,
    'particle': particle_type,
    'bounds': bounds_type,
    'fields': fields_type,
    # TODO fields, concentrations, fluxes, etc.
}

TYPES_DICT = {
    **SPATIO_FLUX_TYPES,
    'positive_float': positive_float,
    'positive_array': positive_array,
    'boundary_side': boundary_side,
    'substrate_role': substrate_role_type,
    'kinetics': kinetics_type,
    'reaction': reaction_type,
}


def register_types(core):
    for type_name, type_schema in TYPES_DICT.items():
        core.register(type_name, type_schema)
    for process_name, process in PROCESS_DICT.items():
        core.register_process(process_name, process)
    return core


def core_import(core=None, config=None):
    if not core:
        core = ProcessTypes()
    register_types(core)
    return core
