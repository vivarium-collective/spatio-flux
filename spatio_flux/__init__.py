"""
TODO: import all processes here and add to core
TODO -- make a "register_types" function that takes a core, registers all types and returns the core.
"""

from process_bigraph import ProcessTypes
from spatio_flux.processes import register_processes


def apply_non_negative(schema, current, update, core):
    new_value = current + update
    return max(0, new_value)


positive_float = {
    '_type': 'positive_float',
    '_inherit': 'float',
    '_apply': apply_non_negative}


bounds_type = {
    'lower': 'maybe[float]',
    'upper': 'maybe[float]'}


particle_type = {
    'id': 'string',
    'position': 'tuple[float,float]',
    'size': 'float',
    'local': 'map[float]',
    'exchange': 'map[float]',    # {mol_id: delta_value}
}


def register_types(core):
    core.register('positive_float', positive_float)
    core.register('bounds', bounds_type)
    core.register('particle', particle_type)
    
    return register_processes(core)
