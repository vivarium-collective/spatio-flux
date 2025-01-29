"""
TODO: import all processes here and add to core
TODO -- make a "register_types" function that takes a core, registers all types and returns the core.
"""

from process_bigraph import ProcessTypes
from spatio_flux.processes import PROCESS_DICT


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

TYPES_DICT = {
    'positive_float': positive_float,
    'bounds': bounds_type,
    'particle': particle_type
}


def register_types(core):
    for type_name, type_schema in TYPES_DICT.items():
        core.register(type_name, type_schema)
    for process_name, process in PROCESS_DICT.items():
        core.register_process(process_name, process)
    return core
