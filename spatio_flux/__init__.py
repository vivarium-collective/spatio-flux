"""
TODO: import all processes here and add to core
TODO -- make a "register_types" function that takes a core, registers all types and returns the core.
"""

from bigraph_schema import default
from process_bigraph import ProcessTypes
from spatio_flux.processes import PROCESS_DICT
from spatio_flux.processes.dfba import build_path


def apply_non_negative(schema, current, update, core):
    new_value = current + update
    return max(0, new_value)


positive_float = {
    '_inherit': 'float',
    '_apply': apply_non_negative}


bounds_type = {
    'lower': 'maybe[float]',
    'upper': 'maybe[float]'}


particle_type = {
    'id': 'string',
    'position': 'tuple[float,float]',
    'size': 'float',
    'mass': default('float', 1.0),
    'local': 'map[float]',
    'exchange': 'map[float]',    # {mol_id: delta_value}
}

boundary_side = 'enum[left,right,top,bottom]'


substrate_role_type = 'enum[reactant,product,enzyme]'
kinetics_type = {
    'vmax': 'float',
    'kcat': 'float',
    'role': 'substrate_role'}
reaction_type = 'map[kinetics]'


TYPES_DICT = {
    'positive_float': positive_float,
    'bounds': bounds_type,
    'particle': particle_type,
    'boundary_side': boundary_side,
    'substrate_role': substrate_role_type,
    'kinetics': kinetics_type,
    'reaction': reaction_type}


def register_types(core):
    for type_name, type_schema in TYPES_DICT.items():
        core.register(type_name, type_schema)
    for process_name, process in PROCESS_DICT.items():
        core.register_process(process_name, process)
    return core
