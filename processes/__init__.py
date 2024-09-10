"""
TODO: import all processes here and add to core
TODO -- make a "register_types" function that takes a core, registers all types and returns the core.
"""

from process_bigraph import ProcessTypes

# make type system
core = ProcessTypes()

def apply_non_negative(schema, current, update, core):
    new_value = current + update
    return max(0, new_value)


positive_float = {
    '_type': 'positive_float',
    '_inherit': 'float',
    '_apply': apply_non_negative
}
core.register('positive_float', positive_float)

bounds_type = {
    'lower': 'maybe[float]',
    'upper': 'maybe[float]'
}
core.register_process('bounds', bounds_type)

