from dataclasses import dataclass, is_dataclass, field
from bigraph_schema.schema import Float, Array
from bigraph_schema.methods import apply


@dataclass(kw_only=True)
class SetFloat(Float):
    pass


@dataclass(kw_only=True)
class PositiveFloat(Float):
    pass


@dataclass(kw_only=True)
class Concentration(PositiveFloat):
    pass


@dataclass(kw_only=True)
class Count(PositiveFloat):
    pass


@dataclass(kw_only=True)
class PositiveArray(Array):
    pass


@apply.dispatch
def render(schema: PositiveFloat):
    return 'positive_float'

@apply.dispatch
def render(schema: PositiveArray):
    return 'positive_array'

@apply.dispatch
def render(schema: Concentration):
    return 'concentration'

@apply.dispatch
def render(schema: Count):
    return 'count'

@apply.dispatch
def render(schema: SetFloat):
    return 'set_float'


@apply.dispatch
def apply(schema: SetFloat, state, update, path):
    return update


@apply.dispatch
def apply(schema: PositiveFloat, state, update, path):
    new_value = state + update
    return max(0, new_value), []
    

@apply.dispatch
def apply(schema: PositiveArray, current, update, path):
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
                return np.maximum(0, current_array + update_dict), []

    if not isinstance(current, np.ndarray):
        if isinstance(update, dict):
            raise ValueError("Cannot apply dict update to scalar current")
        return np.maximum(0, current + update), []

    result = np.copy(current)
    recursive_update(result, current, update)
    return result, []


positive_types = {
    'positive_float': PositiveFloat,
    'positive_array': PositiveArray,
    'count': Count,
    'concentration': Concentration,
    'set_float': SetFloat}




