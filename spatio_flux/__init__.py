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
from spatio_flux.plots.plot import plot_species_distributions_with_particles_to_gif


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
    'exchange': 'map[counts]',  # TODO is this counts?
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

substrate_role_type = 'enum[reactant,product,enzyme]'
kinetics_type = {
    'vmax': 'float',
    'kcat': 'float',
    'role': 'substrate_role'}
reaction_type = 'map[kinetics]'

# fields_type = 'map[concentration]'
fields_type =  {
    '_type': 'map',
    '_value': {
        '_type': 'array',
        # '_shape': self.config['n_bins'],
        '_data': 'concentration'
    },
}


def apply_conc_counts_volume(schema, current, update, top_schema, top_state, path, core):
    """
    Type: {
        'volume': float,          # container size
        'counts': float,          # total amount
        'concentration': float,   # counts / volume
    }

    Semantics:
      - Updates are treated as *deltas*:
          update = {
              'volume': ΔV (optional),
              'counts': ΔN (optional),
              'concentration': ΔC (optional),
          }
      - Counts are the canonical amount.
      - Concentration is derived: concentration = counts / volume.
      - If volume changes, we keep counts (amount) fixed and recompute concentration.
      - If concentration changes, we interpret ΔC as an additional amount: ΔN_conc = ΔC * V_new.
    """
    if current is None:
        current = {'volume': 0.0, 'counts': 0.0, 'concentration': 0.0}

    if not isinstance(update, dict):
        raise ValueError(
            f"Update to conc_counts_volume at {path} must be a dict, got {type(update)}"
        )

    # Extract current state
    volume = float(current.get('volume', 0.0))
    counts = float(current.get('counts', 0.0))

    # Extract deltas (default to 0)
    dV = float(update.get('volume', 0.0)) if 'volume' in update else 0.0
    dN = float(update.get('counts', 0.0)) if 'counts' in update else 0.0
    dC = float(update.get('concentration', 0.0)) if 'concentration' in update else 0.0

    # 1. Update volume first
    V_new = volume + dV
    if V_new <= 0:
        raise ValueError(
            f"Volume would become non-positive at {path}: {V_new}"
        )

    # 2. Interpret all changes as changes in amount (counts are canonical)
    amount = counts
    d_amount_from_counts = dN
    d_amount_from_conc = dC * V_new  # concentration * volume = counts (in arbitrary units)

    amount_new = amount + d_amount_from_counts + d_amount_from_conc

    # Enforce non-negativity on amount
    if amount_new < 0:
        amount_new = 0.0

    counts_new = amount_new
    concentration_new = counts_new / V_new if V_new > 0 else 0.0

    return {
        'volume': V_new,
        'counts': counts_new,
        'concentration': concentration_new,
    }



conc_counts_volume_type = {
    'volume': 'float',
    'counts': 'float',
    'concentration': 'float',
    # custom _apply controls how updates are combined.
    '_apply': apply_conc_counts_volume,
}


SPATIO_FLUX_TYPES = {
    'position': 'tuple[float,float]',
    'counts': 'float',
    'concentration': positive_float,
    'set_float': set_float,
    'particle': simple_particle_type,
    'complex_particle': particle_type,
    'bounds': bounds_type,
    'fields': fields_type,
    'conc_counts_volume': conc_counts_volume_type,
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
