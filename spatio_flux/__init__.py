"""
TODO: import all processes here and add to core
TODO -- make a "register_types" function that takes a core, registers all types and returns the core.
"""
import numpy as np

from bigraph_schema import default
from process_bigraph import ProcessTypes, register_types as register_process_types
from vivarium.vivarium import VivariumTypes

from spatio_flux.processes import PROCESS_DICT
from spatio_flux.processes.configs import build_path
from spatio_flux.processes.particles import BrownianMovement
from spatio_flux.plots.plot import plot_species_distributions_with_particles_to_gif


# --- apply functions ---

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

def apply_conc_counts(schema, current, update, top_schema, top_state, path, core):
    """
    Generic conc-counts-volume handler that works for both scalars and fields.

    Type (conceptually):
    {
        'volume': float or np.ndarray,        # container size per site
        'counts': float or np.ndarray,        # total amount per site
        'concentration': float or np.ndarray, # counts / volume per site
    }

    Semantics:
      - Updates are treated as *deltas*:
          update = {
              'volume': ΔV (optional),
              'counts': ΔN (optional),
              'concentration': ΔC (optional),
          }
      - Counts are canonical.
      - Concentration is derived: concentration = counts / volume.
      - If volume changes (and we're allowed to), we keep counts fixed and recompute concentration.
      - If concentration changes, we interpret ΔC as an additional amount: ΔN_conc = ΔC * V_new.

    Extra schema flag:
      - schema.get('_fixed_volume', False) -> if True, any non-zero ΔV is disallowed.
    """

    # Allow schema to control whether volume is fixed (e.g. for fields)
    fixed_volume = bool(schema.get('_fixed_volume', False))

    if current is None:
        current = {'volume': 0.0, 'counts': 0.0, 'concentration': 0.0}

    if not isinstance(update, dict):
        raise ValueError(
            f"Update to conc_counts at {path} must be a dict, got {type(update)}"
        )

    # --- Extract current state as arrays ---
    volume_current = current.get('volume', 0.0)
    counts_current = current.get('counts', 0.0)

    volume_arr = np.asarray(volume_current, dtype=float)
    counts_arr = np.asarray(counts_current, dtype=float)

    # --- Handle volume update / fixed volume ---
    if fixed_volume:
        # Volume is not allowed to change
        if 'volume' in update:
            dV_arr = np.asarray(update['volume'], dtype=float)
            if np.any(dV_arr != 0.0):
                raise ValueError(
                    f"Volume updates are not allowed for conc_counts at {path} "
                    f"(schema has _fixed_volume=True)"
                )
        V_new_arr = volume_arr
    else:
        dV_arr = np.asarray(update.get('volume', 0.0), dtype=float)
        V_new_arr = volume_arr + dV_arr
        if np.any(V_new_arr <= 0):
            raise ValueError(
                f"Volume would become non-positive at {path}: {V_new_arr}"
            )

    # --- Deltas for counts and concentration ---
    dN_arr = np.asarray(update.get('counts', 0.0), dtype=float)
    dC_arr = np.asarray(update.get('concentration', 0.0), dtype=float)

    # Broadcasting handles scalar vs array cases
    d_amount_from_counts = dN_arr
    d_amount_from_conc = dC_arr * V_new_arr

    counts_new = counts_arr + d_amount_from_counts + d_amount_from_conc

    # Enforce non-negativity elementwise
    counts_new = np.where(counts_new < 0, 0.0, counts_new)

    conc_new = counts_new / V_new_arr

    # --- Helper to keep scalars as scalars, arrays as arrays ---
    def maybe_scalar(x, prefer_scalar):
        arr = np.asarray(x, dtype=float)
        if prefer_scalar and arr.shape == ():
            return float(arr)
        return arr

    # Detect whether original was scalar
    counts_was_scalar = np.asarray(counts_current).shape == ()
    volume_was_scalar = np.asarray(volume_current).shape == ()

    # For volume:
    V_out = maybe_scalar(V_new_arr, volume_was_scalar)

    # For counts and concentration:
    counts_out = maybe_scalar(counts_new, counts_was_scalar)
    conc_out = maybe_scalar(conc_new, counts_was_scalar)

    return {
        'volume': V_out,
        'counts': counts_out,
        'concentration': conc_out,
    }


# --- Types ---

conc_counts_type = {
    'volume': 'float',
    'counts': 'float',
    'concentration': 'float',
    '_apply': apply_conc_counts,
}

conc_counts_field_type = {
    # Could be scalar (uniform per site) or an array with a given shape
    'volume': 'float',  # or an array spec if you prefer per-site volumes

    'counts': {
        '_type': 'array',
        '_data': 'float',
        # '_shape': (nx, ny)  # whatever n_bins is
    },
    'concentration': {
        '_type': 'array',
        '_data': 'float',
        # '_shape': (nx, ny)
    },

    # tells the unified apply to disallow Δvolume
    '_fixed_volume': True,
    '_apply': apply_conc_counts,
}

fields_type =  {
    '_type': 'map',
    '_value': {
        '_type': 'array',
        # '_shape': self.config['n_bins'],
        '_data': 'concentration'
    },
}

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
    'exchange': 'map[counts]',
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



#--- Register types ---

SPATIO_FLUX_TYPES = {
    'position': 'tuple[float,float]',
    'counts': 'float',
    'concentration': positive_float,
    'set_float': set_float,
    'particle': simple_particle_type,
    'complex_particle': particle_type,
    'bounds': bounds_type,
    'fields': fields_type,
    'conc_counts': conc_counts_type,
    'conc_counts_field': conc_counts_field_type,
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


def build_core():
    """Construct and return a Vivarium core with process + spatio-flux types registered."""
    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)
    return core
