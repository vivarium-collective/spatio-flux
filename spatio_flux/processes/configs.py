import numpy as np
from bigraph_schema import deep_merge

from process_bigraph import default
from spatio_flux.library.helpers import initialize_fields, build_path
from spatio_flux.processes import MonodKinetics
from spatio_flux.processes.particles import BrownianMovement
from spatio_flux.processes.dfba import get_dfba_process_from_registry, MODEL_REGISTRY_DFBA

default_config = {
    'total_time': 100.0,
    'bounds': (10.0, 20.0),
    'n_bins': (8, 16),
    'mol_ids': ['glucose', 'acetate', 'dissolved biomass', 'detritus'],
    'field_diffusion_rate': 1e-1,
    'field_advection_rate': (0, 0),
    'initial_min_max': {
        'glucose': (10, 10),
        'acetate': (0, 0),
        'dissolved biomass': (0, 0.1),
        'detritus': (0, 0)
    },
    # set particles
    'n_particles': 10,
    'particle_diffusion_rate': 1e-1,
    'particle_advection_rate': (0, -0.1),
    'particle_add_probability': 0.3,
    'particle_boundary_to_add': ['top'],
    'particle_boundary_to_remove': ['top', 'bottom', 'left', 'right'],
    'particle_field_interactions': {
        'glucose': {
            'vmax': 0.1,
            'Km': 1.0,
            'interaction_type': 'uptake'
        },
        'detritus': {
            'vmax': -0.1,
            'Km': 1.0,
            'interaction_type': 'secretion'
        },
    },
}


# ===========
# Single DFBA
# ===========

def get_dfba_config(
        model_file="textbook",
        kinetic_params=None,
        substrate_update_reactions=None,
        bounds=None
):
    return {
        "default_model_file": model_file,
        "kinetic_params": kinetic_params,
        "substrate_update_reactions": substrate_update_reactions,
        "bounds": bounds
    }


def get_single_dfba_process(
        model_id="ecoli core",
        mol_ids=None,
        biomass_id="dissolved biomass",
        bounds=None,
        path=None,
        i=None,
        j=None,
):
    """
    Constructs a configuration dictionary for a dynamic FBA process with optional path indices.
    """
    if path is None:
        path = ["..", "fields"]
    if mol_ids is None:
        mol_ids = ["glucose", "acetate"]

    # remove "biomass" from mol_ids if it exists
    if biomass_id in mol_ids:
        mol_ids.remove(biomass_id)

    config = {'mol_ids': mol_ids, 'biomass_id': biomass_id}

    if model_id in MODEL_REGISTRY_DFBA:
        dfba_config = MODEL_REGISTRY_DFBA.get(model_id, {})
    else:
        dfba_config = get_dfba_config(model_file=model_id)

    config = deep_merge(config, dfba_config)

    return {
        "_type": "process",
        "address": "local:DynamicFBA",
        "config": config,
        "inputs": {
            "substrates": {mol_id: build_path(path, mol_id, i, j) for mol_id in mol_ids},
            "biomass": build_path(path, biomass_id, i, j)
        },
        "outputs": {
            "substrates": {mol_id: build_path(path, mol_id, i, j) for mol_id in mol_ids},
            "biomass": build_path(path, biomass_id, i, j)
        }
    }

def get_spatial_dfba_process(
        model_id=None,  # choose default from ['ecoli core', 'ecoli', 'cdiff', 'pputida', 'yeast', 'llactis']
        config=None,
        path=None,
):
    assert 'n_bins' in config, "Configuration must include 'n_bins' for spatial DFBA."

    path = path or ['fields']
    mol_ids = config.get("mol_ids") or ["glucose", "acetate"]
    biomass_id = config.get("biomass_id") or "dissolved biomass"

    # remove "biomass" from mol_ids if it exists
    if biomass_id in mol_ids:
        mol_ids.remove(biomass_id)

    if model_id in MODEL_REGISTRY_DFBA:
        dfba_config = MODEL_REGISTRY_DFBA.get(model_id, {})
    else:
        dfba_config = get_dfba_config(model_file=model_id)

    config = deep_merge(config, dfba_config)

    return {
        "_type": "process",
        "address": "local:SpatialDFBA",
        "config": config,
        "inputs": {
            "fields": {mol_id: build_path(path, mol_id) for mol_id in mol_ids},
            "biomass": build_path(path, biomass_id)
        },
        "outputs": {
            "fields": {mol_id: build_path(path, mol_id) for mol_id in mol_ids},
            "biomass": build_path(path, biomass_id)
        }
    }

# ============
# Spatial DFBA
# ============

def get_fields(n_bins, mol_ids, initial_min_max=None, initial_fields=None):
    initial_min_max = initial_min_max or {}
    initial_fields = initial_fields or {}

    for mol_id in mol_ids:
        if mol_id not in initial_fields:
            minmax = initial_min_max.get(mol_id, (0, 1))
            initial_fields[mol_id] = np.random.uniform(
                low=minmax[0],
                high=minmax[1],
                size=n_bins
            )

    return initial_fields

def get_fields_with_schema(
        n_bins,
        mol_ids=None,
        initial_min_max=None,  # {mol_id: (min, max)}
        initial_fields=None
):
    initial_min_max = initial_min_max or {}
    initial_fields = initial_fields or {}

    if mol_ids is None:
        if initial_min_max:
            mol_ids = list(initial_min_max.keys())
        else:
            mol_ids = ["glucose", "acetate", "dissolved biomass"]

    initial_fields = get_fields(n_bins, mol_ids, initial_min_max, initial_fields)

    return {
        "_type": "map",
        "_value": {
            "_type": "array",
            "_shape": n_bins,
            "_data": "concentration"
        },
        **initial_fields,
    }

def get_spatial_many_dfba(
        n_bins=(5, 5),
        model_file=None,
        mol_ids=None,
        biomass_id="dissolved biomass",
):
    dfba_processes_dict = {}
    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            # get a process state for each bin
            dfba_process = get_dfba_process_from_registry(
                model_id=model_file, path=["..", "fields"],
                biomass_id=biomass_id, i=i, j=j)
            dfba_processes_dict[f"dFBA[{i},{j}]"] = dfba_process

    return dfba_processes_dict

def get_spatial_many_dfba_with_fields(
        n_bins=(5, 5),
        model_file=None,
        mol_ids=None,
        initial_min_max=None,  # {mol_id: (min, max)}
        initial_fields=None,   # {mol_id: np.ndarray or list of floats}
):
    return {
        "fields": get_fields_with_schema(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max, initial_fields=initial_fields),
        "spatial_dfba": get_spatial_many_dfba(model_file=model_file, mol_ids=mol_ids, n_bins=n_bins)
    }

# ===================
# Diffusion-Advection
# ===================

def get_diffusion_advection_process(
        bounds=(10.0, 10.0),
        n_bins=(5, 5),
        mol_ids=None,
        default_diffusion_rate=1e-1,
        default_advection_rate=(0, 0),
        diffusion_coeffs=None,
        advection_coeffs=None,
):
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'dissolved biomass']
    if diffusion_coeffs is None:
        diffusion_coeffs = {}
    if advection_coeffs is None:
        advection_coeffs = {}

    # fill in the missing diffusion and advection rates
    diffusion_coeffs_all = {
        mol_id: diffusion_coeffs.get(mol_id, default_diffusion_rate)
        for mol_id in mol_ids
    }
    advection_coeffs_all = {
        mol_id: advection_coeffs.get(mol_id, default_advection_rate)
        for mol_id in mol_ids
    }

    return {
            '_type': 'process',
            'address': 'local:DiffusionAdvection',
            'config': {
                'n_bins': n_bins,
                'bounds': bounds,
                'default_diffusion_rate': 1e-1,
                'default_diffusion_dt': 1e-1,
                'diffusion_coeffs': diffusion_coeffs_all,
                'advection_coeffs': advection_coeffs_all,
            },
            'inputs': {
                'fields': ['fields']
            },
            'outputs': {
                'fields': ['fields']
            }
        }

# =================
# Particle Movement
# =================

def get_particle_movement_process(
        n_bins=(20, 20),
        bounds=(10.0, 10.0),
        diffusion_rate=1e-1,
        advection_rate=(0, 0),
        add_probability=0.0,
        boundary_to_add=['top'],
        boundary_to_remove=['top', 'bottom', 'left', 'right'],
        division_mass_threshold=0.0,
):
    config = locals()
    # Remove any key-value pair where the value is None
    config = {key: value for key, value in config.items() if value is not None}

    return {
        '_type': 'process',
        'address': 'local:BrownianMovement',
        'config': config,
        'inputs': {
            'particles': ['particles'],
            # 'fields': ['fields']
        },
        'outputs': {
            'particles': ['particles'],
            # 'fields': ['fields']
        },
    }

def get_particle_exchange_process(
        n_bins=(20, 20),
        bounds=(10.0, 10.0),
        rates_are_per_time=True,
        apply_mass_balance=False,
):
    config = locals()
    # Remove any key-value pair where the value is None
    config = {key: value for key, value in config.items() if value is not None}

    return {
        '_type': 'process',
        'address': 'local:ParticleExchange',
        'config': config,
        'inputs': {
            'particles': ['particles'],
            'fields': ['fields'],
        },
        'outputs': {
            'particles': ['particles'],
            'fields': ['fields'],
        },
    }

def get_particle_divide_process(
        division_mass_threshold=0.0
):
    return {
        '_type': 'process',
        'address': 'local:ParticleDivision',
        'config': {
            'division_mass_threshold': division_mass_threshold
        },
        'inputs': {
            'particles': ['particles']
        },
        'outputs': {
            'particles': ['particles']
        }
    }


# ===============
# Particle-COMETS
# ===============

def get_kinetic_particle_composition(core, config=None):
    config = config or core.default(MonodKinetics.config_schema)
    return {
        'particles': {
            '_type': 'map',
            '_value': {
                # '_inherit': 'particle',
                'kinetics': {
                    '_type': 'process',
                    'address': default('string', 'local:MonodKinetics'),
                    'config': default('quote', config),
                    '_inputs': {
                        'mass': 'float',
                        'substrates': 'map[concentration]'
                    },
                    '_outputs':  {
                        'mass': 'float',
                        'substrates': 'map[float]'
                    },
                    'inputs': default(
                        'tree[wires]', {
                            'mass': ['mass'],
                            'substrates': ['local']}),
                    'outputs': default(
                        'tree[wires]', {
                            'mass': ['mass'],
                            'substrates': ['exchange']})
                }
            }
        }
    }

def get_particles_state(
        n_bins=(10, 10),
        bounds=(10.0, 10.0),
        fields=None,
        n_particles=10,
        mass_range=None,
):
    fields = fields or {}
    # add particles process
    particles = BrownianMovement.generate_state(
        config={
            'n_particles': n_particles,
            'bounds': bounds,
            'fields': fields,
            'n_bins': n_bins,
            'mass_range': mass_range,
        })
    return particles['particles']

# ==============
# dFBA-Particles
# ==============

def get_dfba_particle_composition(core=None, model_file=None):
    if model_file in MODEL_REGISTRY_DFBA:
        config = MODEL_REGISTRY_DFBA[model_file]
    else:
        config = get_dfba_config(model_file=model_file)
    return {
        'particles': {
            '_type': 'map',
            '_value': {
                'dFBA': {
                    '_type': 'process',
                    'address': default('string', 'local:DynamicFBA'),
                    'config': default('quote', config),
                    'inputs': default('tree[wires]', {
                        'substrates': ['local'],
                        'biomass': ['mass']
                    }),
                    'outputs': default('tree[wires]', {
                        'substrates': ['exchange'],
                        'biomass': ['mass']
                    })
                }
            }
        }
    }
