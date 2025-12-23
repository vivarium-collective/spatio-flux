import numpy as np
from bigraph_schema import deep_merge

from process_bigraph import default
from spatio_flux.library.tools import initialize_fields, build_path
from spatio_flux.processes import MonodKinetics, get_kinetics_process_from_registry
from spatio_flux.processes.particles import generate_multiple_particles_state, INITIAL_MASS_RANGE
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
    'particle_add_rate': 0.3,
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

def get_spatial_dFBA_process(
        model_id=None,  # choose default from ['ecoli core', 'ecoli', 'cdiff', 'pputida', 'yeast', 'llactis']
        config=None,
        path=None,
):
    """
    optional model_id if no 'model_grid' in config
    2D grid of DFBA processes
    """
    assert 'n_bins' in config, "Configuration must include 'n_bins' for spatial DFBA."

    path = path or ['fields']
    mol_ids = config.get("mol_ids") or ["glucose", "acetate"]
    biomass_id = config.get("biomass_id") or "dissolved biomass"

    # remove "biomass" from mol_ids if it exists
    if biomass_id in mol_ids:
        mol_ids.remove(biomass_id)

    if model_id and 'model_grid' not in config:
        nx, ny = config['n_bins']  # (x bins, y bins)
        # model_grid is (ny rows, nx cols): index [y][x]
        config['model_grid'] = [[model_id for _ in range(nx)] for _ in range(ny)]

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

import numpy as np

def get_fields(
    n_bins,
    mol_ids,
    initial_min_max=None,
    initial_fields=None,
    dtype=float,
):
    """
    Create spatial fields consistent with the (x,y) vs (row,col) convention.

    Parameters
    ----------
    n_bins : tuple (nx, ny)
        Number of bins in x and y (config-space).
    mol_ids : iterable of str
        Molecule IDs to create fields for.
    initial_min_max : dict, optional
        {mol_id: (min, max)} for random initialization.
    initial_fields : dict, optional
        Predefined fields. Must already have shape (ny, nx).
    dtype : numpy dtype
        dtype for created arrays.

    Returns
    -------
    dict : {mol_id: ndarray}
        Each ndarray has shape (ny, nx).
    """
    initial_min_max = initial_min_max or {}
    initial_fields = dict(initial_fields) if initial_fields else {}

    nx, ny = n_bins
    shape = (ny, nx)  # numpy arrays are (rows=y, cols=x)

    for mol_id in mol_ids:
        if mol_id in initial_fields:
            arr = np.asarray(initial_fields[mol_id])
            if arr.shape != shape:
                raise ValueError(
                    f"Initial field '{mol_id}' has shape {arr.shape}, "
                    f"expected {shape} (ny, nx) from n_bins={n_bins}"
                )
            continue

        lo, hi = initial_min_max.get(mol_id, (0.0, 1.0))
        initial_fields[mol_id] = np.random.uniform(
            low=lo,
            high=hi,
            size=shape
        ).astype(dtype)

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
            "_shape": (n_bins[1], n_bins[0]),  # (rows, cols) == (y_bins, x_bins)
            "_data": "float"
        },
        **initial_fields,
    }

def get_spatial_many_dfba(
        n_bins=(5, 5),
        model_id=None,
        mol_ids=None,
        biomass_id="dissolved biomass",
        path=None,
):
    if path is None:
        path = ["..", "fields"]
    nx, ny = n_bins
    dfba_processes_dict = {}
    for y in range(ny):         # rows
        for x in range(nx):     # cols
            # get a process state for each bin
            dfba_process = get_dfba_process_from_registry(
                model_id=model_id, path=path,
                biomass_id=biomass_id, i=x, j=y)
            dfba_processes_dict[f"dFBA[{x},{y}]"] = dfba_process

    return dfba_processes_dict


def get_spatial_many_kinetics(
        n_bins=(5, 5),
        model_id="single_substrate_assimilation",
        biomass_id="dissolved biomass",
        mol_ids=None,
        path=None,
):
    if path is None:
        path = ["..", "fields"]
    kinetics_processes_dict = {}
    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            kinetics_process = get_kinetics_process_from_registry(
                model_id=model_id, mol_ids=mol_ids,
                path=path, biomass_id=biomass_id, i=i, j=j)
            kinetics_processes_dict[f"monod_kinetics[{i},{j}]"] = kinetics_process
    return kinetics_processes_dict



def get_spatial_many_dfba_with_fields(
        n_bins=(5, 5),
        model_file=None,
        mol_ids=None,
        initial_min_max=None,  # {mol_id: (min, max)}
        initial_fields=None,   # {mol_id: np.ndarray or list of floats}
):
    return {
        "fields": get_fields_with_schema(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max, initial_fields=initial_fields),
        "spatial_dfba": get_spatial_many_dfba(model_id=model_file, mol_ids=mol_ids, n_bins=n_bins)
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

def get_brownian_movement_process(
        n_bins=(20, 20),
        bounds=(10.0, 10.0),
        diffusion_rate=1e-1,
        advection_rate=(0, 0),
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
        },
        'outputs': {
            'particles': ['particles'],
        },
    }

def get_boundaries_process(
    particle_process_name,
    bounds=(10.0, 10.0),
    add_rate=0.0,
    boundary_to_add=('top',),
    # sides that ABSORB (remove) particles; all other sides REFLECT by default
    boundary_to_remove=(),  # e.g. ('right',) or ('top','bottom','left','right')
    clamp_survivors=True,
    buffer=1e-4,
    mass_range=INITIAL_MASS_RANGE,
):
    """
    Build the ManageBoundaries step spec.

    Semantics:
      - By default, all sides are reflecting barriers (closed box).
      - Any side listed in boundary_to_remove is absorbing (particle removed if it crosses).
      - No "pass" behavior exists.
    """
    config = {
        'bounds': bounds,
        'add_rate': float(add_rate),
        'boundary_to_add': list(boundary_to_add),
        'boundary_to_remove': list(boundary_to_remove),
        'clamp_survivors': bool(clamp_survivors),
        'buffer': float(buffer),
        'mass_range': mass_range,
    }

    return {
        '_type': 'step',
        'address': 'local:ManageBoundaries',
        'config': config,
        'inputs': {
            'particles': ['particles'],
            'process_interval': [particle_process_name, 'interval']
        },
        'outputs': {
            'particles': ['particles'],
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
        '_type': 'step',
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
        '_type': 'step',
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


def get_mass_total_step(
        mass_sources=None,
        mass_key="mass",
):
    return {
        '_type': 'step',
        'address': 'local:ParticleTotalMass',
        'config': {
            'mass_sources': mass_sources or [],
            'mass_key': mass_key
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
                'monod_kinetics': {
                    '_type': 'process',
                    'address': default('protocol', 'local:MonodKinetics'),
                    'config': default('node', config),
                    'inputs': default('wires', {
                        'substrates': ['local'],
                        'biomass': ['mass']
                    }),
                    'outputs': default('wires', {
                        'substrates': ['exchange'],
                        'biomass': ['mass']
                    })
                },
            }
        }
    }

def get_particles_state(
    bounds=(10.0, 10.0),
    n_particles=10,
    mass_range=None,
):
    """
    Convenience wrapper to generate an initial particles map.

    Note:
      - n_bins and fields are intentionally NOT used here.
      - ParticleExchange will initialize local/exchange state later.
    """
    particles_state = generate_multiple_particles_state({
        'bounds': bounds,
        'n_particles': n_particles,
        'mass_range': mass_range,
    })

    return particles_state['particles']


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
                    'config': default('node', config),
                    'inputs': default('wires', {
                        'substrates': ['local'],
                        'biomass': ['mass']
                    }),
                    'outputs': default('wires', {
                        'substrates': ['exchange'],
                        'biomass': ['mass']
                    })
                }
            }
        }
    }



def get_community_dfba_particle_composition(core=None, models=None, default_address="local:DynamicFBA"):
    """
    Build a particle composition with multiple DynamicFBA processes per particle.
    Only supports the dict approach:
        models = {
            "ecoli core": {
                "model_file": "textbook",
                "substrate_update_reactions": {...},
                "kinetic_params": {...},
                "bounds": {...},
            },
            "ecoli variant": {
                ... variation ...
            },
        }
    """
    if models is None or not isinstance(models, dict) or len(models) == 0:
        raise ValueError("get_community_dfba_particle_composition requires a non-empty dict 'models'")
    allowed = {"model_file", "kinetic_params", "substrate_update_reactions", "bounds"}

    mass_names = {k: f"{k} mass" for k in models.keys()}
    processes = {}
    for model_key, model_cfg in models.items():
        if not isinstance(model_key, str) or not model_key:
            raise ValueError(f"Model key must be a non-empty string, got: {model_key!r}")
        if not isinstance(model_cfg, dict):
            raise ValueError(f"models[{model_key!r}] must be a dict")

        # Resolve defaults (registry or derived from model_file)
        model_file = model_cfg.get("model_file", model_key)
        if model_key in MODEL_REGISTRY_DFBA:
            base = dict(MODEL_REGISTRY_DFBA[model_key])
        else:
            base = get_dfba_config(model_file=model_file)

        # Merge: explicit model_cfg overrides base
        merged = {**base, **model_cfg}

        # Filter to DynamicFBA.config_schema keys only
        config = {k: merged.get(k) for k in allowed}

        # Ensure required keys exist in some form
        if not config.get("model_file"):
            raise ValueError(f"models[{model_key!r}] must define 'model_file' (or be resolvable via defaults)")
        config["kinetic_params"] = config.get("kinetic_params") or {}
        config["substrate_update_reactions"] = config.get("substrate_update_reactions") or {}
        config["bounds"] = config.get("bounds") or {}

        # Use model_key directly as the process node name
        processes[model_key] = {
            "_type": "process",
            "address": default("string", default_address),
            "config": default("node", config),
            "inputs": default(
                "wires",
                {
                    "substrates": ["local"],
                    "biomass": [mass_names[model_key]],
                },
            ),
            "outputs": default(
                "wires",
                {
                    "substrates": ["exchange"],
                    "biomass": [mass_names[model_key]],
                },
            ),
        }

    # add a mass step
    processes['particle_mass'] = {
        '_type': 'step',
        'address': 'local:ParticleTotalMass',
        'config': {
            'mass_sources': list(mass_names.values()),
            'mass_key': 'mass'
        },
        'inputs': {'particle': ['..', '*']},
        'outputs': {'particle': ['..', '*']}
    }


    return {
        "particles": {
            "_type": "map",
            "_value": processes,
        }
    }


