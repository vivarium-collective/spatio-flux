import numpy as np

from process_bigraph import default
from spatio_flux.library.helpers import initialize_fields, build_path
from spatio_flux.processes import MinimalParticle
from spatio_flux.processes.particles import Particles


default_config = {
    'total_time': 100.0,
    'bounds': (10.0, 20.0),
    'n_bins': (8, 16),
    'mol_ids': ['glucose', 'acetate', 'biomass', 'detritus'],
    'field_diffusion_rate': 1e-1,
    'field_advection_rate': (0, 0),
    'initial_min_max': {
        'glucose': (10, 10),
        'acetate': (0, 0),
        'biomass': (0, 0.1),
        'detritus': (0, 0)
    },
    # set particles
    'n_particles': 10,
    'particle_diffusion_rate': 1e-1,
    'particle_advection_rate': (0, -0.1),
    'particle_add_probability': 0.3,
    'particle_boundary_to_add': ['top'],
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


# =====
# DFBA
# =====

def get_dfba_config(
        model_file="textbook",
        kinetic_params=None,
        substrate_update_reactions=None,
        biomass_identifier="biomass",
        bounds=None
):
    if substrate_update_reactions is None:
        substrate_update_reactions = {
            "glucose": "EX_glc__D_e",
            "acetate": "EX_ac_e"}
    if bounds is None:
        bounds = {
            "EX_o2_e": {"lower": -2, "upper": None},
            "ATPM": {"lower": 1, "upper": 1}}
    if kinetic_params is None:
        kinetic_params = {
            "glucose": (0.5, 1),
            "acetate": (0.5, 2)}
    return {
        "model_file": model_file,
        "kinetic_params": kinetic_params,
        "substrate_update_reactions": substrate_update_reactions,
        "biomass_identifier": biomass_identifier,
        "bounds": bounds
    }


def get_dfba_process_state(
        model_file="textbook",
        mol_ids=None,
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
        mol_ids = ["glucose", "acetate", "biomass"]

    # remove "biomass" from mol_ids if it exists
    if "biomass" in mol_ids:
        mol_ids.remove("biomass")

    return {
        "_type": "process",
        "address": "local:DynamicFBA",
        "config": get_dfba_config(model_file=model_file),
        "inputs": {
            "substrates": {mol_id: build_path(path, mol_id, i, j) for mol_id in mol_ids},
            "biomass": build_path(path, "biomass", i, j)
        },
        "outputs": {
            "substrates": {mol_id: build_path(path, mol_id, i, j) for mol_id in mol_ids},
            "biomass": build_path(path, "biomass", i, j)
        }
    }


def get_spatial_dfba_spec(
        n_bins=(5, 5),
        mol_ids=None
):
    if mol_ids is None:
        mol_ids = ["glucose", "acetate", "biomass"]
    dfba_processes_dict = {}
    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            dfba_processes_dict[f"[{i},{j}]"] = get_dfba_process_state(mol_ids=mol_ids, path=["..", "fields"], i=i, j=j)
    return dfba_processes_dict


def get_spatial_dfba_state(
        n_bins=(5, 5),
        mol_ids=None,
        initial_min_max=None,  # {mol_id: (min, max)}
):
    if mol_ids is None:
        mol_ids = ["glucose", "acetate", "biomass"]

    initial_fields = initialize_fields(n_bins=n_bins, initial_min_max=initial_min_max)

    return {
        "fields": {
            "_type": "map",
            "_value": {
                "_type": "array",
                "_shape": n_bins,
                "_data": "positive_float"
            },
            **initial_fields,
        },
        "spatial_dfba": get_spatial_dfba_spec(n_bins=n_bins, mol_ids=mol_ids)
    }


# ===================
# Diffusion-Advection
# ===================

def get_diffusion_advection_process_state(
        bounds=(10.0, 10.0),
        n_bins=(5, 5),
        mol_ids=None,
        default_diffusion_rate=1e-1,
        default_advection_rate=(0, 0),
        diffusion_coeffs=None,
        advection_coeffs=None,
):
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']
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


def get_diffusion_advection_state(
        bounds=(10.0, 10.0),
        n_bins=(5, 5),
        mol_ids=None,
        initial_max=None,
        default_diffusion_rate=1e-1,
        default_advection_rate=(0, 0),
        diffusion_coeffs=None,
        advection_coeffs=None,
):
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']
    if initial_max is None:
        initial_max = {
            'glucose': 20,
            'acetate': 0,
            'biomass': 0.1
        }
    initial_fields = {
        mol_id: np.random.uniform(low=0, high=initial_max[mol_id], size=n_bins)
        for mol_id in mol_ids}

    return {
        'fields': {
            '_type': 'map',
            '_value': {
                '_type': 'array',
                '_shape': n_bins,
                '_data': 'positive_float'
            },
            **initial_fields,
        },
        'diffusion': get_diffusion_advection_process_state(
            bounds=bounds,
            n_bins=n_bins,
            mol_ids=mol_ids,
            default_diffusion_rate=default_diffusion_rate,
            default_advection_rate=default_advection_rate,
            diffusion_coeffs=diffusion_coeffs,
            advection_coeffs=advection_coeffs,
        ),
    }


# ================
# Particle & Field
# ================

def get_particle_movement_process_state(
        n_bins=(20, 20),
        bounds=(10.0, 10.0),
        diffusion_rate=1e-1,
        advection_rate=(0, 0),
        add_probability=0.0,
        boundary_to_add=['top'],
):
    config = locals()
    # Remove any key-value pair where the value is None
    config = {key: value for key, value in config.items() if value is not None}

    return {
        '_type': 'process',
        'address': 'local:Particles',
        'config': config,
        'inputs': {
            'particles': ['particles'],
            'fields': ['fields']},
        'outputs': {
            'particles': ['particles'],
            'fields': ['fields']
        }
    }


def get_particles_state(
        n_bins=(20, 20),
        bounds=(10.0, 10.0),
        n_particles=15,
        diffusion_rate=0.1,
        advection_rate=(0, -0.1),
        boundary_to_add=None,
        add_probability=0.4,
        initial_min_max=None,
):
    if boundary_to_add is None:
        boundary_to_add = ['top']
    fields = initialize_fields(n_bins, initial_min_max)

    # initialize particles
    # TODO -- this needs to be a static method??
    particles = Particles.generate_state(
        config={
            'n_particles': n_particles,
            'n_bins': n_bins,
            'bounds': bounds,
            # 'fields': fields
        }
    )

    return {
        'fields': fields,
        'particles': particles['particles'],
        'particle_movement': get_particle_movement_process_state(
            n_bins=n_bins,
            bounds=bounds,
            diffusion_rate=diffusion_rate,
            advection_rate=advection_rate,
            add_probability=add_probability,
            boundary_to_add=boundary_to_add,
        )
    }


# ============================
# Particle COMETS Compositions
# ============================

def get_minimal_particle_composition(core, config=None):
    config = config or core.default(MinimalParticle.config_schema)
    return {
        'particles': {
            '_type': 'map',
            '_value': {
                # '_inherit': 'particle',
                'minimal_particle': {
                    '_type': 'process',
                    'address': default('string', 'local:MinimalParticle'),
                    'config': default('quote', config),
                    '_inputs': {
                        'mass': 'float',
                        'substrates': 'map[positive_float]'
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


def get_particle_comets_state(
        n_bins=(10, 10),
        bounds=(10.0, 10.0),
        mol_ids=None,
        n_particles=10,
        field_diffusion_rate=1e-1,
        field_advection_rate=(0, 0),
        particle_diffusion_rate=1e-1,
        particle_advection_rate=(0, 0),
        particle_add_probability=0.3,
        particle_boundary_to_add=None,
        initial_min_max=None,
):
    particle_boundary_to_add = particle_boundary_to_add or default_config['particle_boundary_to_add']
    mol_ids = mol_ids or default_config['mol_ids']
    initial_min_max = initial_min_max or default_config['initial_min_max']

    # make the composite state with dFBA based on grid size
    composite_state = get_spatial_dfba_state(
        n_bins=n_bins,
        mol_ids=mol_ids,
        initial_min_max=initial_min_max
    )
    # add diffusion/advection process
    composite_state['diffusion'] = get_diffusion_advection_process_state(
        bounds=bounds,
        n_bins=n_bins,
        mol_ids=mol_ids,
        default_diffusion_rate=field_diffusion_rate,
        default_advection_rate=field_advection_rate,
        diffusion_coeffs=None,  #TODO -- add diffusion coeffs config
        advection_coeffs=None,
    )

    # initialize fields
    fields = {}
    for field, minmax in initial_min_max.items():
        fields[field] = np.random.uniform(low=minmax[0], high=minmax[1], size=n_bins)

    # add particles process
    particles = Particles.generate_state(
        config={
            'n_particles': n_particles,
            'bounds': bounds,
            'fields': fields,
            'n_bins': n_bins,
        })

    composite_state['particles'] = particles['particles']
    composite_state['particles_process'] = get_particle_movement_process_state(
        n_bins=n_bins,
        bounds=bounds,
        diffusion_rate=particle_diffusion_rate,
        advection_rate=particle_advection_rate,
        add_probability=particle_add_probability,
        boundary_to_add=particle_boundary_to_add,
    )

    return composite_state


def get_particles_dfba_state(
        core,
        n_bins=(10, 10),
        bounds=(10.0, 10.0),
        mol_ids=None,
        n_particles=10,
        field_diffusion_rate=1e-1,
        field_advection_rate=(0, 0),
        particle_diffusion_rate=1e-1,
        particle_advection_rate=(0, 0),
        particle_add_probability=0.3,
        particle_boundary_to_add=None,
        particle_field_interactions=None,
        initial_min_max=None,
):
    particle_boundary_to_add = particle_boundary_to_add or default_config['particle_boundary_to_add']
    mol_ids = mol_ids or default_config['mol_ids']
    particle_field_interactions = particle_field_interactions or default_config['particle_field_interactions']
    initial_min_max = initial_min_max or default_config['initial_min_max']
    for mol_id in particle_field_interactions.keys():
        if mol_id not in mol_ids:
            mol_ids.append(mol_id)
        if mol_id not in initial_min_max:
            initial_min_max[mol_id] = (0, 1)

    # TODO -- add fields?
    composite_state = {}

    # add diffusion/advection process
    composite_state['diffusion'] = get_diffusion_advection_process_state(
        bounds=bounds,
        n_bins=n_bins,
        mol_ids=mol_ids,
        default_diffusion_rate=field_diffusion_rate,
        default_advection_rate=field_advection_rate,
        diffusion_coeffs=None,  #TODO -- add diffusion coeffs config
        advection_coeffs=None,
    )
    # initialize fields
    fields = {}
    for field, minmax in initial_min_max.items():
        fields[field] = np.random.uniform(low=minmax[0], high=minmax[1], size=n_bins)

    # add particles process
    particles = Particles.generate_state(
        config={
            'n_particles': n_particles,
            'bounds': bounds,
            'fields': fields,
            'n_bins': n_bins,
        })

    composite_state['fields'] = fields
    composite_state['particles'] = particles['particles']
    composite_state['particles_process'] = get_particle_movement_process_state(
        n_bins=n_bins,
        bounds=bounds,
        diffusion_rate=particle_diffusion_rate,
        advection_rate=particle_advection_rate,
        add_probability=particle_add_probability,
        boundary_to_add=particle_boundary_to_add,
    )

    return composite_state



def get_dfba_particle_composition(core=None, config=None):
    config = config or get_dfba_config()
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
                    # 'inputs': {
                    #     'substrates': ['local']
                    #     # TODO --do we have rewire?
                    #     # 'substrates': {
                    #     #     '_path': ['local'],
                    #     #     'biomass': ['mass']
                    #     # }
                    # },
                    # 'outputs': {
                    #     'substrates': ['exchange']
                    # }
                }
            }
        }
    }
