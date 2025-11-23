from spatio_flux.processes.dfba import (
    DynamicFBA, SpatialDFBA, MODEL_REGISTRY_DFBA, get_dfba_process_from_registry, get_field_names)
from spatio_flux.processes.diffusion_advection import DiffusionAdvection
from spatio_flux.processes.particles import BrownianMovement, DIVISION_MASS_THRESHOLD, ParticleDivision, ParticleExchange
from spatio_flux.processes.kinetics import MonodKinetics
from spatio_flux.processes.pymunk_particles import PymunkParticleMovement, get_newtonian_particles_state

# configs
from spatio_flux.processes.configs import (
    default_config, get_single_dfba_process, get_fields,
    get_spatial_many_dfba_with_fields, get_diffusion_advection_process, get_kinetic_particle_composition,
    get_dfba_particle_composition, get_spatial_dFBA_process, get_fields_with_schema, get_spatial_many_dfba, get_particles_state,
    get_brownian_movement_process, get_particle_exchange_process, initialize_fields, get_particle_divide_process,
)

from process_bigraph.emitter import RAMEmitter

PROCESS_DICT = {
    'ram-emitter': RAMEmitter,
    'DynamicFBA': DynamicFBA,
    'SpatialDFBA': SpatialDFBA,
    'DiffusionAdvection': DiffusionAdvection,
    'BrownianMovement': BrownianMovement,
    'PymunkParticleMovement': PymunkParticleMovement,
    'MonodKinetics': MonodKinetics,
    'ParticleDivision': ParticleDivision,
    'ParticleExchange': ParticleExchange,
}


def register_processes(core):
    for process_name, process in PROCESS_DICT.items():
        core.register_process(process_name, process)
    return core


def get_dfba_single_doc(core=None, config=None):
    return {
        'dFBA': {
            '_type': 'process',
            'address': 'local:DynamicFBA',
            'config': {'model_file': 'textbook'},
        }
    }


def get_spatial_dfba_doc(core=None, config=None):
    return {
        'spatial_dFBA': {
            '_type': 'process',
            'address': 'local:SpatialDFBA',
            'config': {'n_bins': (5, 10),  'model_file': 'textbook'},
        }
    }


def get_diffusion_advection_doc(core=None, config=None):
    return {
        'diffusion': {
            '_type': 'process',
            'address': 'local:DiffusionAdvection',
            'config': {
                'n_bins': (5, 10),
                'bounds': (5.0, 10.0),
                'default_diffusion_rate': 1e-1,
                'default_diffusion_dt': 1e-1,
                'diffusion_coeffs': {},
                'advection_coeffs': {},
            },
        }
    }


def get_particles_doc(core=None, config=None):
    return {
        'brownian_movement': {
            '_type': 'process',
            'address': 'local:BrownianMovement',
            'config': {
                'n_bins': (5, 10),
                'bounds': (5.0, 10.0),
                'diffusion_rate': 1e-1,
                'advection_rate': (0.0, -0.1),
                'add_probability': 0.0,
            },
        },
        'particle_exchange': {
            '_type': 'process',
            'address': 'local:ParticleExchange',
            'config': {
                'n_bins': (5, 10),
                'bounds': (5.0, 10.0),
            },
        }
    }


def get_minimal_kinetic_doc(core=None, config=None):
    return {
        'monod_kinetics': {
            '_type': 'process',
            'address': 'local:MonodKinetics',
            'config': {},
        }
    }


def get_division_doc(core=None, config=None):
    return {
        'particle_division': {
            '_type': 'process',
            'address': 'local:ParticleDivision',
            'config': {
                'mass_threshold': 2.0,
            },
        }
    }


def get_newtonian_particles_doc(core=None, config=None):
    return {
        'newtonian_particles': {
            '_type': 'process',
            'address': 'local:PymunkParticleMovement',
            'config': {
                'n_bins': (5, 10),
                'bounds': (5.0, 10.0),
                'gravity': (0.0, -9.81),
                'time_step': 0.01,
            },
        }
    }

PROCESS_DOCS = {
    'dfba': get_dfba_single_doc,
    'spatial_dfba': get_spatial_dfba_doc,
    'diffusion_advection': get_diffusion_advection_doc,
    'minimal_kinetic': get_minimal_kinetic_doc,
    'particle_movement': get_particles_doc,
    'particle_division': get_division_doc,
    'newtonian_particles': get_newtonian_particles_doc,
}
