from spatio_flux.processes.dfba import DynamicFBA, SpatialDFBA
from spatio_flux.processes.diffusion_advection import DiffusionAdvection
from spatio_flux.processes.particles import Particles, MinimalParticle

# configs
from spatio_flux.processes.configs import (
    get_particle_dfba_state, default_config, get_particle_comets_state, get_single_dfba_process, get_fields,
    get_spatial_many_dfba_with_fields, get_diffusion_advection_process, get_minimal_particle_composition,
    get_dfba_particle_composition, get_spatial_dfba_process, get_fields_with_schema, get_spatial_many_dfba, get_particles_state,
    get_particle_movement_process, initialize_fields)

PROCESS_DICT = {
    'DynamicFBA': DynamicFBA,
    'SpatialDFBA': SpatialDFBA,
    'DiffusionAdvection': DiffusionAdvection,
    'Particles': Particles,
    'MinimalParticle': MinimalParticle,
}


def register_processes(core):
    for process_name, process in PROCESS_DICT.items():
        core.register_process(process_name, process)
    return core
    
