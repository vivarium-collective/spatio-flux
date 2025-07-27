from spatio_flux.processes.dfba import DynamicFBA
from spatio_flux.processes.diffusion_advection import DiffusionAdvection
from spatio_flux.processes.particles import Particles, MinimalParticle

# configs
from spatio_flux.processes.dfba import get_single_dfba_spec, get_spatial_dfba_state
from spatio_flux.processes.diffusion_advection import get_diffusion_advection_spec, get_diffusion_advection_state
from spatio_flux.processes.particles import (
    get_particles_state, get_minimal_particle_composition, get_dfba_particle_composition)
from spatio_flux.processes.configs import get_particles_dfba_state, default_config, get_particle_comets_state


PROCESS_DICT = {
    'DynamicFBA': DynamicFBA,
    'DiffusionAdvection': DiffusionAdvection,
    'Particles': Particles,
    'MinimalParticle': MinimalParticle,
}


def register_processes(core):
    for process_name, process in PROCESS_DICT.items():
        core.register_process(process_name, process)
    return core
    
