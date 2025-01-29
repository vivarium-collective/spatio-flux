from spatio_flux.processes.dfba import DynamicFBA
from spatio_flux.processes.diffusion_advection import DiffusionAdvection
from spatio_flux.processes.particles import Particles, MinimalParticle


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
    
