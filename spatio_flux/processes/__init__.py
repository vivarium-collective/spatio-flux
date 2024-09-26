from spatio_flux.processes.dfba import DynamicFBA
from spatio_flux.processes.diffusion_advection import DiffusionAdvection
from spatio_flux.processes.particles import Particles, MinimalParticle


def register_processes(core):
    core.register_process('DynamicFBA', DynamicFBA)
    core.register_process('DiffusionAdvection', DiffusionAdvection)
    core.register_process('Particles', Particles)
    core.register_process('MinimalParticle', MinimalParticle)

    return core
    
