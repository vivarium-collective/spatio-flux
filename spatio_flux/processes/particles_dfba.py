"""
Particle-COMETS composite made of diffusion-advection and particle processes, with a dFBA within each particle.
"""
import numpy as np
from process_bigraph import Composite, default
from spatio_flux.viz.plot import plot_time_series, plot_species_distributions_with_particles_to_gif


# TODO -- need to do this to register???
from spatio_flux.processes.dfba import DynamicFBA, dfba_config, get_spatial_dfba_state
from spatio_flux.processes.diffusion_advection import DiffusionAdvection, get_diffusion_advection_spec
from spatio_flux.processes.particles import Particles, get_particles_spec, get_particles_state
from spatio_flux.processes.particle_comets import default_config


# default_config = {
#     'total_time': 10.0,
#     # environment size
#     'bounds': (10.0, 20.0),
#     'n_bins': (8, 16),
#     # set fields
#     'mol_ids': ['biomass', 'detritus'],
#     'field_diffusion_rate': 1e-1,
#     'field_advection_rate': (0, 0),
#     'initial_min_max': {'biomass': (0, 0.1), 'detritus': (0, 0)},
#     # set particles
#     'n_particles': 10,
#     'particle_diffusion_rate': 1e-1,
#     'particle_advection_rate': (0, -0.1),
#     'particle_add_probability': 0.3,
#     'particle_boundary_to_add': ['top'],
#     'field_interactions': {
#         'biomass': {'vmax': 0.1, 'Km': 1.0, 'interaction_type': 'uptake'},
#         'detritus': {'vmax': -0.1, 'Km': 1.0, 'interaction_type': 'secretion'},
#     },
# }


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
        field_interactions=None,
        initial_min_max=None,
):
    particle_boundary_to_add = particle_boundary_to_add or default_config['particle_boundary_to_add']
    mol_ids = mol_ids or default_config['mol_ids']
    field_interactions = field_interactions or default_config['field_interactions']
    initial_min_max = initial_min_max or default_config['initial_min_max']
    for mol_id in field_interactions.keys():
        if mol_id not in mol_ids:
            mol_ids.append(mol_id)
        if mol_id not in initial_min_max:
            initial_min_max[mol_id] = (0, 1)

    # TODO -- add fields?
    composite_state = {}

    # add diffusion/advection process
    composite_state['diffusion'] = get_diffusion_advection_spec(
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
    composite_state['particles_process'] = get_particles_spec(
        n_bins=n_bins,
        bounds=bounds,
        diffusion_rate=particle_diffusion_rate,
        advection_rate=particle_advection_rate,
        add_probability=particle_add_probability,
        boundary_to_add=particle_boundary_to_add,
    )

    return composite_state
