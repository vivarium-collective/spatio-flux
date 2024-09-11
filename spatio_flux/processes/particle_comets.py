"""
Particle-COMETS composite made of dFBAs, diffusion-advection, and particle processes.
"""

from process_bigraph import Composite
from spatio_flux import core
from spatio_flux.viz.plot import plot_time_series, plot_species_distributions_with_particles_to_gif


# TODO -- need to do this to register???
from spatio_flux.processes.dfba import DynamicFBA, get_spatial_dfba_state
from spatio_flux.processes.diffusion_advection import DiffusionAdvection, get_diffusion_advection_spec
from spatio_flux.processes.particles import Particles, get_particles_spec, get_particles_state


def get_particle_comets_state(
        n_bins=(10, 10),
        bounds=(10.0, 10.0),
        mol_ids=None,
        initial_max=None,
        n_particles=10,
        field_diffusion_rate=1e-1,
        field_advection_rate=(0, 0),
        particle_diffusion_rate=1e-1,
        particle_advection_rate=(0, 0),
        particle_add_probability=0.3,
        particle_boundary_to_add=None,
):
    if particle_boundary_to_add is None:
        particle_boundary_to_add = ['top']
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']

        # make the composite state with dFBA based on grid size
    composite_state = get_spatial_dfba_state(
        n_bins=n_bins,
        mol_ids=mol_ids,
        initial_max={
            'glucose': 20,
            'acetate': 0,
            'biomass': 0.1
        }
    )
    # add diffusion/advection process
    composite_state['diffusion'] = get_diffusion_advection_spec(
        bounds=bounds,
        n_bins=n_bins,
        mol_ids=mol_ids,
        default_diffusion_rate=field_diffusion_rate,
        default_advection_rate=field_advection_rate,
        diffusion_coeffs=None,
        advection_coeffs=None,
    )
    # add particles process
    particles = Particles.initialize_particles(
        n_particles=n_particles,
        bounds=bounds,
    )
    composite_state['particles'] = particles
    composite_state['particles_process'] = get_particles_spec(
        n_bins=n_bins,
        bounds=bounds,
        diffusion_rate=particle_diffusion_rate,
        advection_rate=particle_advection_rate,
        add_probability=particle_add_probability,
        boundary_to_add=particle_boundary_to_add,
    )
    return composite_state


def run_particle_comets(
        total_time=100.0,
        bounds=(10.0, 20.0),
        n_bins=(8, 16),
        mol_ids=None,
        field_diffusion_rate=1e-1,
        field_advection_rate=(0, 0),
        n_particles=10,
        particle_diffusion_rate=1e-1,
        particle_advection_rate=(0, -0.1),
        particle_add_probability=0.3,
        particle_boundary_to_add=None,
):
    # make the composite state
    if particle_boundary_to_add is None:
        particle_boundary_to_add = ['top']

    composite_state = get_particle_comets_state(
        n_bins=n_bins,
        bounds=bounds,
        # set fields
        mol_ids=mol_ids,
        field_diffusion_rate=field_diffusion_rate,
        field_advection_rate=field_advection_rate,
        # set particles
        n_particles=n_particles,
        particle_diffusion_rate=particle_diffusion_rate,
        particle_advection_rate=particle_advection_rate,
        particle_add_probability=particle_add_probability,
        particle_boundary_to_add=particle_boundary_to_add,
    )

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'},
    }, core=core)

    # save the document
    sim.save(filename='particle_comets.json', outdir='out', include_schema=True)

    # TODO -- save a viz figure of the initial state

    # simulate
    print('Simulating...')
    sim.update({}, total_time)
    particle_comets_results = sim.gather_results()
    # print(comets_results)

    print('Plotting results...')
    # plot timeseries
    plot_time_series(
        particle_comets_results,
        coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)],
        out_dir='out',
        filename='particle_comets_timeseries.png'
    )

    plot_species_distributions_with_particles_to_gif(
        particle_comets_results,
        out_dir='out',
        filename='particle_comets_with_fields.gif',
        title='',
        skip_frames=1,
        bounds=bounds,
    )


if __name__ == '__main__':
    run_particle_comets()
