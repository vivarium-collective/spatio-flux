"""
Particle-COMETS composite made of diffusion-advection and particle processes, with a dFBA within each particle.
"""

from process_bigraph import Composite, default
from spatio_flux import core
from spatio_flux.viz.plot import plot_time_series, plot_species_distributions_with_particles_to_gif


# TODO -- need to do this to register???
from spatio_flux.processes.dfba import DynamicFBA, dfba_config, get_spatial_dfba_state
from spatio_flux.processes.diffusion_advection import DiffusionAdvection, get_diffusion_advection_spec
from spatio_flux.processes.particles import Particles, get_particles_spec, get_particles_state


default_config = {
    'total_time': 10.0,
    # environment size
    'bounds': (10.0, 20.0),
    'n_bins': (8, 16),
    # set fields
    'mol_ids': ['biomass', 'detritus'],
    'field_diffusion_rate': 1e-1,
    'field_advection_rate': (0, 0),
    'initial_min_max': {'biomass': (0, 0.1), 'detritus': (0, 0)},
    # set particles
    'n_particles': 10,
    'particle_diffusion_rate': 1e-1,
    'particle_advection_rate': (0, -0.1),
    'particle_add_probability': 0.3,
    'particle_boundary_to_add': ['top'],
    'field_interactions': {
        'biomass': {'vmax': 0.1, 'Km': 1.0, 'interaction_type': 'uptake'},
        'detritus': {'vmax': -0.1, 'Km': 1.0, 'interaction_type': 'secretion'},
    },
}


def get_particle_dfba_state(
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
        field_interactions=field_interactions,
    )
    return composite_state


def run_particle_dfba(
        total_time=10.0,
        **kwargs
):
    # make the composite state
    composite_state = get_particle_dfba_state(**kwargs)

    composition = {
        'particles': {
            '_type': 'list',
            '_element': {
                'dFBA': {
                    '_type': 'process',
                    'address': default('string', 'local:DynamicFBA'),
                    'config': default('tree[any]', dfba_config(model_file='textbook')),
                    'inputs': default('tree[wires]', {'substrates': ['local']}),
                    'outputs': default('tree[wires]', {'substrates': ['local']})
                }
            }
        }
    }

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'composition': composition,
        'state': composite_state,
        'emitter': {'mode': 'all'},
    }, core=core)

    import ipdb; ipdb.set_trace()

    # save the document
    sim.save(filename='particle_comets.json', outdir='out', include_schema=True)

    # TODO -- save a viz figure of the initial state

    # simulate
    print('Simulating...')
    sim.update({}, total_time)
    particle_comets_results = sim.gather_results()
    # print(comets_results)

    print('Plotting results...')
    n_bins = kwargs.get('n_bins', default_config['n_bins'])
    bounds = kwargs.get('bounds', default_config['bounds'])
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
    run_particle_dfba(**default_config)
