from bigraph_viz import plot_bigraph
from process_bigraph import Composite, default
from vivarium.vivarium import VivariumTypes

from spatio_flux import register_types
from spatio_flux.viz.plot import (
    plot_time_series,
    plot_species_distributions_to_gif,
    plot_species_distributions_with_particles_to_gif,
    plot_particles
)
from spatio_flux.processes.dfba import get_single_dfba_spec, get_spatial_dfba_state, dfba_config
from spatio_flux.processes.diffusion_advection import get_diffusion_advection_spec, get_diffusion_advection_state
from spatio_flux.processes.particles import MinimalParticle, get_particles_state
from spatio_flux.processes.particle_comets import get_particle_comets_state, default_config
from spatio_flux.processes.particles_dfba import get_particles_dfba_state, default_config


def run_dfba_single(
        total_time=60,
        mol_ids=None,
        core=None,
):
    single_dfba_config = {
        'dfba': get_single_dfba_spec(path=['fields']),
        'fields': {
            'glucose': 10,
            'acetate': 0,
            'biomass': 0.1
        }
    }

    # make the simulation
    sim = Composite({
        'state': single_dfba_config,
        'emitter': {'mode': 'all'}
    }, core=core)

    # save the document
    sim.save(filename='single_dfba.json', outdir='out')

    # simulate
    print('Simulating...')
    sim.update({}, total_time)

    # gather results
    dfba_results = sim.gather_results()

    print('Plotting results...')
    # plot timeseries
    plot_time_series(
        dfba_results,
        # coordinates=[(0, 0), (1, 1), (2, 2)],
        out_dir='out',
        filename='dfba_single_timeseries.png',
    )


def run_dfba_spatial(
        total_time=60,
        n_bins=(3, 3),  # TODO -- why can't do (5, 10)??
        mol_ids=None,
        core=None
):
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']
    composite_state = get_spatial_dfba_state(
        n_bins=n_bins,
        mol_ids=mol_ids,
    )

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'}
    }, core=core)

    # save the document
    sim.save(filename='spatial_dfba.json', outdir='out')

    # # save a viz figure of the initial state
    # plot_bigraph(
    #     state=sim.state,
    #     schema=sim.composition,
    #     core=core,
    #     out_dir='out',
    #     filename='dfba_spatial_viz'
    # )

    # simulate
    print('Simulating...')
    sim.update({}, total_time)

    # gather results
    dfba_results = sim.gather_results()

    print('Plotting results...')
    # plot timeseries
    plot_time_series(
        dfba_results,
        coordinates=[(0, 0), (1, 1), (2, 2)],
        out_dir='out',
        filename='spatial_dfba_timeseries.png',
    )

    # make video
    plot_species_distributions_to_gif(
        dfba_results,
        out_dir='out',
        filename='spatial_dfba_results.gif',
        title='',
        skip_frames=1
    )


def run_diffusion_process(
        total_time=60,
        bounds=(10.0, 10.0),
        n_bins=(10, 10),
        core=None,
):
    composite_state = get_diffusion_advection_state(
        bounds=bounds,
        n_bins=n_bins,
        mol_ids=['glucose', 'acetate', 'biomass'],
        advection_coeffs={
            'biomass': (0, -0.1)
        }
    )

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'},
    }, core=core)

    # save the document
    sim.save(filename='diffadv.json', outdir='out')

    # save a viz figure of the initial state
    plot_bigraph(
        state=sim.state,
        schema=sim.composition,
        core=core,
        out_dir='out',
        filename='diffadv_viz'
    )

    # simulate
    print('Simulating...')
    sim.update({}, total_time)

    # gather results
    diffadv_results = sim.gather_results()

    print('Plotting results...')
    # plot 2d video
    plot_species_distributions_to_gif(
        diffadv_results,
        out_dir='out',
        filename='diffadv_results.gif',
        title='',
        skip_frames=1
    )


def run_particles(
        core,
        total_time=60,  # Total frames
        bounds=(10.0, 20.0),  # Bounds of the environment
        n_bins=(20, 40),  # Number of bins in the x and y directions
        n_particles=1,  # 20
        diffusion_rate=0.1,
        advection_rate=(0, -0.1),
        add_probability=0.4,
        field_interactions=None,
        initial_min_max=None,
):
    # Get all local variables as a dictionary
    kwargs = locals()
    kwargs.pop('total_time')  # 'total_time' is only used here, so we pop it

    # initialize particles state
    composite_state = get_particles_state(**kwargs)

    # TODO -- is this how to link in the minimal_particle process?
    # declare minimal particle in the composition
    composition = {
        'particles': {
            '_type': 'map',
            '_value': {
                # '_inherit': 'particle',
                'minimal_particle': {
                    '_type': 'process',
                    'address': default('string', 'local:MinimalParticle'),


                    # TODO: test to see if we only need to provide the default value
                    #   in the process composition
                    # {'_default': 'local:MinimalParticle'}


                    'config': default('quote', core.default(MinimalParticle.config_schema)),
                    'inputs': default('tree[wires]', {'substrates': ['local']}),  # TODO -- what sets this??? Particles
                    'outputs': default('tree[wires]', {'substrates': ['exchange']})
                }
            }
        }
    }

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
        'composition': composition,
        'emitter': {'mode': 'all'},
    }, core=core)

    # save the document
    sim.save(
        filename='particles.json',
        outdir='out')

    # save a viz figure of the initial state
    plot_bigraph(
        state=sim.state,
        schema=sim.composition,
        core=core,
        out_dir='out',
        filename='particles_viz'
    )

    # simulate
    print('Simulating...')
    sim.update({}, total_time)

    # gather results
    particles_results = sim.gather_results()
    emitter_results = particles_results[('emitter',)]
    # resort results
    particles_history = [p['particles'] for p in emitter_results]

    print('Plotting...')
    # plot particles
    plot_particles(
        # total_time=total_time,
        history=particles_history,
        env_size=((0, bounds[0]), (0, bounds[1])),
        out_dir='out',
        filename='particles.gif',
    )

    plot_species_distributions_with_particles_to_gif(
        particles_results,
        out_dir='out',
        filename='particle_with_fields.gif',
        title='',
        skip_frames=1,
        bounds=bounds,
    )


def run_particle_comets(
        core,
        total_time=50.0,
        **kwargs
):
    # make the composite state
    composite_state = get_particle_comets_state(**kwargs)

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'},
    }, core=core)

    # save the document
    sim.save(
        filename='particle_comets.json',
        outdir='out')

    # # save a viz figure of the initial state
    # plot_bigraph(
    #     state=sim.state,
    #     schema=sim.composition,
    #     core=core,
    #     out_dir='out',
    #     filename='particles_comets_viz'
    # )

    # simulate
    print('Simulating...')
    sim.update({}, total_time)
    particle_comets_results = sim.gather_results()
    # print(comets_results)

    print('Plotting results...')
    n_bins = composite_state['particles_process']['config']['n_bins']
    bounds = composite_state['particles_process']['config']['bounds']

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


def run_particles_dfba(
    core,
    total_time=10.0,
    n_bins=None,
    bounds=None):

    # make the composite state
    composite_state = get_particles_dfba_state(core)

    composition = {
        'particles': {
            '_type': 'map',
            '_value': {
                'dFBA': {
                    '_type': 'process',
                    'address': default('string', 'local:DynamicFBA'),
                    'config': default('quote', dfba_config(model_file='textbook')),
                    'inputs': default('tree[wires]', {'substrates': ['local']}),
                    'outputs': default('tree[wires]', {'substrates': ['exchange']})
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

    # save the document
    sim.save(
        filename='particle_comets.json',
        outdir='out')

    # TODO -- save a viz figure of the initial state

    # simulate
    print('Simulating...')
    sim.update({}, total_time)
    particle_comets_results = sim.gather_results()

    print('Plotting results...')
    n_bins = composite_state['particles_process']['config']['n_bins']
    bounds = composite_state['particles_process']['config']['bounds']

    # plot timeseries
    plot_time_series(
        particle_comets_results,
        coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)],
        out_dir='out',
        filename='particle_dfba_timeseries.png'
    )

    plot_species_distributions_with_particles_to_gif(
        particle_comets_results,
        out_dir='out',
        filename='particle_dfba_with_fields.gif',
        title='',
        skip_frames=1,
        bounds=bounds,
    )



if __name__ == '__main__':
    core = VivariumTypes()
    core = register_types(core)

    # run_dfba_single(core=core)
    # run_dfba_spatial(core=core, n_bins=(4,4), total_time=60)
    # run_diffusion_process(core=core)
    run_particles(core)
    run_particle_comets(core)
    run_particles_dfba(core)
