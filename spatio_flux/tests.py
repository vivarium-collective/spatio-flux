from bigraph_viz import plot_bigraph
from process_bigraph import Composite, default
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results
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
from spatio_flux.processes.particles import MinimalParticle, get_particles_state, get_minimal_particle_composition
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
        },
        'emitter': emitter_from_wires({
            'global_time': ['global_time'],
            'fields': ['fields']}),
    }

    # make the simulation
    sim = Composite({
        'state': single_dfba_config,
    }, core=core)

    # save the document
    sim.save(filename='single_dfba.json', outdir='out')

    # simulate
    print('Simulating...')
    sim.update({}, total_time)

    # gather results
    dfba_results = gather_emitter_results(sim)

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

    composite_state['emitter'] = emitter_from_wires({
        'global_time': ['global_time'],
        'fields': ['fields']})

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
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
    dfba_results = gather_emitter_results(sim)

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

    composite_state['emitter'] = emitter_from_wires({
        'global_time': ['global_time'],
        'fields': ['fields']})

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
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
    diffadv_results = gather_emitter_results(sim)

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
    composition = get_minimal_particle_composition(core)# {
    #     'particles': {
    #         '_type': 'map',
    #         '_value': {
    #             # '_inherit': 'particle',
    #             'minimal_particle': {
    #                 '_type': 'process',
    #                 'address': default('string', 'local:MinimalParticle'),


    #                 # TODO: run to see if we only need to provide the default value
    #                 #   in the process composition
    #                 # {'_default': 'local:MinimalParticle'}


    #                 'config': default('quote', core.default(MinimalParticle.config_schema)),
    #                 'inputs': default(
    #                     'tree[wires]', {
    #                         'mass': ['mass'],
    #                         'substrates': ['local']}),
    #                 'outputs': default(
    #                     'tree[wires]', {
    #                         'mass': ['mass'],
    #                         'substrates': ['exchange']})
    #             }
    #         }
    #     }
    # }

    composite_state['emitter'] = emitter_from_wires({
        'global_time': ['global_time'],
        'particles': ['particles'],
        'fields': ['fields']})

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
        'composition': composition,
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
    particles_results = gather_emitter_results(sim)
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


def run_comets(
        core,
        total_time=100.0,
        bounds=(10.0, 10.0),
        n_bins=(10, 10),
        mol_ids=None,
        initial_min_max=None,
):
    mol_ids = mol_ids or default_config['mol_ids']
    initial_min_max = initial_min_max or default_config['initial_min_max']

    # make the composite state
    composite_state = get_spatial_dfba_state(
        n_bins=n_bins,
        mol_ids=mol_ids,
        initial_min_max=initial_min_max,
    )
    composite_state['diffusion'] = get_diffusion_advection_spec(
        bounds=bounds,
        n_bins=n_bins,
        mol_ids=mol_ids,
        default_diffusion_rate=1e-1,
        default_advection_rate=(0, 0),
        diffusion_coeffs=None,  # TODO add all these config options
        advection_coeffs=None,
    )

    composite_state['emitter'] = emitter_from_wires({
        'global_time': ['global_time'],
        'fields': ['fields']})

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
    }, core=core)

    # save the document
    sim.save(filename='comets.json', outdir='out')

    # TODO: FIX VIZ (unexpected wire type?)
    # # save a viz figure of the initial state
    # plot_bigraph(
    #     state=sim.state,
    #     schema=sim.composition,
    #     core=core,
    #     out_dir='out',
    #     filename='comets_viz'
    # )

    # simulate
    print('Simulating...')
    sim.update({}, total_time)
    comets_results = gather_emitter_results(sim)
    # print(comets_results)

    print('Plotting results...')
    # plot timeseries
    plot_time_series(
        comets_results,
        coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)],
        out_dir='out',
        filename='comets_timeseries.png',
    )

    # plot 2d video
    plot_species_distributions_to_gif(
        comets_results,
        out_dir='out',
        filename='comets_results.gif',
        title='',
        skip_frames=1)


def run_particle_comets(
        core,
        total_time=10.0,
        **kwargs
):
    kwargs_dict = dict(kwargs)

    # override mol_ids
    kwargs_dict['mol_ids'] = ['glucose', 'acetate', 'biomass', 'waste']
    kwargs_dict['n_bins'] = (5, 5)
    # n_bins = (10, 10),
    # bounds = (10.0, 10.0),

    # make the composite state
    composite_state = get_particle_comets_state(**kwargs_dict)

    composite_state['emitter'] = emitter_from_wires({
        'global_time': ['global_time'],
        'particles': ['particles'],
        'fields': ['fields']})

    composition = get_minimal_particle_composition(core) # {

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'composition': composition,
        'state': composite_state,
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
    particle_comets_results = gather_emitter_results(sim)
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
    total_time=20.0,
    n_bins=None,
    bounds=None
):

    # make the composite state
    composite_state = get_particles_dfba_state(core)

    composite_state['emitter'] = emitter_from_wires({
        'global_time': ['global_time'],
        'particles': ['particles'],
        'fields': ['fields']})

    # # reduce the diffusion timestep
    # composite_state['diffusion']['interval'] = 0.1

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
        'state': composite_state
    }, core=core)

    # save the document
    sim.save(
        filename='particle_comets.json',
        outdir='out')

    # plot the bigraph
    print('Saving bigraph diagram...')
    plot_bigraph(
        state=sim.state,
        schema=sim.composition,
        core=core,
        out_dir='out',
        filename='particles_dfba_viz',
        dpi = '240',
    )

    # TODO -- save a viz figure of the initial state

    # simulate
    print('Simulating...')
    sim.update({}, total_time)
    particle_comets_results = gather_emitter_results(sim)

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

def run_vivarium_interface():
    from vivarium import Vivarium
    from spatio_flux import PROCESS_DICT, TYPES_DICT

    mol_ids = ["glucose", "acetate", "biomass"]
    path = ["fields"]
    i = 0
    j = 0

    # Function to build the path with optional indices
    def build_path(mol_id):
        base_path = path + [mol_id]
        if i is not None:
            base_path.append(i)
        if j is not None:
            base_path.append(j)
        return base_path

    v = Vivarium(processes=PROCESS_DICT, types=TYPES_DICT)

    v.add_process(name="dFBA",
                  process_id="DynamicFBA",
                  config={
                      "model_file": "textbook",
                      "kinetic_params": {
                          "glucose": (0.5, 1),
                          "acetate": (0.5, 2)},
                      "substrate_update_reactions": {
                          "glucose": "EX_glc__D_e",
                          "acetate": "EX_ac_e"},
                      "biomass_identifier": "biomass",
                      "bounds": {
                          "EX_o2_e": {"lower": -2, "upper": None},
                          "ATPM": {"lower": 1, "upper": 1}
                      }
                  },
                  )

    v.connect_process(
        process_name="dFBA",
        # inputs={
        #         "substrates": ["fields",],  # {mol_id: ['fields', mol_id] for mol_id in mol_ids}
        #     },
        # outputs={
        #         "substrates": ["fields",],  # {mol_id: ['fields', mol_id] for mol_id in mol_ids}
        #     }
        inputs={
            "substrates": {mol_id: build_path(mol_id) for mol_id in mol_ids}
        },
        outputs={
            "substrates": {mol_id: build_path(mol_id) for mol_id in mol_ids}
        }
    )
    v.diagram(dpi='70')


if __name__ == '__main__':
    core = VivariumTypes()
    core = register_types(core)

    # run_dfba_single(core=core)
    # run_dfba_spatial(core=core, n_bins=(4,4), total_time=60)
    # run_diffusion_process(core=core)
    # run_particles(core)
    # run_comets(core=core)
    # run_particle_comets(core)
    run_particles_dfba(core)
    # run_vivarium_interface()
