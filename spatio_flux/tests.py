from datetime import datetime

from bigraph_viz import plot_bigraph
from process_bigraph import Composite, default, register_types as register_process_types
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results
from vivarium.vivarium import VivariumTypes

from spatio_flux import register_types
from spatio_flux.library.functions import initialize_fields
from spatio_flux.viz.plot import (
    plot_time_series,
    plot_species_distributions_to_gif,
    plot_species_distributions_with_particles_to_gif,
    plot_particles
)
from spatio_flux.processes.dfba import get_single_dfba_spec, get_spatial_dfba_state, dfba_config
from spatio_flux.processes.diffusion_advection import get_diffusion_advection_spec, get_diffusion_advection_state
from spatio_flux.processes.particles import (
    MinimalParticle, get_particles_state, get_minimal_particle_composition, get_dfba_particle_composition)
from spatio_flux.processes.particle_comets import get_particle_comets_state, default_config
from spatio_flux.processes.particles_dfba import get_particles_dfba_state, default_config


# =====================
# Utility Functions
# =====================

def get_standard_emitter():
    """
    Returns a standard emitter specification for capturing global time and fields.
    """
    return emitter_from_wires({
        'global_time': ['global_time'],
        'fields': ['fields'],
        'particles': ['particles'],
    })


def run_composite_document(document, time=None, core=None, name=None):
    """
    Instantiates and runs a Composite simulation.

    Args:
        document (dict): Composition document with initial state and optional schema.
        time (float): Simulation duration.
        core (VivariumTypes): Core schema registration object.
        name (str): Output name prefix.

    Returns:
        dict: Simulation results emitted during the run.
    """
    time = time or 60
    if core is None:
        core = VivariumTypes()
        core = register_types(core)
    if name is None:
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'spatio_flux_{date}'

    document = {'state': document} if 'state' not in document else document
    if 'emitter' not in document['state']:
        document['state']['emitter'] = get_standard_emitter()

    print('Making the composite...')
    sim = Composite(document, core=core)

    # Save composition JSON
    sim.save(filename=f'{name}.json', outdir='out')

    # Save visualization of the initial composition
    plot_bigraph(
        state=sim.state,
        schema=sim.composition,
        core=core,
        out_dir='out',
        filename=f'{name}_viz'
    )

    print('Simulating...')
    sim.run(time)
    results = gather_emitter_results(sim)
    return results[('emitter',)]


# =====================
# Simulation Use Cases
# =====================

def run_dfba_single(total_time=60, core=None):
    mol_ids = ["glucose", "acetate", "biomass"]
    doc = {
        'dfba': get_single_dfba_spec(
            path=['fields'],
            mol_ids=mol_ids,
        ),
        'fields': {
            'glucose': 10,
            'acetate': 0,
            'biomass': 0.1
        },
    }

    # run the composite document
    results = run_composite_document(doc, time=total_time, core=core, name='dfba_single')

    # plotting
    plot_time_series(results, out_dir='out', filename='dfba_single_timeseries.png')


def run_dfba_spatial(total_time=60, core=None):
    n_bins = (5, 5)
    mol_ids = ['glucose', 'acetate', 'biomass']
    initial_min_max = {"glucose": (0, 20), "acetate": (0, 0), "biomass": (0, 0.1)}
    state = get_spatial_dfba_state(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max)

    # run the composite document
    results = run_composite_document(document=state, time=total_time, core=core, name='spatial_dfba')

    # plotting
    plot_time_series(results, coordinates=[(0, 0), (1, 1), (2, 2)], out_dir='out', filename='spatial_dfba_timeseries.png')
    plot_species_distributions_to_gif(results, out_dir='out', filename='spatial_dfba_results.gif')


def run_diffusion_process(total_time=60, core=None):
    bounds, n_bins = (10.0, 10.0), (10, 10)
    mol_ids = ['glucose', 'acetate', 'biomass']
    advection_coeffs = {'biomass': (0, -0.1)}
    state = get_diffusion_advection_state(
        bounds=bounds, n_bins=n_bins,
        mol_ids=mol_ids,
        advection_coeffs=advection_coeffs
    )

    # run the composite document
    results = run_composite_document(state, time=total_time, core=core, name='diff_advec')

    # plotting
    plot_species_distributions_to_gif(results, out_dir='out', filename='diff_advec_results.gif')


def run_particles(total_time=60, core=None):
    bounds = (10.0, 20.0)
    initial_min_max = {
        'biomass': (0.5, 2.0),
        'detritus': (0, 0),
    }
    state = get_particles_state(
        bounds=bounds, n_bins=(10, 20),
        n_particles=1, diffusion_rate=0.1, advection_rate=(0, -0.1), add_probability=0.4,
        initial_min_max=initial_min_max
    )
    doc = {
        'state': state,
        'composition': get_minimal_particle_composition(core)
    }

    # run the composite document
    results = run_composite_document(doc, time=total_time, core=core, name='particles')

    # plotting
    history = [step['particles'] for step in results]
    plot_particles(history=history, env_size=((0, bounds[0]), (0, bounds[1])), out_dir='out', filename='particles.gif')
    plot_species_distributions_with_particles_to_gif(results, out_dir='out', filename='particle_with_fields.gif', bounds=bounds)


def run_comets(total_time=60, core=None):
    bounds, n_bins = (10.0, 10.0), (10, 10)
    mol_ids = default_config['mol_ids']
    initial_min_max = default_config['initial_min_max']
    state = get_spatial_dfba_state(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max)
    state['diffusion'] = get_diffusion_advection_spec(bounds, n_bins, mol_ids)

    # run the composite document
    results = run_composite_document(state, time=total_time, core=core, name='comets')

    # plotting
    plot_time_series(results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename='comets_timeseries.png')
    plot_species_distributions_to_gif(results, out_dir='out', filename='comets_results.gif')


def run_particle_comets(total_time=60, core=None):
    state = get_particle_comets_state()
    doc = {'composition': get_minimal_particle_composition(core), 'state': state}

    # run the composite document
    results = run_composite_document(doc, time=total_time, core=core, name='particle_comets')

    # plotting
    n_bins, bounds = state['particles_process']['config']['n_bins'], state['particles_process']['config']['bounds']
    plot_time_series(results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename='particle_comets_timeseries.png')
    plot_species_distributions_with_particles_to_gif(results, out_dir='out', filename='particle_comets_with_fields.gif', bounds=bounds)


def run_particles_dfba(total_time=60, core=None):
    mol_ids = default_config['mol_ids']
    state = get_particles_dfba_state(
        core,
        mol_ids=mol_ids)
    doc = {
        'state': state,
        'composition': get_dfba_particle_composition(),
    }

    # run the composite document
    results = run_composite_document(doc, time=total_time, core=core, name='particles_dfba')

    # plotting
    n_bins, bounds = state['particles_process']['config']['n_bins'], state['particles_process']['config']['bounds']
    plot_time_series(results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename='particle_dfba_timeseries.png')
    plot_species_distributions_with_particles_to_gif(results, out_dir='out', filename='particle_dfba_with_fields.gif', bounds=bounds)


if __name__ == '__main__':
    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    run_dfba_single(core=core)
    run_dfba_spatial(core=core)
    run_diffusion_process(core=core)
    run_particles(core=core)
    run_comets(core=core)
    run_particle_comets(core=core)
    run_particles_dfba(core=core)
