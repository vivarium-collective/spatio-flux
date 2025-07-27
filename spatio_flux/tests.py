from process_bigraph import default, register_types as register_process_types
from vivarium.vivarium import VivariumTypes

from spatio_flux import register_types
from spatio_flux.library.helpers import run_composite_document
from spatio_flux.viz.plot import (
    plot_time_series,
    plot_particles_mass,
    plot_species_distributions_to_gif,
    plot_species_distributions_with_particles_to_gif,
    plot_particles
)
from spatio_flux.processes import (
    get_dfba_process_state, get_spatial_dfba_state,
    get_diffusion_advection_process_state, get_diffusion_advection_state,
    get_particles_state, get_minimal_particle_composition, get_dfba_particle_composition,
    get_particles_dfba_state, default_config, get_particle_comets_state
)


# =====================
# Simulation Use Cases
# =====================

def run_dfba_single(total_time=60, core=None):
    mol_ids = ["glucose", "acetate", "biomass"]
    doc = {
        'dfba': get_dfba_process_state(
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


def run_comets(total_time=60, core=None):
    bounds, n_bins = (10.0, 10.0), (10, 10)
    mol_ids = ['glucose', 'acetate', 'biomass']
    initial_min_max = {
        'glucose': (10, 10),
        'acetate': (0, 0),
        'biomass': (0, 0.1),
    }
    state = get_spatial_dfba_state(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max)
    state['diffusion'] = get_diffusion_advection_process_state(bounds, n_bins, mol_ids)

    # run the composite document
    results = run_composite_document(state, time=total_time, core=core, name='comets')

    # plotting
    plot_time_series(results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename='comets_timeseries.png')
    plot_species_distributions_to_gif(results, out_dir='out', filename='comets_results.gif')


def run_particles(total_time=60, core=None):
    bounds = (10.0, 20.0)
    initial_min_max = {
        'glucose': (0.5, 2.0),
        'detritus': (0, 0),
    }
    particle_config = {
        'grow': {
            'vmax': 0.01,
            'reactant': 'glucose',
            'product': 'mass',
        },
        'release': {
            'vmax': 0.001,
            'reactant': 'mass',
            'product': 'detritus',
        }
    }
    state = get_particles_state(
        bounds=bounds, n_bins=(10, 20),
        n_particles=1, diffusion_rate=0.1, advection_rate=(0, -0.1), add_probability=0.4,
        initial_min_max=initial_min_max
    )
    doc = {
        'state': state,
        'composition': get_minimal_particle_composition(core=core, config=particle_config)
    }

    # run the composite document
    results = run_composite_document(doc, time=total_time, core=core, name='particles')

    # plotting
    history = [step['particles'] for step in results]
    plot_particles(history=history, env_size=((0, bounds[0]), (0, bounds[1])), out_dir='out', filename='particles.gif')
    plot_species_distributions_with_particles_to_gif(results, out_dir='out', filename='particle_with_fields.gif', bounds=bounds)


def run_particle_comets(total_time=60, core=None):
    particle_config = {
        'grow': {
            'vmax': 0.01,
            'reactant': 'glucose',
            'product': 'mass',
        },
        'release': {
            'vmax': 0.001,
            'reactant': 'mass',
            'product': 'detritus',
        }
    }
    state = get_particle_comets_state(
        mol_ids=['glucose', 'acetate', 'detritus', 'biomass']
    )
    doc = {
        'composition': get_minimal_particle_composition(core, config=particle_config),
        'state': state
    }

    # run the composite document
    results = run_composite_document(doc, time=total_time, core=core, name='particle_comets')

    # plotting
    n_bins, bounds = state['particles_process']['config']['n_bins'], state['particles_process']['config']['bounds']
    plot_time_series(results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename='particle_comets_timeseries.png')
    plot_species_distributions_with_particles_to_gif(results, out_dir='out', filename='particle_comets_with_fields.gif', bounds=bounds)


def run_particles_dfba(total_time=60, core=None):
    state = get_particles_dfba_state(
        core,
        n_particles=2,
        particle_add_probability=0.3,
        particle_boundary_to_add=[
            'top', 'bottom', 'left', 'right'
        ],
        particle_boundary_to_remove=[
            'top', 'bottom', 'left', 'right'
        ],
        mol_ids=['glucose', 'acetate', 'detritus'],
        initial_min_max={
            'glucose': (1, 10),
            'acetate': (0, 0),
            'detritus': (0, 0)
        }
    )
    particle_composition = get_dfba_particle_composition()

    doc = {
        'state': state,
        'composition': particle_composition,
    }

    # run the composite document
    results = run_composite_document(doc, time=total_time, core=core, name='particles_dfba')

    # plotting
    n_bins, bounds = state['particles_process']['config']['n_bins'], state['particles_process']['config']['bounds']
    plot_time_series(
        results, field_names=['glucose', 'acetate', 'detritus'],
        coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename='particle_dfba_timeseries.png')
    plot_particles_mass(
        results, out_dir='out', filename='particle_dfba_mass.png')
    plot_species_distributions_with_particles_to_gif(
        results, out_dir='out', filename='particle_dfba_with_fields.gif', bounds=bounds)


if __name__ == '__main__':
    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    run_dfba_single(core=core)
    run_dfba_spatial(core=core)
    run_diffusion_process(core=core)
    run_comets(core=core)
    run_particles(core=core)
    run_particle_comets(core=core)
    run_particles_dfba(core=core)
