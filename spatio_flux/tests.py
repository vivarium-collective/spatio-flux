"""
Simulation Runner and Visualizer for Spatio-Temporal Flux Processes

This script defines a suite of modular simulation experiments involving dynamic
spatial processes such as diffusion, advection, dynamic flux balance analysis (DFBA),
and particle-based modeling, including hybrid and compositional scenarios. It provides
functionality to generate simulation documents, run them using the Vivarium simulation
framework, and produce a variety of plots and GIFs for visualization.

Each experiment (e.g., 'dfba_single', 'comets', 'particle_dfba') has a corresponding
document generator and plotting routine, allowing flexible execution and output analysis.

Usage:
    python <script_name>.py --tests dfba_single comets --output out/

"""
import argparse
import numpy as np
import time

from process_bigraph import default, register_types as register_process_types
from vivarium.vivarium import VivariumTypes

from spatio_flux import register_types
from spatio_flux.library.helpers import run_composite_document, prepare_output_dir, generate_html_report
from spatio_flux.viz.plot import ( plot_time_series, plot_particles_mass, plot_species_distributions_to_gif,
    plot_species_distributions_with_particles_to_gif, plot_particles
)
from spatio_flux.processes import (
    get_single_dfba_process, get_spatial_many_dfba, get_spatial_dfba_process, get_fields, get_fields_with_schema,
    get_diffusion_advection_process, get_particle_movement_process, initialize_fields, get_minimal_particle_composition,
    get_dfba_particle_composition, get_particles_state
)

DEFAULT_BOUNDS = (5.0, 10.0)
DEFAULT_BINS = (10, 20)
DEFAULT_BINS_SMALL = (5, 10)
DEFAULT_ADVECTION = (0, -0.1)
DEFAULT_DIFFUSION = 0.1
DEFAULT_ADD_PROBABILITY = 0.4
DEFAULT_ADD_BOUNDARY = ['top', 'left', 'right']
DEFAULT_REMOVE_BOUNDARY = ['left', 'right']
DEFAULT_INITIAL_MIN_MAX = {
        'glucose': (10, 10),
        'acetate': (0, 0),
        'biomass': (0, 0.1),
        'detritus': (0, 0)
    }

DEFAULT_RUNTIME = 10

SIMULATION_CONFIGS = {
    'dfba_single': {'time': DEFAULT_RUNTIME},
    'spatial_many_dfba': {'time': DEFAULT_RUNTIME},
    'spatial_dfba_process': {'time': DEFAULT_RUNTIME},
    'diffusion_process': {'time': DEFAULT_RUNTIME},
    'comets': {'time': DEFAULT_RUNTIME},
    'particles': {'time': DEFAULT_RUNTIME},
    'particle_comets': {'time': DEFAULT_RUNTIME},
    'particle_dfba': {'time': DEFAULT_RUNTIME},
}

DESCRIPTIONS = {
    'dfba_single': 'This simulation runs a single dFBA (dynamic Flux Balance Analysis) process, tracking external concentrations and biomass.',
    'spatial_many_dfba': 'This simulation introduces a spatial lattice, with a single dFBA process in each lattice site.',
    'spatial_dfba_process': 'This simulation introduces a spatial lattice, with a spatial dFBA process that runs all the lattice sites',
    'diffusion_process': 'This simulation includes finite volume method for diffusion and advection on a lattice.',
    'comets': 'This simulation combines dFBA at each lattice site with diffusion/advection to make a spatio-temporal FBA.',
    'particles': 'This simulation uses Brownian particles with mass moving randomly in space, and with a minimal reaction process inside of each particle uptaking or secreting from the field.',
    'particle_comets': 'This simulation extends COMETS with particles that have internal minimal reaction processes.',
    'particle_dfba': 'This simulation puts dFBA inside of the particles, interacting with external fields and adding biomass into the particle mass, reflected by the particle size.',
}

def reversed_tuple(tu):
    return tuple(reversed(tu))

def inverse_tuple(tu):
    return tuple(-x for x in tu)


# ===================================================================
# Functions to get documents and make plots for different experiments
# ===================================================================

# --- DFBA Single ---------------------------------------------------

def get_dfba_single_doc(core=None):
    model_file = "textbook"
    mol_ids = ["glucose", "acetate", "biomass"]
    return {
        "dFBA": get_single_dfba_process(model_file=model_file, mol_ids=mol_ids, path=['fields']),
        "fields": {'glucose': 10, 'acetate': 0, 'biomass': 0.1}
    }

def plot_dfba_single(results, state):
    plot_time_series(results, out_dir='out', filename='dfba_single_timeseries.png')

# --- Many DFBA Spatial ---------------------------------------------------

def get_spatial_many_dfba_doc(core=None):
    model_file = "textbook"
    mol_ids = ['glucose', 'acetate', 'biomass']
    initial_min_max = {"glucose": (0, 20), "acetate": (0, 0), "biomass": (0, 0.1)}
    n_bins = reversed_tuple(DEFAULT_BINS_SMALL)
    return {
        "fields": get_fields_with_schema(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max),
        "spatial_dfba": get_spatial_many_dfba(model_file=model_file, mol_ids=mol_ids, n_bins=n_bins)
    }

def plot_spatial_many_dfba(results, state):
    plot_time_series(results, coordinates=[(0, 0), (1, 1), (2, 2)], out_dir='out', filename='spatial_many_dfba_timeseries.png')
    plot_species_distributions_to_gif(results, out_dir='out', filename='spatial_many_dfba_results.gif')

# --- DFBA Spatial Process ---------------------------------------------

def get_spatial_dfba_process_doc(core=None):
    model_file = "textbook"
    mol_ids = ['glucose', 'acetate', 'biomass']
    initial_min_max = {"glucose": (0, 20), "acetate": (0, 0), "biomass": (0, 0.1)}
    n_bins = reversed_tuple(DEFAULT_BINS_SMALL)
    return {
        "fields": get_fields_with_schema(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max),
        "spatial_dfba": get_spatial_dfba_process(model_file=model_file, mol_ids=mol_ids, n_bins=n_bins)
    }

def plot_dfba_process_spatial(results, state):
    plot_species_distributions_to_gif(results, out_dir='out', filename='spatial_dfba_process.gif')

# --- Diffusion Advection-----------------------------------------------

def get_diffusion_process_doc(core=None):
    mol_ids = ['glucose', 'acetate', 'biomass']
    advection_coeffs = {'biomass': inverse_tuple(DEFAULT_ADVECTION)}
    n_bins = reversed_tuple(DEFAULT_BINS)
    bounds = reversed_tuple(DEFAULT_BOUNDS)
    bins_x, bins_y = n_bins
    # initialize fields
    acetate_field = np.zeros((bins_x, bins_y))
    glc_field = np.random.uniform(low=0.1,high=1,size=n_bins)
    biomass_field = np.zeros((bins_x, bins_y))
    biomass_field[4:5,:] = 2
    return {
        "fields": {'biomass': biomass_field, 'glucose': glc_field, 'acetate': acetate_field},
        "diffusion": get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, advection_coeffs=advection_coeffs),
    }

def plot_diffusion_process(results, state):
    plot_species_distributions_to_gif(results, out_dir='out', filename='diffusion_process.gif')

# --- COMETS -----------------------------------------------------------

def get_comets_doc(core=None):
    model_file = "textbook"
    mol_ids = ['glucose', 'acetate', 'biomass']
    n_bins = reversed_tuple(DEFAULT_BINS)
    bounds = reversed_tuple(DEFAULT_BOUNDS)
    diffusion_coeffs = {'glucose': 3e-1, 'acetate': 1e-1, 'biomass': 1e-1}
    advection_coeffs = {'biomass': inverse_tuple(DEFAULT_ADVECTION)}
    bins_y, bins_x = n_bins

    # initialize acetate concentration to zero
    acetate_field = np.zeros((bins_y, bins_x))

    # a vertical glucose concentration gradient
    vertical_gradient = np.linspace(0, 10, bins_y).reshape(-1, 1)  # Create the gradient for a single column.
    glc_field = np.repeat(vertical_gradient, bins_x, axis=1)  # Replicate the gradient across all columns.

    # place some biomass
    biomass_field = np.zeros((bins_y, bins_x))
    biomass_field[0, int(bins_x/4):int(3*bins_x/4)] = 0.1
    initial_fields = {'biomass': biomass_field, 'glucose': glc_field, 'acetate': acetate_field}
    return {
        "fields": get_fields_with_schema(n_bins=n_bins, mol_ids=mol_ids, initial_fields=initial_fields),
        "spatial_dfba": get_spatial_dfba_process(model_file=model_file, mol_ids=mol_ids, n_bins=n_bins),
        # "spatial_dfba": get_spatial_many_dfba(model_file=model_file, mol_ids=mol_ids, n_bins=n_bins),
        "diffusion": get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, advection_coeffs=advection_coeffs, diffusion_coeffs=diffusion_coeffs)
    }

def plot_comets(results, state):
    n_bins = state['diffusion']['config']['n_bins']
    plot_time_series(results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename='comets_timeseries.png')
    plot_species_distributions_to_gif(results, out_dir='out', filename='comets_results.gif')

# --- Particles -----------------------------------------------------------

def get_particles_doc(core=None):
    initial_min_max = {'glucose': (0.5, 2.0), 'detritus': (0, 0)}
    particle_config = {
        'reactions': {
            'grow': {'reactant': 'glucose', 'product': 'mass'},
            'release': {'reactant': 'mass', 'product': 'detritus'}
        },
        'kinetic_params': {
            'glucose': (0.5, 0.01),
            'mass': (1.0, 0.001)
        }
    }
    n_bins = DEFAULT_BINS
    bounds = DEFAULT_BOUNDS
    n_particles = 1
    diffusion_rate = DEFAULT_DIFFUSION
    advection_rate = DEFAULT_ADVECTION
    add_probability = DEFAULT_ADD_PROBABILITY
    return {
        "state": {
            "fields": initialize_fields(n_bins, initial_min_max),
            "particles": get_particles_state(n_particles=n_particles, n_bins=n_bins, bounds=bounds),
            "particle_movement": get_particle_movement_process( n_bins=n_bins, bounds=bounds,
                diffusion_rate=diffusion_rate, advection_rate=advection_rate, add_probability=add_probability)
        },
        "composition": get_minimal_particle_composition(core=core, config=particle_config)
    }

def plot_particles_sim(results, state):
    bounds = state['particle_movement']['config']['bounds']
    history = [step['particles'] for step in results]
    plot_particles(history=history, env_size=((0, bounds[0]), (0, bounds[1])), out_dir='out', filename='particles.gif')
    plot_species_distributions_with_particles_to_gif(results, out_dir='out', filename='particles_with_fields.gif', bounds=bounds)

# --- Particle-COMETS ----------------------------------------------------

def get_particle_comets_doc(core=None):
    model_file = "textbook"
    mol_ids = ['glucose', 'acetate', 'detritus', 'biomass']
    particle_config = {
        'reactions': {
            'grow': {'reactant': 'glucose', 'product': 'mass'},
            'release': {'reactant': 'mass', 'product': 'detritus'}
        },
        'kinetic_params': {
            'glucose': (0.5, 0.01),
            'mass': (1.0, 0.001)
        }
    }
    n_bins = DEFAULT_BINS
    bounds = DEFAULT_BOUNDS
    particle_advection = DEFAULT_ADVECTION
    n_particles = 10
    add_probability = 0.1

    fields = get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=DEFAULT_INITIAL_MIN_MAX)
    fields['biomass'] = np.zeros(n_bins)  # Initialize biomass field to zero
    fields['biomass'][0, int(n_bins[0]/4):int(3*n_bins[0]/4)] = 0.1  # Add some biomass in the first row
    return {
        "state": {
            "fields": fields,
            "particles": get_particles_state(n_particles=n_particles, bounds=bounds, n_bins=n_bins, fields=fields, mass_range=(1E0, 1E1)),
            "spatial_dfba": get_spatial_dfba_process(model_file=model_file, mol_ids=mol_ids, n_bins=n_bins),
            # "spatial_dfba": get_spatial_many_dfba(model_file=model_file, mol_ids=mol_ids, n_bins=n_bins),
            "diffusion": get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids),
            "particle_movement": get_particle_movement_process(n_bins=n_bins, bounds=bounds,
                                                               add_probability=add_probability, advection_rate=particle_advection)
        },
        "composition": get_minimal_particle_composition(core, config=particle_config)
    }

def plot_particle_comets(results, state):
    bounds = state['particle_movement']['config']['bounds']
    n_bins = state['particle_movement']['config']['n_bins']
    plot_time_series(results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename='particle_comets_timeseries.png')
    plot_species_distributions_with_particles_to_gif(results, out_dir='out', filename='particle_comets_with_fields.gif', bounds=bounds)

# --- dFBA-Particles ---------------------------------------------------

def get_particle_dfba_doc(core=None):
    model_file = "textbook"
    mol_ids = ['glucose', 'acetate']
    initial_min_max = {'glucose': (1, 10), 'acetate': (0, 0)}
    bounds = DEFAULT_BOUNDS
    n_bins = DEFAULT_BINS
    advection_coeffs = {'biomass': inverse_tuple(DEFAULT_ADVECTION)}
    n_particles = 2
    add_probability = 0.1
    particle_advection = DEFAULT_ADVECTION
    fields = get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max)
    return {
        "state": {
            "fields": fields,
            "diffusion": get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, advection_coeffs=advection_coeffs),
            "particles": get_particles_state(n_particles=n_particles, n_bins=n_bins, bounds=bounds, fields=fields),
            "particle_movement": get_particle_movement_process(n_bins=n_bins, bounds=bounds, advection_rate=particle_advection,
                                                               add_probability=add_probability)
        },
        "composition": get_dfba_particle_composition(model_file=model_file)
    }

def plot_particle_dfba(results, state):
    n_bins = state['particle_movement']['config']['n_bins']
    bounds = state['particle_movement']['config']['bounds']
    plot_time_series(results, field_names=['glucose', 'acetate'], coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)],
                     out_dir='out', filename='particle_dfba_timeseries.png')
    plot_particles_mass(results, out_dir='out', filename='particle_dfba_mass.png')
    plot_species_distributions_with_particles_to_gif(results, bounds=bounds, out_dir='out', filename='particle_dfba_with_fields.gif')


# ==================================================
# Functions for running tests and generating reports
# ==================================================

DOCUMENT_CREATORS = {
    'dfba_single': get_dfba_single_doc,
    'spatial_many_dfba': get_spatial_many_dfba_doc,
    'spatial_dfba_process': get_spatial_dfba_process_doc,
    'diffusion_process': get_diffusion_process_doc,
    'comets': get_comets_doc,
    'particles': get_particles_doc,
    'particle_comets': get_particle_comets_doc,
    'particle_dfba': get_particle_dfba_doc,
}

PLOTTERS = {
    'dfba_single': plot_dfba_single,
    'spatial_many_dfba': plot_spatial_many_dfba,
    'spatial_dfba_process': plot_dfba_process_spatial,
    'diffusion_process': plot_diffusion_process,
    'comets': plot_comets,
    'particles': plot_particles_sim,
    'particle_comets': plot_particle_comets,
    'particle_dfba': plot_particle_dfba,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run selected simulations.")
    parser.add_argument(
        '--tests', nargs='*', default=None,
        help='Names of tests to run. If none given, runs the full set.'
    )
    parser.add_argument('--output', default='out', help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()

    test_names = list(SIMULATION_CONFIGS.keys())
    tests_to_run = args.tests if args.tests else test_names
    print(f"\nSelected tests to run: {', '.join(tests_to_run)}\n")

    output_dir = args.output
    prepare_output_dir(output_dir)

    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    total_sim_time = 0.0  # To track simulation time only
    runtimes = {}

    for name in tests_to_run:
        print(f"\n🚀 Running test: {name}")
        if name not in DOCUMENT_CREATORS:
            print(f"Skipping unknown test: '{name}' (no document creator found)")
            continue

        print("Creating document...")
        doc = DOCUMENT_CREATORS[name](core=core)

        print("Sending document...")
        config = SIMULATION_CONFIGS[name]
        sim_start = time.time()
        results = run_composite_document(doc, core=core, name=name, **config)
        sim_end = time.time()

        sim_elapsed = sim_end - sim_start
        runtimes[name] = sim_elapsed
        total_sim_time += sim_elapsed

        print("Generating plots...")
        PLOTTERS[name](results, doc.get('state', doc))

        print(f"✅ Completed: {name} in {sim_elapsed:.2f} seconds")

    print(f"\nCompiling HTML report...")
    generate_html_report(output_dir, SIMULATION_CONFIGS, DESCRIPTIONS, runtimes, total_sim_time)

    print(f"\nTotal simulation time for all the tests: {total_sim_time:.2f} seconds")


if __name__ == '__main__':
    main()
