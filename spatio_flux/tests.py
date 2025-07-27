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
from pathlib import Path
import shutil

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
    get_diffusion_advection_process, get_diffusion_advection_state,
    get_particle_movement_state, get_minimal_particle_composition, get_dfba_particle_composition,
    get_particle_dfba_state, default_config, get_particle_comets_state
)

DEFAULT_BOUNDS = (5.0, 10.0)
DEFAULT_BINS = (5, 10)
DEFAULT_ADVECTION = (0, -0.1)
DEFAULT_DIFFUSION = 0.1
DEFAULT_ADD_PROBABILITY = 0.4
DEFAULT_ADD_BOUNDARY = ['top', 'left', 'right']
DEFAULT_REMOVE_BOUNDARY = ['left', 'right']

DEFAULT_RUNTIME = 40

SIMULATIONS = {
    'dfba_single': {'time': DEFAULT_RUNTIME},
    'dfba_spatial': {'time': DEFAULT_RUNTIME},
    'diffusion_process': {'time': DEFAULT_RUNTIME},
    'comets': {'time': DEFAULT_RUNTIME},
    'particles': {'time': DEFAULT_RUNTIME},
    'particle_comets': {'time': DEFAULT_RUNTIME},
    'particle_dfba': {'time': DEFAULT_RUNTIME},
}

def reversed_bins(n_bins):
    return tuple(reversed(n_bins))


# ===================================================================
# Functions to get documents and make plots for different experiments
# ===================================================================

# --- DFBA Single ---------------------------------------------------

def get_dfba_single_doc(core=None):
    mol_ids = ["glucose", "acetate", "biomass"]
    return {
        'dfba': get_dfba_process_state(path=['fields'], mol_ids=mol_ids),
        'fields': {
            'glucose': 10,
            'acetate': 0,
            'biomass': 0.1,
        }
    }

def plot_dfba_single(results, state):
    plot_time_series(results, out_dir='out', filename='dfba_single_timeseries.png')

# --- DFBA Spatial ---------------------------------------------------

def get_dfba_spatial_doc(core=None):
    mol_ids = ['glucose', 'acetate', 'biomass']
    initial_min_max = {"glucose": (0, 20), "acetate": (0, 0), "biomass": (0, 0.1)}
    return get_spatial_dfba_state(n_bins=reversed_bins(DEFAULT_BINS), mol_ids=mol_ids, initial_min_max=initial_min_max)

def plot_dfba_spatial(results, state):
    plot_time_series(results, coordinates=[(0, 0), (1, 1), (2, 2)],
                     out_dir='out', filename='dfba_spatial_timeseries.png')
    plot_species_distributions_to_gif(results, out_dir='out', filename='dfba_spatial_results.gif')

# --- Diffusion Advection-----------------------------------------------

def get_diffusion_process_doc(core=None):
    mol_ids = ['glucose', 'acetate', 'biomass']
    advection_coeffs = {'biomass': DEFAULT_ADVECTION}
    return get_diffusion_advection_state(bounds=reversed_bins(DEFAULT_BOUNDS), n_bins=reversed_bins(DEFAULT_BINS),
                                         mol_ids=mol_ids, advection_coeffs=advection_coeffs)

def plot_diffusion_process(results, state):
    plot_species_distributions_to_gif(results, out_dir='out', filename='diffusion_process.gif')

# --- COMETS -----------------------------------------------------------

def get_comets_doc(core=None):
    mol_ids = ['glucose', 'acetate', 'biomass']
    initial_min_max = {'glucose': (10, 10), 'acetate': (0, 0), 'biomass': (0, 0.1)}
    advection_coeffs = {'biomass': DEFAULT_ADVECTION}
    state = get_spatial_dfba_state(n_bins=reversed_bins(DEFAULT_BINS), mol_ids=mol_ids, initial_min_max=initial_min_max)
    state['diffusion'] = get_diffusion_advection_process(mol_ids=mol_ids, advection_coeffs=advection_coeffs,
        bounds=reversed_bins(DEFAULT_BOUNDS), n_bins=reversed_bins(DEFAULT_BINS))
    return state

def plot_comets(results, state):
    n_bins = state['diffusion']['config']['n_bins']
    plot_time_series(results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename='comets_timeseries.png')
    plot_species_distributions_to_gif(results, out_dir='out', filename='comets_results.gif')

# --- Particles -----------------------------------------------------------

def get_particles_doc(core=None):
    initial_min_max = {'glucose': (0.5, 2.0), 'detritus': (0, 0)}
    # Particle configuration
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
    state = get_particle_movement_state(n_particles=1, bounds=DEFAULT_BOUNDS, n_bins=DEFAULT_BINS,
                                        diffusion_rate=DEFAULT_DIFFUSION, advection_rate=DEFAULT_ADVECTION,
                                        add_probability=DEFAULT_ADD_PROBABILITY, initial_min_max=initial_min_max)
    return {'state': state, 'composition': get_minimal_particle_composition(core=core, config=particle_config)}

def plot_particles_sim(results, state):
    bounds = state['particle_movement']['config']['bounds']
    history = [step['particles'] for step in results]
    plot_particles(history=history, env_size=((0, bounds[0]), (0, bounds[1])), out_dir='out', filename='particles.gif')
    plot_species_distributions_with_particles_to_gif(results, out_dir='out', filename='particles_with_fields.gif', bounds=bounds)

# --- Particle-COMETS ----------------------------------------------------

def get_particle_comets_doc(core=None):
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
    state = get_particle_comets_state(
        bounds=DEFAULT_BOUNDS, n_bins=DEFAULT_BINS, particle_advection_rate=DEFAULT_ADVECTION, mol_ids=mol_ids)
    return {'state': state, 'composition': get_minimal_particle_composition(core, config=particle_config)}

def plot_particle_comets(results, state):
    bounds = state['particle_movement']['config']['bounds']
    n_bins = state['particle_movement']['config']['n_bins']
    plot_time_series(results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename='particle_comets_timeseries.png')
    plot_species_distributions_with_particles_to_gif(results, out_dir='out', filename='particle_comets_with_fields.gif', bounds=bounds)

# --- dFBA-Particles ---------------------------------------------------

def get_particle_dfba_doc(core=None):
    mol_ids = ['glucose', 'acetate']
    initial_min_max = {'glucose': (1, 10), 'acetate': (0, 0)}
    state = get_particle_dfba_state(core,
                                    n_particles=2,
                                    bounds=DEFAULT_BOUNDS, n_bins=DEFAULT_BINS,
                                    particle_add_probability=DEFAULT_ADD_PROBABILITY,
                                    particle_boundary_to_add=DEFAULT_ADD_BOUNDARY,
                                    particle_boundary_to_remove=DEFAULT_ADD_BOUNDARY,
                                    particle_advection_rate=DEFAULT_ADVECTION,
                                    mol_ids=mol_ids, initial_min_max=initial_min_max)
    return {'state': state, 'composition': get_dfba_particle_composition()}

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
    'dfba_spatial': get_dfba_spatial_doc,
    'diffusion_process': get_diffusion_process_doc,
    'comets': get_comets_doc,
    'particles': get_particles_doc,
    'particle_comets': get_particle_comets_doc,
    'particle_dfba': get_particle_dfba_doc,
}

PLOTTERS = {
    'dfba_single': plot_dfba_single,
    'dfba_spatial': plot_dfba_spatial,
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


def prepare_output_dir(output_dir):
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"🧹 Clearing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def generate_html_report(output_dir):
    report_file = 'report.html'
    output_dir = Path(output_dir)
    all_files = list(output_dir.glob('*'))

    html = ['<html><head><title>Simulation Results</title></head><body>']
    html.append('<h1>Simulation Results</h1>')

    # Group files by test based on prefix matching
    test_figures = {test: [] for test in SIMULATIONS.keys()}
    others = []

    for file in all_files:
        if file.name == report_file:
            continue  # Skip self
        matched = False
        for test in test_figures:
            if file.name.startswith(test):
                test_figures[test].append(file)
                matched = True
                break
        if not matched:
            others.append(file)

    # Add ordered sections per test
    for test_name in SIMULATIONS.keys():
        files = sorted(test_figures[test_name])
        if not files:
            continue

        html.append(f'<h2>{test_name}</h2>')
        for file in files:
            name = file.name
            if name.endswith(('.png', '.gif')):
                html.append(f'<h3>{name}</h3>')
                html.append(f'<img src="{name}" style="max-width:100%"><hr>')
            else:
                html.append(f'<p>Generated file: {name}</p>')

    # Add miscellaneous files
    if others:
        html.append('<h2>Other Generated Files</h2>')
        for file in sorted(others):
            html.append(f'<p>{file.name}</p>')

    html.append('</body></html>')
    report_path = output_dir / report_file
    with open(report_path, 'w') as f:
        f.write('\n'.join(html))


def main():
    args = parse_args()

    test_names = list(SIMULATIONS.keys())
    tests_to_run = args.tests if args.tests else test_names
    print(f"\nSelected tests to run: {', '.join(tests_to_run)}\n")

    output_dir = args.output
    prepare_output_dir(output_dir)

    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    for name in tests_to_run:
        print(f"\n🚀 Running test: {name}")
        if name not in DOCUMENT_CREATORS:
            print(f"Skipping unknown test: '{name}' (no document creator found)")
            continue

        print("Creating document...")
        doc = DOCUMENT_CREATORS[name](core=core)

        print("Sending document...")
        config = SIMULATIONS[name]
        results = run_composite_document(doc, core=core, name=name, **config)

        print("Generating plots...")
        PLOTTERS[name](results, doc.get('state', doc))

        print(f"✅ Completed: {name}")

    print("\n📝 Compiling HTML report...")
    generate_html_report(output_dir)


if __name__ == '__main__':
    main()
