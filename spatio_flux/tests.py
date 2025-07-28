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
import numpy as np
import time
import json

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

DEFAULT_RUNTIME = 10

SIMULATIONS = {
    'dfba_single': {'time': DEFAULT_RUNTIME},
    'dfba_spatial': {'time': DEFAULT_RUNTIME},
    'diffusion_process': {'time': DEFAULT_RUNTIME},
    'comets': {'time': DEFAULT_RUNTIME},
    'particles': {'time': DEFAULT_RUNTIME},
    'particle_comets': {'time': DEFAULT_RUNTIME},
    'particle_dfba': {'time': DEFAULT_RUNTIME},
}

DESCRIPTIONS = {
    'dfba_single': 'This simulation runs a single dynamic FBA (Flux Balance Analysis) process, tracking external concentrations and biomass.',
    'dfba_spatial': 'This simulation introduces a spatial lattice, with a single dFBA process in each lattice site.',
    'diffusion_process': 'This simulation includes finite volume diffusion and advection on a lattice.',
    'comets': 'This simulation combines dFBA with diffusion/advection to make a spatio-temporal FBA.',
    'particles': 'This simulation models includes particles with mass moving randomly in space, and with a minimal reaction process in each particle uptaking or secreting into its internal mass.',
    'particle_comets': 'This simulation extends COMETS with particle that have minimal reaction processes.',
    'particle_dfba': 'This simulation puts dFBA inside of the particles, interacting with external fields and putting on biomass into the particle mass.',
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

    bins_x = DEFAULT_BINS[1]
    bins_y = DEFAULT_BINS[0]

    # Initialize acetate concentration across the grid to zero.
    acetate_field = np.zeros((bins_x, bins_y))

    # Generate a vertical glucose concentration gradient from 1 at the top to 0 at the bottom.
    vertical_gradient = np.linspace(10, 0, bins_x).reshape(-1, 1)  # Create the gradient for a single column.
    glc_field = np.repeat(vertical_gradient, bins_y, axis=1)  # Replicate the gradient across all columns.

    # place some biomass
    biomass_field = np.zeros((bins_x, bins_y))
    biomass_field[0:2, :] = 1.0

    initial_fields = {'biomass': biomass_field, 'glucose': glc_field, 'acetate': acetate_field}
    state = get_spatial_dfba_state(n_bins=reversed_bins(DEFAULT_BINS), mol_ids=mol_ids,
                                   initial_fields=initial_fields,
                                   initial_min_max=initial_min_max)
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


from html import escape as html_escape


def generate_html_report(output_dir, runtimes=None, total_runtime=None):
    output_dir = Path(output_dir)
    report_path = output_dir / 'report.html'
    all_files = list(output_dir.glob('*'))

    html = [
        '<html><head><title>Simulation Results</title>',
        '<style>',
        'body { font-family: sans-serif; padding: 20px; background: #fcfcfc; color: #222; }',
        'h1, h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; }',
        'pre { background-color: #f8f8f8; padding: 8px; border: 1px solid #ddd; overflow-x: auto; }',
        'details { margin: 6px 0; padding-left: 1em; }',
        'summary { font-weight: 600; cursor: pointer; }',
        'code { background: #f1f1f1; padding: 2px 4px; border-radius: 4px; }',
        'nav ul { list-style: none; padding-left: 0; }',
        'nav ul li { margin: 5px 0; }',
        'a.download-btn { display: inline-block; margin: 8px 0; padding: 4px 8px; background: #eee; border: 1px solid #ccc; text-decoration: none; font-size: 0.9em; border-radius: 4px; }',
        '</style>',
        '</head><body>',
        '<h1>Simulation Results</h1>'
    ]

    # Table of Contents
    html.append('<nav><h2>Contents</h2><ul>')
    for test in SIMULATIONS:
        html.append(f'<li><a href="#{test}">{test}</a></li>')
    html.append('</ul></nav>')

    test_files = {test: [] for test in SIMULATIONS}
    others = []

    for file in all_files:
        if file.name == report_path.name:
            continue
        for test in test_files:
            if file.name.startswith(test):
                test_files[test].append(file)
                break
        else:
            others.append(file)

    def json_to_html(obj):
        if isinstance(obj, dict):
            return ''.join(
                f"<details><summary>{html_escape(str(k))}</summary>{json_to_html(v)}</details>"
                for k, v in obj.items()
            )
        elif isinstance(obj, list):
            return ''.join(
                f"<details><summary>[{i}]</summary>{json_to_html(v)}</details>"
                for i, v in enumerate(obj)
            )
        else:
            return f"<code>{html_escape(json.dumps(obj))}</code>"

    for test, files in test_files.items():
        if not files:
            continue
        html.append(f'<h2 id="{test}">{test}</h2>')

        # Optional description
        description = DESCRIPTIONS.get(test, '')
        if description:
            html.append(f'<p><em>{description}</em></p>')

        # Runtime
        if runtimes and test in runtimes:
            html.append(f'<p><strong>Runtime:</strong> {runtimes[test]:.2f} seconds</p>')

        # Sort files
        json_file = next((f for f in files if f.suffix == '.json'), None)
        viz_file = next((f for f in files if f.name == f"{test}_viz.png"), None)
        pngs = sorted(f for f in files if f.suffix == '.png' and f != viz_file)
        gifs = sorted(f for f in files if f.suffix == '.gif')

        # JSON viewer
        if json_file:
            html.append(f'<h3>{json_file.name}</h3>')
            html.append(f'<a class="download-btn" href="{json_file.name}" target="_blank">View full JSON</a>')
            try:
                with open(json_file, 'r') as jf:
                    full_data = json.load(jf)
                state_data = full_data.get('state', {})
                if state_data:
                    html.append(json_to_html(state_data))
                else:
                    html.append('<p><em>No "state" key found.</em></p>')
            except Exception as e:
                html.append(f'<pre>Could not load JSON: {e}</pre>')

        # Bigraph visualization
        if viz_file:
            html.append(f'<h3>{viz_file.name}</h3>')
            html.append(f'<img src="{viz_file.name}" style="max-width:100%"><hr>')

        # PNG plots
        for f in pngs:
            html.append(f'<h3>{f.name}</h3><img src="{f.name}" style="max-width:100%"><hr>')

        # GIFs
        for f in gifs:
            html.append(f'<h3>{f.name}</h3><img src="{f.name}" style="max-width:100%"><hr>')

    if others:
        html.append('<h2>Other Generated Files</h2>')
        for f in sorted(others):
            html.append(f'<p>{f.name}</p>')

    if total_runtime:
        html.append(f'<h2>Total Runtime</h2><p><strong>{total_runtime:.2f} seconds</strong></p>')

    html.append('</body></html>')

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

    runtimes = {}
    total_start = time.time()

    for name in tests_to_run:
        print(f"\n🚀 Running test: {name}")
        if name not in DOCUMENT_CREATORS:
            print(f"Skipping unknown test: '{name}' (no document creator found)")
            continue

        print("Creating document...")
        doc = DOCUMENT_CREATORS[name](core=core)

        print("Sending document...")
        config = SIMULATIONS[name]

        start_time = time.time()
        results = run_composite_document(doc, core=core, name=name, **config)
        end_time = time.time()
        elapsed = end_time - start_time
        runtimes[name] = elapsed

        print("Generating plots...")
        PLOTTERS[name](results, doc.get('state', doc))

        print(f"✅ Completed: {name} in {elapsed:.2f} seconds")

    total_elapsed = time.time() - total_start
    print(f"\nCompiling HTML report...")
    generate_html_report(output_dir, runtimes, total_elapsed)

    print(f"\nTotal runtime for all tests: {total_elapsed:.2f} seconds")


if __name__ == '__main__':
    main()
