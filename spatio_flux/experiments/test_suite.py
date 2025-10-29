"""
Simulation Runner and Visualizer for Spatio-Flux Processes and Composites

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
import time
import numpy as np

from process_bigraph import register_types as register_process_types
from vivarium.vivarium import VivariumTypes

from spatio_flux import register_types
from spatio_flux.library.helpers import run_composite_document, prepare_output_dir, generate_html_report, \
    reversed_tuple, inverse_tuple
from spatio_flux.viz.plot import ( plot_time_series, plot_particles_mass, plot_species_distributions_to_gif,
    plot_species_distributions_with_particles_to_gif, plot_particles, plot_model_grid,
    plot_snapshots_grid,
)
from spatio_flux.processes import (
    get_spatial_many_dfba, get_spatial_dfba_process, get_fields, get_fields_with_schema, get_field_names,
    get_diffusion_advection_process, get_particle_movement_process, initialize_fields, get_minimal_particle_composition,
    get_dfba_particle_composition, get_particles_state, MODEL_REGISTRY_DFBA, get_dfba_process_from_registry,
    get_particle_divide_process, DIVISION_MASS_THRESHOLD,
)


# ====================================================================
# Functions to get documents and make plots for different compositions
# ====================================================================

# --- DFBA Single ---------------------------------------------------

def get_dfba_single_doc(
        core=None,
        config=None,
):
    model_id = config.get('model_id', 'ecoli core')
    biomass_id = config.get('biomass_id', f'{model_id} biomass')
    dfba_process = get_dfba_process_from_registry(
        model_id=model_id,
        biomass_id=biomass_id,
        path=['fields']
    )
    substrates = list(dfba_process['inputs']['substrates'].keys())
    initial_fields = config.get('initial_fields', {'glucose': 2, 'acetate': 0})
    if biomass_id not in initial_fields:
        initial_fields[biomass_id] = 0.1
    for substrate in substrates:
        if substrate not in initial_fields:
            initial_fields[substrate] = 10.0
    doc = {
        f'{model_id} dFBA': dfba_process,
        'fields': initial_fields
    }
    return doc

def plot_dfba_single(results, state, config=None, filename='dfba_single_timeseries.png'):
    config = config or {}
    field_names = list(state['fields'].keys())
    filename = config.get('filename', 'dfba_single_timeseries')
    plot_time_series(results, field_names=field_names, out_dir='out', filename=f'{filename}.png',)

# --- Multiple DFBAs ---------------------------------------------------

def get_multi_dfba(core=None, config=None):
    mol_ids = ['glucose', 'acetate']
    model_ids = list(MODEL_REGISTRY_DFBA.keys())
    dfbas = {}
    for model_id, spec in MODEL_REGISTRY_DFBA.items():
        process_id = f'dfba_{model_id}'
        dfba_process = get_dfba_process_from_registry(
            model_id=model_id,
            biomass_id=model_id,
            path=['fields']
        )
        dfbas[process_id] = dfba_process

    initial_biomass = {organism: 0.1 for organism in model_ids}
    field_names = get_field_names(MODEL_REGISTRY_DFBA)
    more_fields = {mol_id: 0.1 for mol_id in field_names if mol_id not in ['glucose', 'acetate']}
    doc = {
        **dfbas,
        'fields': {
            'glucose': 10,
            'acetate': 0,
            **more_fields,
            **initial_biomass
        }
    }
    return doc

def plot_multi_dfba(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'dfba_multi_timeseries.png')
    model_ids = list(MODEL_REGISTRY_DFBA.keys())
    field_names = get_field_names(MODEL_REGISTRY_DFBA)
    species_ids = model_ids + field_names
    plot_time_series(
        results, field_names=species_ids,
        log_scale=True,
        normalize=True,
        out_dir='out', filename=filename)

# --- Many DFBA Spatial ---------------------------------------------------

def get_spatial_many_dfba_doc(core=None, config=None):
    dissolved_model_file = 'ecoli core'
    mol_ids = ['glucose', 'acetate', 'biomass']
    initial_min_max = {'glucose': (0, 20), 'acetate': (0, 0), 'biomass': (0, 0.1)}
    n_bins = reversed_tuple(DEFAULT_BINS_SMALL)
    return {
        'fields': get_fields_with_schema(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max),
        'spatial_dfba': get_spatial_many_dfba(model_file=dissolved_model_file, mol_ids=mol_ids, n_bins=n_bins)
    }

def plot_spatial_many_dfba(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'spatial_many_dfba')
    plot_time_series(results, coordinates=[(0, 0), (1, 1), (2, 2)], out_dir='out', filename=f'{filename}_timeseries.png')
    plot_species_distributions_to_gif(
        results, out_dir='out', filename=f'{filename}_video.gif')

# --- DFBA Spatial Process ---------------------------------------------

def build_model_grid(n_bins, model_positions=None):
    rows, cols = n_bins

    # Start with all empty strings
    model_grid = [['' for _ in range(cols)] for _ in range(rows)]

    # If we have models to place, insert them
    if model_positions:
        for model_id, positions in model_positions.items():
            for (r, c) in positions:
                if 0 <= r < rows and 0 <= c < cols:
                    model_grid[r][c] = model_id
                else:
                    raise ValueError(f"Position {(r, c)} is out of bounds for grid {n_bins}")

    return model_grid


def get_spatial_dfba_process_doc(core=None, config=None):
    # make the fields
    mol_ids = ['glucose', 'acetate', 'biomass']
    initial_min_max = {'glucose': (20, 20), 'glycolate': (10,10), 'ammonium': (10, 10),
                       'acetate': (0, 0), 'biomass': (0.01, 0.01)}
    n_bins = reversed_tuple(DEFAULT_BINS_SMALL)
    initial_fields = {}
    initial_fields = get_fields(n_bins, mol_ids, initial_min_max, initial_fields)

    bins_y, bins_x = n_bins
    horizontal_gradient = np.linspace(0, 20, bins_x).reshape(1, -1)  # Create the gradient for a single row.
    initial_fields['glucose'] = np.repeat(horizontal_gradient, bins_y, axis=0)  # Replicate the gradient across all rows.

    model_positions = {
        'ecoli core': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        'ecoli': [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
        'cdiff': [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
        'pputida': [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)],
        'yeast': [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
        'llactis': [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)]
    }
    model_grid = build_model_grid(n_bins=n_bins, model_positions=model_positions)

    # make the spatial dfba with different models and parameters
    spatial_dfba_config = {
        'n_bins': n_bins,
        'models': MODEL_REGISTRY_DFBA,
        'model_grid': model_grid
    }
    doc = {
        'fields': initial_fields,
        'spatial_dfba': get_spatial_dfba_process(model_id=None, config=spatial_dfba_config)
    }
    return doc

def plot_dfba_process_spatial(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'spatial_dfba_process')
    model_grid = state['spatial_dfba']['config']['model_grid']
    plot_time_series(results, coordinates=[(0, 0), (1, 1), (2, 2)], out_dir='out', filename=f'{filename}_timeseries.png')
    plot_model_grid(model_grid, title='model grid', show_border_coords=True, out_dir='out', filename=f'{filename}_model_grid.png')
    plot_species_distributions_to_gif(results, out_dir='out', filename=f'{filename}_video.gif')

# --- Diffusion Advection-----------------------------------------------

def get_diffusion_process_doc(core=None, config=None):
    mol_ids = ['glucose', 'biomass']
    advection_coeffs = {'biomass': inverse_tuple(DEFAULT_ADVECTION)}
    diffusion_coeffs = {'glucose': DEFAULT_DIFFUSION/10, 'biomass': DEFAULT_DIFFUSION/10}
    n_bins = reversed_tuple(DEFAULT_BINS)
    bounds = reversed_tuple(DEFAULT_BOUNDS)
    # initialize fields
    glc_field = np.random.uniform(low=0.1,high=2,size=n_bins)
    biomass_field = np.zeros(n_bins)
    biomass_field[4:5,:] = 10
    return {
        'fields': {'biomass': biomass_field, 'glucose': glc_field},
        'diffusion': get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids,
                                                     diffusion_coeffs=diffusion_coeffs, advection_coeffs=advection_coeffs),
    }

def plot_diffusion_process(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'diffusion_process')
    plot_species_distributions_to_gif(results, out_dir='out', filename=f'{filename}_video.gif')

# --- COMETS -----------------------------------------------------------

def get_comets_doc(core=None, config=None):
    dissolved_model_id = 'ecoli core'
    mol_ids = ['glucose', 'acetate', 'biomass']
    n_bins = reversed_tuple(DEFAULT_BINS)
    bounds = reversed_tuple(DEFAULT_BOUNDS)
    diffusion_coeffs = {'glucose': 0, 'acetate': 1e-1, 'biomass': 5e-2}
    advection_coeffs = {'biomass': inverse_tuple(DEFAULT_ADVECTION)}
    bins_y, bins_x = n_bins

    # initialize acetate concentration to zero
    acetate_field = np.zeros(n_bins)

    # a vertical glucose concentration gradient
    vertical_gradient = np.linspace(0.01, 10, bins_y).reshape(-1, 1)  # Create the gradient for a single column.
    glc_field = np.repeat(vertical_gradient, bins_x, axis=1)  # Replicate the gradient across all columns.

    # place some biomass
    biomass_field = np.zeros(n_bins)
    biomass_field[0:1, int(bins_x/4):int(3*bins_x/4)] = 0.1
    initial_fields = {'biomass': biomass_field, 'glucose': glc_field, 'acetate': acetate_field}

    # place models on the grid
    model_grid = np.zeros(n_bins, dtype='U20')
    model_grid[:] = 'ecoli core'
    model_grid = model_grid.tolist()

    config = {
        'mol_ids': mol_ids,
        'n_bins': n_bins,
        'models': MODEL_REGISTRY_DFBA,
        'model_grid': model_grid
    }
    doc = {
        'fields': get_fields_with_schema(n_bins=n_bins, mol_ids=mol_ids, initial_fields=initial_fields),
        'spatial_dfba': get_spatial_dfba_process(model_id=dissolved_model_id, config=config),
        # 'spatial_dfba': get_spatial_many_dfba(model_file=model_file, mol_ids=mol_ids, n_bins=n_bins),
        'diffusion': get_diffusion_advection_process(
            bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, advection_coeffs=advection_coeffs, diffusion_coeffs=diffusion_coeffs)
    }
    return doc

def plot_comets(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'comets')
    n_bins = state['diffusion']['config']['n_bins']
    bounds = state['diffusion']['config']['bounds']
    plot_time_series(
        results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename=f'{filename}_timeseries.png')

    plot_snapshots_grid(results, field_names=['glucose', 'acetate'], n_snapshots=6,
                        bounds=bounds, out_dir='out', filename=f'{filename}_snapshots.png',
                        suptitle='Fields snapshots')

    plot_species_distributions_to_gif(results, out_dir='out', filename=f'{filename}_video.gif')

# --- Particles -----------------------------------------------------------

def get_particles_doc(core=None, config=None):
    division_mass_threshold = config.get('division_mass_threshold', DIVISION_MASS_THRESHOLD) # divide at mass 5.0

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
        'state': {
            'fields': initialize_fields(n_bins, initial_min_max),
            'particles': get_particles_state(n_particles=n_particles, n_bins=n_bins, bounds=bounds),
            'particle_movement': get_particle_movement_process(n_bins=n_bins, bounds=bounds,
                diffusion_rate=diffusion_rate, advection_rate=advection_rate, add_probability=add_probability),
            'particle_division': get_particle_divide_process(division_mass_threshold=division_mass_threshold),
        },
        'composition': get_minimal_particle_composition(core=core, config=particle_config)
    }

def plot_particles_sim(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'particles')
    bounds = state['particle_movement']['config']['bounds']
    history = [step['particles'] for step in results]
    plot_particles(
        history=history, env_size=((0, bounds[0]), (0, bounds[1])),
        out_dir='out', filename=f'{filename}_particles_alone_video.gif')
    plot_species_distributions_with_particles_to_gif(
        results, out_dir='out', filename=f'{filename}_video.gif', bounds=bounds)

    plot_snapshots_grid(results, field_names=['glucose', 'acetate'], n_snapshots=6,
                        bounds=bounds, out_dir='out', filename=f'{filename}_snapshots.png',
                        suptitle='Fields snapshots')

# --- Particle-COMETS ----------------------------------------------------

def get_particle_comets_doc(core=None, config=None):
    division_mass_threshold=config.get('division_mass_threshold', DIVISION_MASS_THRESHOLD) # divide at mass 5.0

    dissolved_model_id = 'ecoli core'
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
    n_particles = 2
    add_probability = 0.1

    fields = get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=DEFAULT_INITIAL_MIN_MAX)
    fields['biomass'] = np.zeros(n_bins)  # Initialize biomass field to zero
    fields['biomass'][0, int(n_bins[0]/4):int(3*n_bins[0]/4)] = 0.1  # Add some biomass in the first row

    # spatial dfba config
    spatial_dfba_config = {'mol_ids': mol_ids, 'n_bins': n_bins}

    return {
        'state': {
            'fields': fields,
            'particles': get_particles_state(n_particles=n_particles, bounds=bounds, n_bins=n_bins, fields=fields, mass_range=(1E0, 1E1)),
            'spatial_dfba': get_spatial_dfba_process(model_id=dissolved_model_id, config=spatial_dfba_config),
            # 'spatial_dfba': get_spatial_many_dfba(model_file=model_file, mol_ids=mol_ids, n_bins=n_bins),
            'diffusion': get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids),
            'particle_movement': get_particle_movement_process(
                n_bins=n_bins, bounds=bounds, add_probability=add_probability, advection_rate=particle_advection),
            'particle_division': get_particle_divide_process(division_mass_threshold=division_mass_threshold),
        },
        'composition': get_minimal_particle_composition(core, config=particle_config)
    }

def plot_particle_comets(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'particle_comets')
    bounds = state['particle_movement']['config']['bounds']
    n_bins = state['particle_movement']['config']['n_bins']
    plot_time_series(results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename=f'{filename}_timeseries.png')
    plot_snapshots_grid(results, field_names=['glucose', 'acetate'], n_snapshots=6,
                        bounds=bounds, out_dir='out', filename=f'{filename}_snapshots.png')
    plot_species_distributions_with_particles_to_gif(
        results, out_dir='out', filename=f'{filename}_video.gif', bounds=bounds)

# --- dFBA-Particles ---------------------------------------------------

def get_particle_dfba_doc(core=None, config=None):
    particle_model_id = config.get('particle_model_id', 'ecoli core')
    division_mass_threshold=config.get('division_mass_threshold', DIVISION_MASS_THRESHOLD) # divide at mass 5.0

    mol_ids = ['glucose', 'acetate']
    initial_min_max = {'glucose': (1, 10), 'acetate': (0, 0)}
    bounds = DEFAULT_BOUNDS
    n_bins = DEFAULT_BINS
    advection_coeffs = {'biomass': inverse_tuple(DEFAULT_ADVECTION)}
    n_particles = 2
    add_probability = 0.2
    particle_advection = DEFAULT_ADVECTION
    fields = get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max)
    return {
        'state': {
            'fields': fields,
            'diffusion': get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, advection_coeffs=advection_coeffs),
            'particles': get_particles_state(n_particles=n_particles, n_bins=n_bins, bounds=bounds, fields=fields),
            'particle_movement': get_particle_movement_process(
                n_bins=n_bins, bounds=bounds, advection_rate=particle_advection, add_probability=add_probability),
            'particle_division': get_particle_divide_process(division_mass_threshold=division_mass_threshold),
        },
        'composition': get_dfba_particle_composition(model_file=particle_model_id)
    }

def plot_particle_dfba(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'particle_dfba')
    n_bins = state['particle_movement']['config']['n_bins']
    bounds = state['particle_movement']['config']['bounds']
    plot_time_series(results, field_names=['glucose', 'acetate'], coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)],
                     out_dir='out', filename=f'{filename}_timeseries.png')
    plot_particles_mass(results, out_dir='out', filename=f'{filename}_mass.png')
    plot_snapshots_grid(results, field_names=['glucose', 'acetate'], n_snapshots=6,
                        bounds=bounds, out_dir='out', filename=f'{filename}_snapshots.png')
    plot_species_distributions_with_particles_to_gif(
        results, bounds=bounds, out_dir='out', filename=f'{filename}_video.gif')

# --- dFBA-Particles-COMETS ---------------------------------------------------

def get_particle_dfba_comets_doc(core=None, config=None):
    particle_model_id = config.get('particle_model_id', 'ecoli core')
    dissolved_model_id = config.get('dissolved_model_id', 'ecoli core')
    division_mass_threshold=config.get('division_mass_threshold', DIVISION_MASS_THRESHOLD) # divide at mass 5.0

    mol_ids = ['glucose', 'acetate', 'biomass']
    initial_min_max = {'glucose': (1, 5), 'acetate': (0, 0), 'biomass': (0, 0.1)}
    bounds = DEFAULT_BOUNDS
    n_bins = DEFAULT_BINS
    advection_coeffs = {'biomass': inverse_tuple(DEFAULT_ADVECTION)}
    n_particles = 2
    add_probability = 0.2
    particle_advection = DEFAULT_ADVECTION
    fields = get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max)

    # spatial dfba config
    spatial_dfba_config = {'mol_ids': mol_ids, 'n_bins': n_bins}

    doc = {
        'state': {
            'fields': fields,
            'diffusion': get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, advection_coeffs=advection_coeffs),
            'spatial_dfba': get_spatial_dfba_process(model_id=dissolved_model_id, config=spatial_dfba_config),
            'particles': get_particles_state(n_particles=n_particles, n_bins=n_bins, bounds=bounds, fields=fields),
            'particle_movement': get_particle_movement_process(
                n_bins=n_bins, bounds=bounds, advection_rate=particle_advection, add_probability=add_probability),
            'particle_division': get_particle_divide_process(division_mass_threshold=division_mass_threshold),
        },
        'composition': get_dfba_particle_composition(model_file=particle_model_id)
    }
    return doc

def plot_particle_dfba_comets(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'particle_dfba_comets')
    n_bins = state['particle_movement']['config']['n_bins']
    bounds = state['particle_movement']['config']['bounds']
    plot_time_series(results, field_names=['glucose', 'acetate', 'biomass'], coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)],
                     out_dir='out', filename=f'{filename}_timeseries.png')
    plot_particles_mass(results, out_dir='out', filename=f'{filename}_mass.png')
    plot_snapshots_grid(results, field_names=['glucose', 'acetate'], n_snapshots=6,
                        bounds=bounds, out_dir='out', filename=f'{filename}_snapshots.png')
    plot_species_distributions_with_particles_to_gif(
        results, bounds=bounds, out_dir='out', filename=f'{filename}_video.gif')

# ==================================================
# Functions for running tests and generating reports
# ==================================================

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

DEFAULT_RUNTIME_SHORT = 10
DEFAULT_RUNTIME_LONG = 20

SIMULATIONS = {
    'ecoli_core_dfba': {
        'description': 'This simulation runs a dFBA (dynamic Flux Balance Analysis) model of E. coli core metabolism, tracking external concentrations and biomass.',
        'doc_func': get_dfba_single_doc,
        'plot_func': plot_dfba_single,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {'model_id': 'ecoli core', 'initial_fields': {'glucose': 10, 'acetate': 0}},
        'plot_config': {'filename': 'ecoli_core_dfba'}
    },
    'ecoli_dfba': {
        'description': 'This simulation runs a dFBA of the large E. coli metabolic model, iAF1260',
        'doc_func': get_dfba_single_doc,
        'plot_func': plot_dfba_single,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {'model_id': 'ecoli', 'initial_fields': {'glucose': 10, 'formate': 5}},
        'plot_config': {'filename': 'ecoli_dfba'}
    },
    'cdiff_dfba': {
        'description': 'This simulation runs a dFBA model of Clostridioides difficile. iCN900',
        'doc_func': get_dfba_single_doc,
        'plot_func': plot_dfba_single,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {'model_id': 'cdiff', 'initial_fields': {'glucose': 2, 'acetate': 10}},
        'plot_config': {'filename': 'cdiff_dfba'}
    },
    'pputida_dfba': {
        'description': 'This simulation runs a dFBA model of Pseudomonas putida, iJN746',
        'doc_func': get_dfba_single_doc,
        'plot_func': plot_dfba_single,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {'model_id': 'pputida', 'initial_fields': {'glucose': 8, 'pputida biomass': 2}},
        'plot_config': {'filename': 'pputida_dfba'}
    },
    'yeast_dfba': {
        'description': 'This simulation runs a dFBA model of Saccharomyces cerevisiae (yeast), iMM904',
        'doc_func': get_dfba_single_doc,
        'plot_func': plot_dfba_single,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {'model_id': 'yeast', 'initial_fields': {'glucose': 5}},
        'plot_config': {'filename': 'yeast_dfba'}
    },
    'llactis_dfba': {
        'description': 'This simulation runs a dFBA model of Lactococcus lactis, iNF517',
        'doc_func': get_dfba_single_doc,
        'plot_func': plot_dfba_single,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {'model_id': 'llactis', 'initial_fields': {'glucose': 100, 'llactis biomass': 2.0}},
        'plot_config': {'filename': 'llactis_dfba'}
    },
    'multi_dfba': {
        'description': 'This simulation runs multiple dFBA processes in the same environment, each with its own model and parameters.',
        'doc_func': get_multi_dfba,
        'plot_func': plot_multi_dfba,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {},
        'plot_config': {'filename': 'multi_dfba'}
    },
    'spatial_many_dfba': {
        'description': 'This simulation introduces a spatial lattice, with a single dFBA process in each lattice site.',
        'doc_func': get_spatial_many_dfba_doc,
        'plot_func': plot_spatial_many_dfba,
        'time': DEFAULT_RUNTIME_SHORT,
        'config': {'model_id': 'ecoli core'},
        'plot_config': {'filename': 'spatial_many_dfba'}
    },
    'spatial_dfba_process': {
        'description': 'This simulation introduces a spatial lattice, with a spatial dFBA process that runs all the lattice sites',
        'doc_func': get_spatial_dfba_process_doc,
        'plot_func': plot_dfba_process_spatial,
        'time': DEFAULT_RUNTIME_SHORT,
        'config': {},
        'plot_config': {'filename': 'spatial_dfba_process'}
    },
    'diffusion_process': {
        'description': 'This simulation includes finite volume method for diffusion and advection on a lattice.',
        'doc_func': get_diffusion_process_doc,
        'plot_func': plot_diffusion_process,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {}
    },
    'comets': {
        'description': 'This simulation combines dFBA at each lattice site with diffusion/advection to make a spatio-temporal FBA.',
        'doc_func': get_comets_doc,
        'plot_func': plot_comets,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {}
    },
    'particles': {
        'description': 'This simulation uses Brownian particles with mass moving randomly in space, and with a minimal reaction process inside of each particle uptaking or secreting from the field.',
        'doc_func': get_particles_doc,
        'plot_func': plot_particles_sim,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {}
    },
    'particle_comets': {
        'description': 'This simulation extends COMETS with particles that have internal minimal reaction processes.',
        'doc_func': get_particle_comets_doc,
        'plot_func': plot_particle_comets,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {}
    },
    'particle_dfba_fields': {
        'description': 'This simulation puts dFBA inside of the particles, interacting with external fields and adding biomass into the particle mass, reflected by the particle size.',
        'doc_func': get_particle_dfba_doc,
        'plot_func': plot_particle_dfba,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {
            'particle_model_id': 'ecoli core'
        },
        'plot_config': {'filename': 'particle_dfba_fields'}
    },
    'particle_dfba_comets': {
        'description': 'This simulation combines dFBA inside of the particles with COMETS, allowing particles to uptake and secrete from the external fields.',
        'doc_func': get_particle_dfba_comets_doc,
        'plot_func': plot_particle_dfba_comets,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {},
        'plot_config': {'filename': 'particle_dfba_comets'}
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run selected simulations.')
    parser.add_argument(
        '--tests', nargs='*', default=None,
        help='Names of tests to run. If none given, runs the full set.'
    )
    parser.add_argument('--output', default='out', help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output
    prepare_output_dir(output_dir)

    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)


    total_sim_time = 0.0  # To track simulation time only
    runtimes = {}

    test_names = list(SIMULATIONS.keys())
    tests_to_run = args.tests if args.tests else test_names
    print(f"\nSelected tests to run: {', '.join(tests_to_run)}\n")

    for name in tests_to_run:
        print(f"\n🚀 Running test: {name}")
        if name not in SIMULATIONS:
            print(f"Skipping unknown test: '{name}'")
            continue

        sim_info = SIMULATIONS[name]

        print("Creating document...")
        config = sim_info.get('config', {})
        doc = sim_info['doc_func'](core=core, config=config)

        print("Sending document...")
        runtime = sim_info.get('time', DEFAULT_RUNTIME_LONG)
        sim_start = time.time()
        results = run_composite_document(doc, core=core, name=name, time=runtime)
        sim_end = time.time()

        sim_elapsed = sim_end - sim_start
        runtimes[name] = sim_elapsed
        total_sim_time += sim_elapsed

        print("Generating plots...")
        plot_config = sim_info.get('plot_config', {})
        sim_info['plot_func'](results, doc.get('state', doc), config=plot_config)

        print(f"✅ Completed: {name} in {sim_elapsed:.2f} seconds")

    print(f"\nCompiling HTML report...")
    generate_html_report(
        output_dir,
        {k: v['config'] for k, v in SIMULATIONS.items()},
        {k: v['description'] for k, v in SIMULATIONS.items()},
        runtimes,
        total_sim_time
    )


if __name__ == '__main__':
    main()
