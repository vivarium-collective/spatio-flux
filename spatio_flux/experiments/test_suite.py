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
from pandas._libs import interval
from process_bigraph import allocate_core

from spatio_flux.library.tools import run_composite_document, prepare_output_dir, generate_html_report
from spatio_flux.plots.plot import ( plot_time_series, plot_particles_mass, plot_species_distributions_to_gif,
    plot_species_distributions_with_particles_to_gif, plot_particles, plot_model_grid,
    plot_snapshots_grid, fields_and_agents_to_gif, plot_particles_mass_with_submasses, plot_particle_traces
)
# from spatio_flux.plots.plot_core import assemble_type_figures, assemble_process_figures
from spatio_flux.processes.pymunk_particles import pymunk_simulation_to_gif
from spatio_flux.processes.monod_kinetics import MODEL_REGISTRY_KINETICS, get_monod_kinetics_process_from_config
from spatio_flux.processes.adapters import get_conc_count_adapter
from spatio_flux.processes import (
    get_spatial_many_dfba, get_spatial_dFBA_process, get_fields, get_fields_with_schema, get_field_names,
    get_diffusion_advection_process, get_brownian_movement_process, get_particle_exchange_process,
    initialize_fields, get_kinetic_particle_composition,
    get_dfba_particle_composition, get_community_dfba_particle_composition, get_particles_state, get_boundaries_process,
    MODEL_REGISTRY_DFBA, get_dfba_process_from_registry,
    get_kinetics_process_from_registry, get_spatial_many_kinetics,
    get_particle_divide_process, DIVISION_MASS_THRESHOLD,
    get_newtonian_particles_state, get_mass_total_step,
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SQUARE_BOUNDS = (50.0, 50.0)  # this is in um
SQUARE_BINS = (10, 10)

DEFAULT_BOUNDS = (40.0, 80.0)
DEFAULT_BINS = (10, 20)
DEFAULT_BINS_SMALL = (2, 4)
DEFAULT_ADVECTION = (0.0, 0.2)
DEFAULT_DIFFUSION = 0.5
DEFAULT_ADD_RATE = 0.1
DEFAULT_ADD_BOUNDARY = ['top', 'left', 'right']
DEFAULT_REMOVE_BOUNDARY = ['left', 'right']
DEFAULT_INITIAL_MIN_MAX = {
        'glucose': (10, 10),
        'acetate': (0, 0),
        'dissolved biomass': (0, 0.1),
        # 'detritus': (0, 0)
    }

DEFAULT_RUNTIME_SHORT = 10
DEFAULT_RUNTIME_LONG = 60
DEFAULT_RUNTIME_LONGER = 200

STANDARD_FIELD_COLORS = {
    "glucose": "#1f77b4",            # blue (matplotlib C0)
    "acetate": "#ff7f0e",            # orange (matplotlib C1)
    "formate": "#9467bd",             # purple (C4)
    "ammonium": "#bcbd22",            # yellow-green (C8)
    "biomass": "#2ca02c",            # green (C2)
    "dfba_biomass": "#2ca02c",       # same green (semantic match)
    "ecoli core biomass": "#2ca02c",
    "ecoli_core_biomass": "#2ca02c",
    "kinetic_biomass": "#1b9e77",     # dark teal-green (distinct, biomass-adjacent)
    "monod_biomass": "#98df8a",      # light green (C2 lighter)
    "dissolved biomass": "#17becf",  # teal (C9)
}



# ====================================================================
# Doc builders
# ====================================================================

# --- Kinetics Single ---------------------------------------------------

def get_kinetics_single_doc(
        core=None,
        config=None,
):
    model_id = config.get('model_id', 'overflow_metabolism')
    model_config = MODEL_REGISTRY_KINETICS[model_id]()
    interval = 0.1
    doc = {
        'monod_kinetics': get_monod_kinetics_process_from_config(model_config=model_config, interval=interval),
        'fields': {
            'glucose': 10,
            'acetate': 0,
            'biomass': 0.1
        }
    }
    return doc

def plot_kinetics_single(results, state, config=None, filename='kinetics_single_timeseries'):
    config = config or {}
    field_names = list(state['fields'].keys())
    filename = config.get('filename', 'kinetics_single_timeseries')
    plot_time_series(results, field_names=field_names, out_dir='out', filename=f'{filename}.png', title='Monod kinetics',
                     figsize=(5, 4),
                     time_units="min",
                     y_label_base="Concentration / Biomass",
                     field_units={"glucose": "mM", "acetate": "mM", "biomass": "gDW"},
                     field_colors=STANDARD_FIELD_COLORS,
                     legend_kwargs={"fontsize": 8, "loc": "best"},
                     )


# --- DFBA Single ---------------------------------------------------

def get_dfba_single_doc(
        core=None,
        config=None,
):
    model_id = config.get('model_id', 'ecoli core')
    biomass_id = 'biomass'
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
    plot_time_series(results, field_names=field_names, out_dir='out', filename=f'{filename}.png', title=f'dFBA',
                     figsize=(4.5, 3.5),
                     time_units="min",
                     y_label_base="Concentration / Biomass",
                     field_units={"glucose": "mM", "acetate": "mM",
                                  "formate": "mM", "ammonium": "mM",
                                  "biomass": "gDW", "kinetic_biomass": "gDW"
                                  },
                     field_colors=STANDARD_FIELD_COLORS,
                     legend_kwargs={"fontsize": 8, "loc": "best"},
                     )

# --- Multiple DFBAs ---------------------------------------------------

def get_community_dfba_doc(core=None, config=None):
    dt = 1.0

    # set up the dfba processes
    model_ids = list(MODEL_REGISTRY_DFBA.keys())
    dfbas = {}
    for model_id, spec in MODEL_REGISTRY_DFBA.items():
        process_id = f'{model_id} dFBA'
        dfba_process = get_dfba_process_from_registry(
            model_id=model_id,
            biomass_id=model_id,
            path=['fields'],
            interval=dt,
        )
        dfbas[process_id] = dfba_process

    initial_biomass = {organism: 0.1 for organism in model_ids}

    # set up the kinetic process
    kinetic_model_id = 'acetate_only'  #'overflow_metabolism'
    kinetic_biomass_id = 'monod biomass'
    kinetic_model_config = MODEL_REGISTRY_KINETICS[kinetic_model_id]()
    initial_biomass[kinetic_biomass_id] = 0.1

    # fields
    field_names = get_field_names(MODEL_REGISTRY_DFBA)
    more_fields = {mol_id: 0.1 for mol_id in field_names if mol_id not in ['glucose', 'acetate']}
    doc = {
        **dfbas,
        'monod_kinetics': get_monod_kinetics_process_from_config(model_config=kinetic_model_config, biomass_id=kinetic_biomass_id, interval=dt),
        'fields': {
            'glucose': 10,
            'acetate': 0,
            **more_fields,
            **initial_biomass
        }
    }
    return doc

def plot_community_dfba(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'dfba_multi_timeseries.png')
    species_ids = [state[s]['inputs']['biomass'][-1] for s in state.keys() if s not in ['fields', 'emitter', 'global_time']]
    plot_time_series(results, field_names=species_ids, log_scale=True, normalize=True, out_dir='out', filename=filename,
                     title='hybrid community',
                     figsize=(4.5, 3.5),
                     time_units="min",
                     y_label_base="Concentration / Biomass",
                     # field_units={"glucose": "mM", "acetate": "mM", "dfba_biomass": "gDW"},
                     field_colors=STANDARD_FIELD_COLORS,
                     legend_kwargs={"fontsize": 8, "loc": "best"},
                     )


# --- DFBA-Monod Community ------------------------------------------------
def get_dfba_kinetics_community_doc(core=None, config=None):
    dfba_model_id = 'ecoli core'
    kinetic_model_id = 'acetate_only'  #'overflow_metabolism'
    dfba_biomass_id = 'dfba biomass'
    kinetic_biomass_id = 'kinetic biomass'

    kinetic_model_config = MODEL_REGISTRY_KINETICS[kinetic_model_id]()
    doc = {
        'dFBA': get_dfba_process_from_registry(model_id=dfba_model_id, biomass_id=dfba_biomass_id, path=['fields']),
        'monod_kinetics': get_monod_kinetics_process_from_config(model_config=kinetic_model_config, biomass_id=kinetic_biomass_id),
        'fields': {
            'glucose': 10,
            'acetate': 0,
            dfba_biomass_id: 0.01,
            kinetic_biomass_id: 0.01,
        }
    }
    return doc

def plot_dfba_kinetics_community(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'dfba_kinetics_community')
    species_ids = ['glucose', 'acetate', 'dfba_biomass', 'kinetic_biomass']
    plot_time_series(results, field_names=species_ids,
                     # log_scale=True,
                     # normalize=True,
                     out_dir='out', filename=filename,
                     figsize=(4.5, 3.5),
                     time_units="min",
                     y_label_base="Concentration / Biomass",
                     field_units={"glucose": "mM", "acetate": "mM", "dfba_biomass": "gDW"},
                     field_colors=STANDARD_FIELD_COLORS,
                     legend_kwargs={"fontsize": 8, "loc": "best"},
                     )


# --- Many DFBA Spatial ---------------------------------------------------

def get_spatial_many_dfba_doc(core=None, config=None):
    dissolved_model_file = 'ecoli core'
    mol_ids = ['glucose', 'acetate', 'dissolved biomass']
    initial_min_max = {'glucose': (0, 20), 'acetate': (0, 0), 'dissolved biomass': (0, 0.1)}
    n_bins = DEFAULT_BINS_SMALL
    return {
        'fields': get_fields_with_schema(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max),
        'spatial_dFBA': get_spatial_many_dfba(model_id=dissolved_model_file, mol_ids=mol_ids, n_bins=n_bins)
    }

def plot_spatial_many_dfba(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'spatial_many_dfba')
    plot_time_series(results, coordinates=[(0, 0), (1, 1), (1, 3)], out_dir='out', filename=f'{filename}_timeseries.png')
    plot_species_distributions_to_gif(results, out_dir='out', filename=f'{filename}_video.gif')

# --- DFBA Spatial Process ---------------------------------------------

def build_model_grid(n_bins, model_positions=None):
    """
    Build a model_grid array
    """
    nx, ny = n_bins  # x bins, y bins
    out_of_bounds = []

    # Validate positions (x, y)
    if model_positions:
        for model_id, positions in model_positions.items():
            for (x, y) in positions:
                if not (0 <= x < nx and 0 <= y < ny):
                    out_of_bounds.append((model_id, (x, y)))

    if out_of_bounds:
        raise ValueError(
            f"The following positions are out of bounds for grid n_bins={n_bins}: {out_of_bounds}"
        )

    # Build empty grid: rows = y, cols = x
    model_grid = [['' for _ in range(nx)] for _ in range(ny)]

    # Fill grid using model_positions (x, y → [y][x])
    if model_positions:
        for model_id, positions in model_positions.items():
            for (x, y) in positions:
                model_grid[y][x] = model_id

    return model_grid


def get_spatial_dfba_process_doc(core=None, config=None):
    # make the fields
    mol_ids = ['glucose', 'acetate', 'glycolate', 'ammonium', 'formate', 'glutamate', 'serine',
               'dissolved biomass']
    initial_min_max = {'glucose': (10, 10), 'glycolate': (10, 10), 'ammonium': (10, 10), 'formate': (10, 10),
                       'glutamate': (10, 10), 'serine': (0, 0),
                       'acetate': (0, 0), 'dissolved biomass': (0.1, 0.1)}
    n_bins = (5, 6)
    initial_fields = {}
    initial_fields = get_fields(n_bins, mol_ids, initial_min_max, initial_fields)

    bins_x, bins_y = n_bins
    horizontal_gradient = np.linspace(0, 20, bins_x).reshape(1, -1)
    initial_fields['glucose'] = np.repeat(horizontal_gradient, bins_y, axis=0)  # (ny, nx)

    model_positions = {
        'ecoli core': [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        'ecoli': [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
        'cdiff': [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        'pputida': [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)],
        'yeast': [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
        'llactis': [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)],
    }
    model_grid = build_model_grid(n_bins=n_bins, model_positions=model_positions)

    # make the spatial dfba with different models and parameters
    spatial_dfba_config = {
        'n_bins': n_bins,
        'models': MODEL_REGISTRY_DFBA,
        'model_grid': model_grid,
        'mol_ids': mol_ids,
    }
    doc = {
        'fields': initial_fields,
        'spatial_dFBA': get_spatial_dFBA_process(config=spatial_dfba_config)
    }
    return doc

def plot_dfba_process_spatial(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'spatial_dfba_process')
    model_grid = state['spatial_dFBA']['config']['model_grid']
    plot_time_series(results, coordinates=[(0, 0), (1, 1), (2, 2)], out_dir='out', filename=f'{filename}_timeseries.png')
    plot_model_grid(model_grid, title='model grid', show_border_coords=True, out_dir='out', filename=f'{filename}_model_grid.png')
    plot_species_distributions_to_gif(results, out_dir='out',
                                      species_to_show=['glucose', 'acetate', 'ammonium', 'formate', 'glutamate', 'dissolved biomass'],
                                      filename=f'{filename}_video.gif')

# --- Diffusion Advection-----------------------------------------------

def get_diffusion_process_doc(core=None, config=None):
    mol_ids = ['glucose', 'dissolved biomass']
    advection_coeffs = {'dissolved biomass': DEFAULT_ADVECTION}
    diffusion_coeffs = {'glucose': DEFAULT_DIFFUSION/10, 'dissolved biomass': DEFAULT_DIFFUSION/10}
    n_bins = DEFAULT_BINS
    bounds = DEFAULT_BOUNDS
    # initialize fields
    glc_field = np.random.uniform(low=0.1,high=2,size=(n_bins[1], n_bins[0]))
    biomass_field = np.zeros((n_bins[1], n_bins[0]))
    biomass_field[4:5,:] = 10
    return {
        'fields': {'dissolved biomass': biomass_field, 'glucose': glc_field},
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
    mol_ids = ['glucose', 'acetate', 'dissolved biomass']
    n_bins = SQUARE_BINS    # (nx, ny)
    bounds = SQUARE_BOUNDS  # (xmax, ymax)
    diffusion_coeffs = {'glucose': 0.0, 'acetate': 1e-1, 'dissolved biomass': 1e-2}
    advection_coeffs = {'dissolved biomass': DEFAULT_ADVECTION}
    nx, ny = n_bins
    shape = (ny, nx)  # numpy arrays are (rows=y, cols=x)

    # Fields
    acetate_field = np.zeros(shape, dtype=float)
    # vertical gradient in y (rows): low at top row, high at bottom row
    glc_y = np.linspace(0.01, 10.0, ny, dtype=float)[:, None]  # (ny, 1)
    glc_field = np.repeat(glc_y, nx, axis=1)  # (ny, nx)
    # biomass strip: top row(s), middle half of x
    biomass_field = np.zeros(shape, dtype=float)
    x0 = nx // 2
    biomass_field[0:1, x0-1:x0+1] = 0.1

    initial_fields = {
        'dissolved biomass': biomass_field,
        'glucose': glc_field,
        'acetate': acetate_field,
    }

    spatial_dfba = get_spatial_many_dfba(model_id=dissolved_model_id, mol_ids=mol_ids, n_bins=n_bins, path=['fields'])
    doc = {
        **spatial_dfba,
        'fields': get_fields_with_schema(n_bins=n_bins, mol_ids=mol_ids, initial_fields=initial_fields),
        'diffusion': get_diffusion_advection_process(
            bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, advection_coeffs=advection_coeffs, diffusion_coeffs=diffusion_coeffs)
    }
    return doc

def plot_comets(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'comets')
    n_snapshots = config.get('n_snapshots', 5)
    n_bins = state['diffusion']['config']['n_bins']
    bounds = state['diffusion']['config']['bounds']

    # get coordinates for time series
    nx, ny = n_bins
    x0 = nx // 2
    coord = [(0, x0), (x0, x0), (nx-1, x0)]

    plot_time_series(results, out_dir='out', filename=f'{filename}_timeseries.png', coordinates=coord,
                     label_fontsize=20)
    plot_snapshots_grid(results, field_names=['glucose', 'acetate', 'dissolved biomass'],
                        n_snapshots=n_snapshots, bounds=bounds, out_dir='out', filename=f'{filename}_snapshots.png',
                        time_units="min",
                        wspace=0.1,
                        hspace=0.1,
                        col_width=1.8,
                        row_height=2.0,
                        )
    plot_species_distributions_to_gif(results, out_dir='out', filename=f'{filename}_video.gif')

# --- Particles -----------------------------------------------------------

def get_brownian_particles_alone_doc(core=None, config=None):
    n_bins = SQUARE_BINS
    bounds = SQUARE_BOUNDS
    n_particles = 1
    time_interval = 0.1
    diffusion_rate = DEFAULT_DIFFUSION
    add_rate = 0.01
    doc = {
        'state': {
            'particles': get_particles_state(n_particles=n_particles, bounds=bounds),
            'brownian_movement': get_brownian_movement_process(bounds=bounds, diffusion_rate=diffusion_rate, interval=time_interval),
            'enforce_boundaries': get_boundaries_process(particle_process_name='brownian_movement', bounds=bounds, add_rate=add_rate),
        },
    }
    return doc


def plot_particles_sim(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'particles')
    bounds = state['brownian_movement']['config']['bounds']
    history = [step['particles'] for step in results]
    plot_particles(history=history, env_size=((0, bounds[0]), (0, bounds[1])), out_dir='out', filename=f'{filename}_video.gif')
    plot_particles_mass(results, out_dir='out', filename=f'{filename}_mass.png')
    plot_species_distributions_with_particles_to_gif(results, out_dir='out', filename=f'{filename}_video.gif', bounds=bounds)
    plot_snapshots_grid(results, field_names=['glucose', 'acetate'], n_snapshots=4, bounds=bounds,
                        out_dir='out', filename=f'{filename}_snapshots.png',
                        time_units="min",
                        wspace=0.1,
                        hspace=0.1,
                        col_width=1.8,
                        row_height=2.0,
                        particles_row='separate'
                        )
    plot_particle_traces(history=history, bounds=bounds, out_dir="out", filename=f'{filename}_particles_traces.png',
                         radius_scaling=0.1, min_brightness=0.1, legend=False, units="µm")

# --- Particles with Monod Kinetics -----------------------------------------------------------

def get_br_particles_kinetics_doc(core=None, config=None):
    division_mass_threshold = config.get('division_mass_threshold', DIVISION_MASS_THRESHOLD) # divide at mass 5.0
    initial_min_max = {'glucose': (5.0, 10.0), 'acetate': (0, 0)}
    initial_min_max = {'glucose': (5.0, 10.0), 'acetate': (0, 0)}

    model_id = config.get('model_id', 'overflow_metabolism')
    particle_config = MODEL_REGISTRY_KINETICS[model_id]()

    mol_ids = list(initial_min_max.keys())
    n_bins = SQUARE_BINS
    bounds = SQUARE_BOUNDS
    n_particles = 1
    particle_diffusion = DEFAULT_DIFFUSION/2
    particle_advection = (0,0) #DEFAULT_ADVECTION
    add_rate = 0.0
    return {
        'state': {
            'fields': get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max),
            'particles': get_particles_state(n_particles=n_particles, bounds=bounds),
            'brownian_movement': get_brownian_movement_process(bounds=bounds,diffusion_rate=particle_diffusion, advection_rate=particle_advection),
            'enforce_boundaries': get_boundaries_process(particle_process_name='brownian_movement', bounds=bounds, add_rate=add_rate),
            'particle_exchange': get_particle_exchange_process(n_bins=n_bins, bounds=bounds),
            'particle_division': get_particle_divide_process(division_mass_threshold=division_mass_threshold),
        },
        # put kinetic metabolism in the particles
        'schema': get_kinetic_particle_composition(core=core, config=particle_config)
    }

# --- dFBA-Particles ---------------------------------------------------

def get_br_particles_dfba_doc(core=None, config=None):
    particle_model_id = config.get('particle_model_id', 'ecoli core')
    division_mass_threshold=config.get('division_mass_threshold', DIVISION_MASS_THRESHOLD) # divide at mass 5.0
    mol_ids = ['glucose', 'acetate']

    bounds = SQUARE_BOUNDS
    n_bins = SQUARE_BINS
    nx, ny = n_bins
    shape = (ny, nx)  # numpy arrays are (rows=y, cols=x)
    acetate_field = np.zeros(shape, dtype=float)
    glc_y = np.linspace(0.01, 10.0, ny, dtype=float)[:, None]  # (ny, 1)
    glc_field = np.repeat(glc_y, nx, axis=1)  # (ny, nx)
    initial_fields = {'glucose': glc_field, 'acetate': acetate_field}

    n_particles = 1
    add_rate = 0.1
    particle_diffusion = DEFAULT_DIFFUSION
    particle_advection = DEFAULT_ADVECTION

    return {
        'state': {
            'fields': get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_fields=initial_fields),
            'particles': get_particles_state(n_particles=n_particles, bounds=bounds),
            'brownian_movement': get_brownian_movement_process(bounds=bounds, diffusion_rate=particle_diffusion, advection_rate=particle_advection),
            'enforce_boundaries': get_boundaries_process(particle_process_name='brownian_movement', bounds=bounds, add_rate=add_rate),
            'particle_exchange': get_particle_exchange_process(n_bins=n_bins, bounds=bounds),
            'particle_division': get_particle_divide_process(division_mass_threshold=division_mass_threshold),
        },
        'schema': get_dfba_particle_composition(model_file=particle_model_id)
    }

def plot_particle_dfba(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'particle_dfba')
    n_bins = state['particle_exchange']['config']['n_bins']
    bounds = state['brownian_movement']['config']['bounds']
    plot_time_series(results, field_names=['glucose', 'acetate'], coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename=f'{filename}_timeseries.png')
    plot_particles_mass(results, out_dir='out', filename=f'{filename}_mass.png')
    plot_snapshots_grid(results, field_names=['glucose', 'acetate'],
                        n_snapshots=5, bounds=bounds, particles_row='separate',
                        out_dir='out', filename=f'{filename}_snapshots.png')
    plot_species_distributions_with_particles_to_gif(results, bounds=bounds, out_dir='out', filename=f'{filename}_video.gif')


# --- Particle-COMETS ----------------------------------------------------

def get_comets_br_particles_kinetics_doc(core=None, config=None):
    division_mass_threshold=config.get('division_mass_threshold', DIVISION_MASS_THRESHOLD) # divide at mass 5.0

    dissolved_model_id = 'ecoli core'
    mol_ids = ['glucose', 'acetate', 'dissolved biomass']
    particle_config = MODEL_REGISTRY_KINETICS['overflow_metabolism']()
    n_bins = SQUARE_BINS
    bounds = SQUARE_BOUNDS
    particle_diffusion = DEFAULT_DIFFUSION
    particle_advection = DEFAULT_ADVECTION
    n_particles = 1
    add_rate = 0.1

    fields = get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=DEFAULT_INITIAL_MIN_MAX)
    n_grid = (n_bins[1], n_bins[0])  # shape (ny, nx)
    fields['dissolved biomass'] = np.zeros(n_grid)  # Initialize biomass field to zero
    fields['dissolved biomass'][0, int(n_grid[0]/4):int(3*n_grid[0]/4)] = 0.1  # Add some biomass in the first row

    # make the spatial dfba with different models and parameters
    spatial_dFBA_config = {
        'n_bins': n_bins,
        'models': MODEL_REGISTRY_DFBA,
        'mol_ids': mol_ids,
    }

    return {
        'state': {
            'fields': fields,
            'particles': get_particles_state(n_particles=n_particles, bounds=bounds, mass_range=(1E0, 1E1)),
            'spatial_dFBA': get_spatial_dFBA_process(config=spatial_dFBA_config, model_id=dissolved_model_id),
            'diffusion': get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids),
            'brownian_movement': get_brownian_movement_process(bounds=bounds, advection_rate=particle_advection, diffusion_rate=particle_diffusion),
            'enforce_boundaries': get_boundaries_process(particle_process_name='brownian_movement', bounds=bounds, add_rate=add_rate),
            'particle_exchange': get_particle_exchange_process(n_bins=n_bins, bounds=bounds),
            'particle_division': get_particle_divide_process(division_mass_threshold=division_mass_threshold),
        },
        'schema': get_kinetic_particle_composition(core, config=particle_config)
    }

def plot_kinetic_particle_comets(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'particle_comets')
    n_snapshots = config.get('n_snapshots', 5)
    bounds = state['brownian_movement']['config']['bounds']
    n_bins = state['particle_exchange']['config']['n_bins']
    plot_time_series(results, coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename=f'{filename}_timeseries.png')
    plot_particles_mass(results, out_dir='out', filename=f'{filename}_mass.png')
    plot_snapshots_grid(results, field_names=['glucose', 'acetate'],
                        n_snapshots=n_snapshots, bounds=bounds, particles_row='separate',
                        out_dir='out', filename=f'{filename}_snapshots.png')
    plot_species_distributions_with_particles_to_gif(results, out_dir='out', filename=f'{filename}_video.gif', bounds=bounds)


# --- dFBA-Particles-COMETS ---------------------------------------------------

def get_comets_br_particles_dfba_doc(core=None, config=None):
    """ Build a composite document for spatial dFBA fields + particle dynamics."""
    # Config / IDs
    config = config or {}
    particle_model_id = config.get("particle_model_id", "ecoli core")
    dissolved_model_id = config.get("dissolved_model_id", "ecoli core")

    # Particle division
    division_mass_threshold = config.get("division_mass_threshold", 3)

    # Spatial grid / bounds
    mol_ids = ["glucose", "acetate", "dissolved biomass"]

    bounds = SQUARE_BOUNDS
    n_bins = SQUARE_BINS
    nx, ny = n_bins
    shape = (ny, nx)  # numpy arrays are (rows=y, cols=x)

    # Initial fields
    acetate_field = np.zeros(shape, dtype=float)
    glc_y = np.linspace(0.01, 10.0, ny, dtype=float)[:, None]   # (ny, 1)
    glc_field = np.repeat(glc_y, nx, axis=1)                    # (ny, nx)
    biomass_field = np.zeros(shape, dtype=float)
    x0, x1 = nx // 4, 3 * nx // 4
    biomass_field[0:1, x0:x1] = 0.1
    initial_fields = {
        "dissolved biomass": biomass_field,
        "glucose": glc_field,
        "acetate": acetate_field,
    }

    # Process parameters
    advection_coeffs = {"dissolved biomass": DEFAULT_ADVECTION}

    n_particles = config.get("n_particles", 1)
    add_rate = config.get("add_rate", 0.3)
    particle_advection = config.get("particle_advection", (0, -0.2))
    particle_diffusion = config.get("particle_diffusion", DEFAULT_DIFFUSION)

    # Subsystems / state blocks
    spatial_dfba = get_spatial_many_dfba(n_bins=n_bins, model_id=dissolved_model_id, mol_ids=mol_ids, path=["fields"])

    state = {
        # Put the dfba processes at the top level
        **spatial_dfba,
        # Field substrate state + transport
        "fields": get_fields_with_schema(n_bins=n_bins, mol_ids=mol_ids, initial_fields=initial_fields),
        "diffusion": get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, advection_coeffs=advection_coeffs),
        # Particles + movement + boundary enforcement + exchange + division
        "particles": get_particles_state(n_particles=n_particles, bounds=bounds),
        "brownian_movement": get_brownian_movement_process(bounds=bounds, advection_rate=particle_advection, diffusion_rate=particle_diffusion),
        "enforce_boundaries": get_boundaries_process(particle_process_name="brownian_movement", bounds=bounds, add_rate=add_rate),
        "particle_exchange": get_particle_exchange_process(n_bins=n_bins, bounds=bounds),
        "particle_division": get_particle_divide_process(division_mass_threshold=division_mass_threshold),
    }

    # Document
    doc = {
        "state": state,
        "schema": get_dfba_particle_composition(model_file=particle_model_id),
    }
    return doc


def plot_particle_dfba_comets(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'particle_dfba_comets')
    n_snapshots = config.get('n_snapshots', 5)
    n_bins = state['particle_exchange']['config']['n_bins']
    bounds = state['brownian_movement']['config']['bounds']
    plot_time_series(results, field_names=['glucose', 'acetate', 'dissolved biomass'], coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename=f'{filename}_timeseries.png')
    plot_particles_mass(results, out_dir='out', filename=f'{filename}_mass.png')
    plot_snapshots_grid(results, field_names=['glucose', 'acetate'],
                        n_snapshots=n_snapshots, bounds=bounds, particles_row='separate',
                        out_dir='out', filename=f'{filename}_snapshots.png')
    plot_species_distributions_with_particles_to_gif(results, bounds=bounds, out_dir='out', filename=f'{filename}_video.gif')


# ---- PYMUNK PARTICLES ------------------------------------------------

def get_newtonian_particles_process(config=None):
    return {
        '_type': 'process',
        'address': 'local:PymunkParticleMovement',
        'config': config,
        'inputs': {
            'particles': ['particles'],
        },
        'outputs': {
            'particles': ['particles'],
        }
    }

def get_newtonian_particles_doc(core=None, config=None):
    n_particles = config.get('n_particles', 1)
    bounds = SQUARE_BOUNDS #(100.0, 300.0)
    add_rate = 0.02

    # run simulation
    config = {
        'gravity': -0.2,  # -9.81,
        'elasticity': 0.1,
        'bounds': bounds,
        'jitter_per_second': 0.5,
        'damping_per_second': .998,
    }

    processes = {
        'newtonian_particles': get_newtonian_particles_process(config=config),
        'enforce_boundaries': get_boundaries_process(particle_process_name='newtonian_particles', bounds=bounds, add_rate=add_rate),
    }

    initial_state = {'particles': get_newtonian_particles_state(
        n_particles=n_particles,
        bounds=config['bounds'],
    )}

    # complete document
    return {
        'state': {
            **initial_state,
            **processes,
        },
    }

def plot_newtonian_particles(results, state, config=None):
    filename = config.get('filename', 'newtonian_particles')
    pymunk_config = state['newtonian_particles']['config']
    bounds = pymunk_config['bounds']
    history = [step['particles'] for step in results]
    plot_particles_mass(results, out_dir='out', filename=f'{filename}_mass.png')
    pymunk_simulation_to_gif(results, filename=f'{filename}_video.gif', config=pymunk_config, agents_key='particles')
    plot_particle_traces(history=history, bounds=bounds, out_dir="out", filename=f'{filename}_particles_traces.png',
                         radius_scaling=0.1, min_brightness=0.1,)

# --- PYMUNK COMETS ------------------------------------------------

def get_newtonian_particle_comets_doc(core=None, config=None):
    config = config or {}

    division_mass_threshold = 0.5
    add_rate = 0.01  # 0.02

    particle_model_id = config.get('particle_model_id', 'ecoli core')
    dissolved_model_id = config.get('dissolved_model_id', 'ecoli core')
    mol_ids = ['glucose', 'acetate', 'dissolved biomass']
    initial_min_max = {'glucose': (0.5, 2), 'acetate': (0, 0), 'dissolved biomass': (0, 0.1)}
    bounds = config.get('bounds', SQUARE_BOUNDS)
    n_bins = config.get('n_bins', tuple(n * 2 for n in SQUARE_BINS))
    advection_coeffs = {'dissolved biomass': DEFAULT_ADVECTION}
    fields = get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max)

    # pymunk
    n_particles = config.get('n_particles', 2)

    # run simulation
    particle_config = {
        'gravity': -1.0,  #-0.2, -9.81,
        'elasticity': 0.1,
        'bounds': bounds,
        'jitter_per_second': 1e-1,   # 0.01,
        'damping_per_second': 1e-1,  #5,  #.995,
    }
    boundary_config = {
        'add_rate': add_rate,
        'boundary_to_remove': [],  # ['right', 'left'],
        'new_particle_radius_range': (0.05, 0.2),
        'new_particle_mass_range': (0.001, 0.01),
    }

    doc = {
        'state': {
            'fields': fields,
            'diffusion': get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, advection_coeffs=advection_coeffs),
            # 'spatial_dFBA': get_spatial_many_dfba(n_bins=n_bins, model_file=dissolved_model_id),
            'spatial_kinetics': get_spatial_many_kinetics(model_id='single_substrate_assimilation', n_bins=n_bins, mol_ids=mol_ids),
            'particles': get_newtonian_particles_state(n_particles=n_particles, bounds=bounds),
            'newtonian_particles': get_newtonian_particles_process(config=particle_config),
            'particle_exchange': get_particle_exchange_process(n_bins=n_bins, bounds=bounds),
            'particle_division': get_particle_divide_process(division_mass_threshold=division_mass_threshold),
            'enforce_boundaries': get_boundaries_process(
                particle_process_name='newtonian_particles', bounds=bounds,
                boundary_to_add=('top',),
                add_rate=boundary_config['add_rate'],
                mass_range=(1e-3, 1e-2)
            ),
        },
        'schema': get_dfba_particle_composition(model_file=particle_model_id)
    }
    return doc


def plot_newtonian_particle_comets(results, state, config=None):
    filename = config.get('filename', 'newtonian_particle_comets')
    pymunk_config = state.get('newtonian_particles', {}).get('config', {})
    n_snapshots = config.get('n_snapshots', 5)
    if 'diffusion' in state:
        bounds = state['diffusion']['config']['bounds']
        n_bins = state['diffusion']['config']['n_bins']
    elif 'particle_exchange' in state:
        bounds = state['particle_exchange']['config']['bounds']
        n_bins = state['particle_exchange']['config']['n_bins']
    else:
        raise ValueError

    # TODO -- replace this hack with proper path management
    for result in results:
        if 'lattice' in result:
            lattice = result.get('lattice')
            if lattice is None:
                continue
            fields = lattice.get('fields')
            if fields is not None:
                result['fields'] = fields

    particles_row = config.get("particles_row", "overlay")
    plot_time_series(results, field_names=['glucose', 'acetate', 'dissolved biomass'],
                     coordinates=[(0, 0), (n_bins[0]-1, n_bins[1]-1)], out_dir='out', filename=f'{filename}_timeseries.png')
    plot_particles_mass(results, out_dir='out', filename=f'{filename}_mass.png')
    plot_particles_mass_with_submasses(results, out_dir='out', filename=f'{filename}_mass_submasses.png')

    submass_colors = {
        "ecoli_1": "#1f77b4",  # blue
        "ecoli_2": "#d62728",  # red
    }
    if pymunk_config:
        fields_and_agents_to_gif(data=results, config=pymunk_config, agents_key='particles', fields_key='fields',
            filename=f'{filename}_video.gif', out_dir='out', figure_size_inches=(10, 6),
                                 show_agent_submasses=True,
                                 submass_color_map=submass_colors,
                                 draw_submass_outline=True,
                                 draw_submass_legend=False,
                                 )
    # snapshot plot
    xmax, ymax = bounds
    world_aspect = ymax / xmax  # e.g. 3.0 for (100,300)
    plot_snapshots_grid(results,
                        bounds=bounds,
                        out_dir='out',
                        filename=f'{filename}_snapshots.png',
                        field_names=['glucose', 'acetate', 'dissolved biomass'],
                        n_snapshots=n_snapshots,
                        particles_row=particles_row,
                        time_units="min",
                        wspace=0.02,
                        hspace=0.08,
                        row_height=2.0,
                        col_width=2.0 / world_aspect,
                        cbar_width=0.04,  # slimmer colorbars
                        show_particle_submasses=True,
                        submass_draw_legend=True,
                        submass_color_map=submass_colors,
                        )


# --- spatio-flux reference composite simulation ---------------------------------------------------

def get_reference_composite_doc(core=None, config=None):
    user_cfg = config or {}
    bounds = user_cfg.get("bounds", SQUARE_BOUNDS)
    n_bins = user_cfg.get("n_bins", SQUARE_BINS)
    depth = user_cfg.get("depth", 1.0)

    # High-level knobs
    division_mass_threshold = 0.4
    add_rate = 0.0
    initial_submasses = {
        'ecoli_1': 0.1,
        'ecoli_2': 0.1
    }

    # Spatial fields state
    glucose_level = 5.0
    biomass_id = "dissolved biomass"
    mol_ids = ["glucose", "acetate", biomass_id]
    initial_min_max = {"glucose": (glucose_level, glucose_level), "acetate": (0.0, 0.0), biomass_id: (0.1, 0.2)}

    # diffusion process config
    diffusion_coeffs = {'glucose': 1e-1, 'acetate': 1e-1, biomass_id: 1e-1}
    advection_coeffs = {
        # biomass_id: (0.0, 0.2), # dissolved biomass floats to the top
        # 'acetate': (0.0, -0.5)  # acetates sinks
    }
    diffusion_boundary_config = {
        "default": {"x": {"type": "periodic"}, "y": {"type": "neumann"}},
        "glucose": {"top": {"type": "dirichlet", "value": glucose_level}},
        "acetate": {"bottom": {"type": "dirichlet", "value": glucose_level}}}

    # Particles + physics config
    n_particles = user_cfg.get("n_particles", 1)
    physics_cfg = {"gravity": -1.0,
                   "elasticity": 0.1,
                   "bounds": bounds,
                   "jitter_per_second": 1e-2,
                   "damping_per_second": 0.95,   # viscous
                   "friction": 0.9}
    boundary_cfg = {"add_rate": add_rate}

    # dFBA Models for community simulation within particles
    models = {
        "ecoli_1": {
            'model_file': 'textbook',
            'substrate_update_reactions': {'glucose': 'EX_glc__D_e', 'acetate': 'EX_ac_e',},
            'kinetic_params': {'glucose': (0.1, 2), 'acetate': (1.0, 0.1)},
            'bounds': {
                'EX_o2_e': {'lower': -2, 'upper': None},
                'ATPM': {'lower': 1, 'upper': 1}
            },
        },
        "ecoli_2": {
            'model_file': 'textbook',
            'substrate_update_reactions': {'glucose': 'EX_glc__D_e', 'acetate': 'EX_ac_e',},
            'kinetic_params': {'glucose': (1.0, 0.1), 'acetate': (0.01, 1)},
            'bounds': {
                'EX_o2_e': {'lower': -2, 'upper': None},
                'ATPM': {'lower': 1, 'upper': 1}
            }
        }
    }

    # State
    fields = get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max)
    particles = get_newtonian_particles_state(n_particles=n_particles, bounds=bounds)

    # put mass metabolism inside the particles
    for pid, internal in particles.items():
        internal['sub_masses'] = initial_submasses.copy()

    # Processes
    diffusion = get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, diffusion_coeffs=diffusion_coeffs, advection_coeffs=advection_coeffs, boundary_conditions=diffusion_boundary_config)
    spatial_kinetics = get_spatial_many_kinetics(model_id="low_yield_glucose_overflow", biomass_id=biomass_id, n_bins=n_bins, mol_ids=mol_ids, path=["fields"])
    newtonian_particles = get_newtonian_particles_process(config=physics_cfg)
    particle_exchange = get_particle_exchange_process(
        n_bins=n_bins, bounds=bounds, concentration_fields_path=['lattice', 'fields'], exchange_fields_path=['lattice', 'exchanges'])
    particle_division = get_particle_divide_process(division_mass_threshold=division_mass_threshold, submass_split_mode='random')
    enforce_boundaries = get_boundaries_process(particle_process_name="newtonian_particles", bounds=bounds, add_rate=boundary_cfg["add_rate"])

    # adapters
    conc_count_adapter = get_conc_count_adapter(conc_path=['fields'], n_bins=n_bins, bounds=bounds, depth=depth)

    # composite schema
    schema = get_community_dfba_particle_composition(models=models)

    doc = {
        "state": {
            "lattice": {
                # "bin_volume": 1.0,
                "fields": fields,
                "exchanges": {mol_id: np.zeros_like(f) for mol_id, f in fields.items()},
                **spatial_kinetics,  # put them at the top level
                "diffusion": diffusion,
                "conc_count_adapter": conc_count_adapter,
            },
            "particles": particles,
            "particle_exchange": particle_exchange,
            "particle_division": particle_division,
            "enforce_boundaries": enforce_boundaries,
            "newtonian_particles": newtonian_particles,
        },
        "schema": schema,
    }
    return doc



# ==================================================
# Functions for running tests and generating reports
# ==================================================

SIMULATIONS = {
    # ---- Metabolism-only models -------------------------------------------
    'monod_kinetics': {
        'description': 'Field-only baseline: Monod uptake/growth on a well-mixed substrate pool (no spatial lattice, no particles). Use to sanity-check kinetics + mass balance.',
        'doc_func': get_kinetics_single_doc,
        'plot_func': plot_kinetics_single,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {'model_id': 'overflow_metabolism'},
        'plot_config': {'filename': 'monod_kinetics'}
    },
    'ecoli_core_dfba': {
        'description': 'Single-cell metabolism baseline: dynamic FBA for E. coli core with external glucose/acetate and biomass over time (no space, no particles).',
        'doc_func': get_dfba_single_doc,
        'plot_func': plot_dfba_single,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {'model_id': 'ecoli core', 'initial_fields': {'glucose': 10, 'acetate': 0}},
        'plot_config': {'filename': 'ecoli_core_dfba'}
    },
    'ecoli_dfba': {
        'description': 'Single-cell metabolism (large model): dynamic FBA using iAF1260 with tracked extracellular fields (e.g., glucose/formate) and biomass. Stress-tests solver + exchange wiring.',
        'doc_func': get_dfba_single_doc,
        'plot_func': plot_dfba_single,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {'model_id': 'ecoli', 'initial_fields': {'glucose': 10, 'formate': 5}},
        'plot_config': {'filename': 'ecoli_dfba'}
    },
    'yeast_dfba': {
        'description': 'Single-cell metabolism (yeast): dynamic FBA using iMM904 with extracellular glucose and biomass. Cross-model check of the dFBA pipeline.',
        'doc_func': get_dfba_single_doc,
        'plot_func': plot_dfba_single,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {'model_id': 'yeast', 'initial_fields': {'glucose': 5}},
        'plot_config': {'filename': 'yeast_dfba'}
    },

    # ---- Multi-metabolism models ------------------------------------------
    'community_dfba': {
        'description': 'Multi-agent well-mixed community: several independent dFBA instances share the same extracellular pools, creating competition/cross-feeding dynamics without space.',
        'doc_func': get_community_dfba_doc,
        'plot_func': plot_community_dfba,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {},
        'plot_config': {'filename': 'community_dfba'}
    },
    'dfba_kinetics_community': {
        'description': 'Hybrid community (well-mixed): mixes Monod-kinetic agents with dFBA agents in a shared environment. Demonstrates heterogeneous process composition under one schema.',
        'doc_func': get_dfba_kinetics_community_doc,
        'plot_func': plot_dfba_kinetics_community,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {},
        'plot_config': {'filename': 'dfba_kinetics_community'}
    },
    'spatial_many_dfba': {
        'description': 'Spatial microenvironment (sitewise dFBA): a lattice where each site runs its own dFBA instance. Useful for validating lattice indexing + per-site state isolation.',
        'doc_func': get_spatial_many_dfba_doc,
        'plot_func': plot_spatial_many_dfba,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {'model_id': 'ecoli core'},
        'plot_config': {'filename': 'spatial_many_dfba'}
    },
    'spatial_dfba_process': {
        'description': 'Spatial microenvironment (vectorized dFBA): one spatial dFBA process updates all lattice sites as a single structured state. Demonstrates batched execution + field coupling.',
        'doc_func': get_spatial_dfba_process_doc,
        'plot_func': plot_dfba_process_spatial,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {},
        'plot_config': {'filename': 'spatial_dfba_process'}
    },

    # ---- Spatial models ---------------------------------------------------
    'diffusion_process': {
        'description': 'Field transport primitive: finite-volume diffusion/advection on a 2D lattice. Use to validate boundary conditions, stability, and transport timescales.',
        'doc_func': get_diffusion_process_doc,
        'plot_func': plot_diffusion_process,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {},
        'plot_config': {'filename': 'diffusion_process'}
    },

    # ---- Brownian Particle composite models --------------------------------
    'brownian_particles': {
        'description': 'Particle-only baseline: Brownian motion of agents with mass in continuous space (no fields, no metabolism). Checks integrator + particle state schema.',
        'doc_func': get_brownian_particles_alone_doc,
        'plot_func': plot_particles_sim,
        'time': DEFAULT_RUNTIME_LONGER,
        'config': {},
        'plot_config': {'filename': 'brownian_particles'}
    },
    'br_particles_kinetics': {
        'description': 'Particle–field coupling (kinetics): Brownian agents sample local lattice concentrations and apply Monod-style exchange, modifying both particle mass and fields.',
        'doc_func': get_br_particles_kinetics_doc,
        'plot_func': plot_particles_sim,
        'time': DEFAULT_RUNTIME_LONGER,
        'config': {},
        'plot_config': {'filename': 'br_particles_kinetics', 'n_snapshots': 6}
    },
    'br_particles_dfba': {
        'description': 'Particle-embedded metabolism: Brownian agents carry internal dFBA; uptake/secretion couples to fields and biomass accumulates into particle mass/size.',
        'doc_func': get_br_particles_dfba_doc,
        'plot_func': plot_particle_dfba,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {
            'particle_model_id': 'ecoli core'
        },
        'plot_config': {'filename': 'br_particles_dfba', "particles_row": "separate"}
    },

    # ---- COMETS-like composite models --------------------------------------
    'comets_diffusion': {
        'description': 'COMETS-style spatial dFBA: per-site dFBA coupled to diffusion/advection, yielding spatiotemporal nutrient gradients and growth fronts on the lattice.',
        'doc_func': get_comets_doc,
        'plot_func': plot_comets,
        'time': DEFAULT_RUNTIME_LONGER,
        'config': {},
        'plot_config': {'filename': 'comets_diffusion', 'n_snapshots': 5}
    },
    'comets_br_particles_kinetics': {
        'description': 'COMETS + motile kinetic agents: adds Brownian particles with internal kinetics that exchange with COMETS fields. Demonstrates moving agents on a diffusing chemical landscape.',
        'doc_func': get_comets_br_particles_kinetics_doc,
        'plot_func': plot_kinetic_particle_comets,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {},
        'plot_config': {'filename': 'comets_br_particles_kinetics', 'n_snapshots': 5}
    },
    'comets_br_particles_dfba': {
        'description': 'COMETS + motile metabolic agents: Brownian particles carry dFBA and exchange with diffusing fields. Look for gradient-following effects and spatially structured growth.',
        'doc_func': get_comets_br_particles_dfba_doc,
        'plot_func': plot_particle_dfba_comets,
        'time': DEFAULT_RUNTIME_LONG,
        'config': {},
        'plot_config': {'filename': 'comets_br_particles_dfba', 'n_snapshots': 4}
    },

    # ---- Pymunk Newtonian Particle composite models ------------------------
    'newtonian_particles': {
        'description': 'Physics-only baseline (Pymunk): rigid-body particles with collisions/crowding in continuous space. Use to validate contact dynamics + boundary enforcement.',
        'doc_func': get_newtonian_particles_doc,
        'plot_func': plot_newtonian_particles,
        'time': DEFAULT_RUNTIME_LONGER,
        'config': {},
        'plot_config': {'filename': 'newtonian_particles'}
    },
    'comets_nt_particles_dfba': {
        'description': 'Mechanochemical + metabolic coupling: Pymunk particles move/collide while COMETS fields diffuse; particles run metabolism (via exchange) against local concentrations.',
        'doc_func': get_newtonian_particle_comets_doc,
        'plot_func': plot_newtonian_particle_comets,
        'time': DEFAULT_RUNTIME_LONG, #ER,
        'config': {},
        'plot_config': {'filename': 'comets_nt_particles_dfba'}
    },

    # ---- Integrated-Composite Demo  ---------------------------------------------
    'spatioflux_reference_demo': {
        'description': 'SpatioFlux demonstration reference composite: Newtonian motile particles + particle–field exchange + internal multi-dFBA (e.g., glucose vs acetate strategies) + Monod/diffusion fields + mass-aggregated division.',
        'doc_func': get_reference_composite_doc,
        'plot_func': plot_newtonian_particle_comets,
        'time':  40,  #300, #DEFAULT_RUNTIME_LONGER*3,
        'config': {},
        'plot_config': {'filename': 'spatioflux_reference_demo', "particles_row": "separate", "n_snapshots": 8}
    },

    'reference_demo_x2y2': {
        'description': 'Different resolution for the spatio-flux reference demo',
        'doc_func': get_reference_composite_doc,
        'plot_func': plot_newtonian_particle_comets,
        'time': 40,
        'config': {'n_bins': [n*2 for n in SQUARE_BINS]},
        'plot_config': {'filename': 'reference_demo_x2y2', "particles_row": "separate", "n_snapshots": 8}
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

    core = allocate_core()

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
        results = run_composite_document(doc, core=core, name=name, time=runtime,
                                         show_types=True, show_values=True)
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
