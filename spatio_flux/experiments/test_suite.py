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
import gc
import os
import time
import matplotlib.pyplot as plt
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

from spatio_flux.composites._constants import (  # noqa: F401
    SQUARE_BOUNDS, SQUARE_BINS,
    DEFAULT_BOUNDS, DEFAULT_BINS, DEFAULT_BINS_SMALL,
    DEFAULT_ADVECTION, DEFAULT_DIFFUSION,
    DEFAULT_ADD_RATE, DEFAULT_ADD_BOUNDARY, DEFAULT_REMOVE_BOUNDARY,
    DEFAULT_INITIAL_MIN_MAX,
    get_newtonian_particles_process,
    build_model_grid,
)
from spatio_flux.composites import REGISTRY as COMPOSITE_REGISTRY
from pbg_superpowers.composite_generator import build_generator

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

def plot_dfba_kinetics_community(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'dfba_kinetics_community')
    species_ids = ['glucose', 'acetate', 'dfba biomass', 'kinetic biomass']
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

def plot_spatial_many_dfba(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'spatial_many_dfba')
    plot_time_series(results, coordinates=[(0, 0), (1, 1), (1, 3)], out_dir='out', filename=f'{filename}_timeseries.png')
    plot_species_distributions_to_gif(results, out_dir='out', filename=f'{filename}_video.gif')

# --- DFBA Spatial Process ---------------------------------------------

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

def plot_diffusion_process(results, state, config=None):
    config = config or {}
    filename = config.get('filename', 'diffusion_process')
    plot_species_distributions_to_gif(results, out_dir='out', filename=f'{filename}_video.gif')

# --- COMETS -----------------------------------------------------------


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
    depth = user_cfg.get("depth", 1 / 25)

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
                'ATPM': {'lower': 3, 'upper': 3}
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

    # Graph-Rewrite steps
    particle_division = get_particle_divide_process(division_mass_threshold=division_mass_threshold, submass_split_mode='random')
    enforce_boundaries = get_boundaries_process(particle_process_name="newtonian_particles", bounds=bounds, add_rate=boundary_cfg["add_rate"])

    # Adapters
    particle_exchange = get_particle_exchange_process(n_bins=n_bins, bounds=bounds, depth=depth)

    # composite schema
    schema = get_community_dfba_particle_composition(models=models)

    doc = {
        "state": {
            **spatial_kinetics,  # put them at the top level
            "fields": fields,
            "diffusion": diffusion,
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
        'generator':   'monod_kinetics',
        'plot_func':   plot_kinetics_single,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {'model_id': 'overflow_metabolism'},
        'plot_config': {'filename': 'monod_kinetics'},
    },
    'ecoli_core_dfba': {
        'generator':   'ecoli_core_dfba',
        'plot_func':   plot_dfba_single,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {'model_id': 'ecoli core',
                        'glucose': 10.0, 'acetate': 0.0},
        'plot_config': {'filename': 'ecoli_core_dfba'},
    },
    'ecoli_dfba': {
        'generator':   'ecoli_dfba',
        'plot_func':   plot_dfba_single,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {'model_id': 'ecoli', 'glucose': 10.0, 'formate': 5.0},
        'plot_config': {'filename': 'ecoli_dfba'},
    },
    'yeast_dfba': {
        'generator':   'yeast_dfba',
        'plot_func':   plot_dfba_single,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {'model_id': 'yeast', 'glucose': 5.0},
        'plot_config': {'filename': 'yeast_dfba'},
    },

    # ---- Multi-metabolism models ------------------------------------------
    'community_dfba': {
        'generator':   'community_dfba',
        'plot_func':   plot_community_dfba,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {},
        'plot_config': {'filename': 'community_dfba'},
    },
    'dfba_kinetics_community': {
        'generator':   'dfba_kinetics_community',
        'plot_func':   plot_dfba_kinetics_community,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {},
        'plot_config': {'filename': 'dfba_kinetics_community'},
    },
    'spatial_many_dfba': {
        'generator':   'spatial_many_dfba',
        'plot_func':   plot_spatial_many_dfba,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {'model_id': 'ecoli core'},
        'plot_config': {'filename': 'spatial_many_dfba'},
    },
    'spatial_dfba_process': {
        'generator':   'spatial_dfba_process',
        'plot_func':   plot_dfba_process_spatial,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {},
        'plot_config': {'filename': 'spatial_dfba_process'},
    },

    # ---- Spatial models ---------------------------------------------------
    'diffusion_process': {
        'generator':   'diffusion_process',
        'plot_func':   plot_diffusion_process,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {},
        'plot_config': {'filename': 'diffusion_process'},
    },

    # ---- Brownian Particle composite models --------------------------------
    'brownian_particles': {
        'generator':   'brownian_particles',
        'plot_func':   plot_particles_sim,
        'time':        DEFAULT_RUNTIME_LONGER,
        'overrides':   {},
        'plot_config': {'filename': 'brownian_particles'},
    },
    'br_particles_kinetics': {
        'generator':   'br_particles_kinetics',
        'plot_func':   plot_particles_sim,
        'time':        DEFAULT_RUNTIME_LONGER,
        'overrides':   {},
        'plot_config': {'filename': 'br_particles_kinetics', 'n_snapshots': 6},
    },
    'br_particles_dfba': {
        'generator':   'br_particles_dfba',
        'plot_func':   plot_particles_sim,
        'time':        DEFAULT_RUNTIME_LONGER,
        'overrides':   {'particle_model_id': 'ecoli core'},
        'plot_config': {'filename': 'br_particles_dfba', 'particles_row': 'separate'},
    },

    # ---- COMETS-like composite models --------------------------------------
    'comets_diffusion': {
        'generator':   'comets_diffusion',
        'plot_func':   plot_comets,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {},
        'plot_config': {'filename': 'comets_diffusion'},
    },
    'comets_br_particles_kinetics': {
        'generator':   'comets_br_particles_kinetics',
        'plot_func':   plot_kinetic_particle_comets,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {},
        'plot_config': {'filename': 'comets_br_particles_kinetics', 'n_snapshots': 5},
    },
    'comets_br_particles_dfba': {
        'generator':   'comets_br_particles_dfba',
        'plot_func':   plot_particle_dfba_comets,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {},
        'plot_config': {'filename': 'comets_br_particles_dfba', 'n_snapshots': 4},
    },

    # ---- Pymunk Newtonian Particle composite models ------------------------
    'newtonian_particles': {
        'generator':   'newtonian_particles',
        'plot_func':   plot_newtonian_particles,
        'time':        DEFAULT_RUNTIME_LONGER,
        'overrides':   {},
        'plot_config': {'filename': 'newtonian_particles'},
    },
    'comets_nt_particles_dfba': {
        'generator':   'comets_nt_particles_dfba',
        'plot_func':   plot_newtonian_particle_comets,
        'time':        DEFAULT_RUNTIME_LONG,
        'overrides':   {},
        'plot_config': {'filename': 'comets_nt_particles_dfba'},
    },

    # ---- Integrated-Composite Demo  ---------------------------------------------
    'spatioflux_reference_demo': {
        'generator':   'spatioflux_reference_demo',
        'plot_func':   plot_newtonian_particle_comets,
        'time':        120,
        'overrides':   {'n_bins': list(SQUARE_BINS)},
        'plot_config': {'filename': 'spatioflux_reference_demo',
                        'particles_row': 'separate', 'n_snapshots': 8},
    },

    'reference_demo_x2y2': {
        'description': 'Different resolution for the spatio-flux reference demo',
        'doc_func': get_reference_composite_doc,
        'plot_func': plot_newtonian_particle_comets,
        'time': 120,
        'config': {
            'n_bins': [n * 2 for n in SQUARE_BINS]
        },
        'plot_config': {'filename': 'reference_demo_x2y2', "particles_row": "separate", "n_snapshots": 8},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run selected simulations.')
    parser.add_argument(
        '--tests', nargs='*', default=None,
        help='Names of tests to run. If none given, runs the full set.'
    )
    parser.add_argument('--output', default='out', help='Output directory')
    parser.add_argument(
        '--skip-existing', action='store_true',
        help=('Reuse cached per-test artifacts: do not wipe the output '
              'directory at start, and skip any test whose '
              '<filename>_viz.png already exists. Lets you add or '
              're-run a single test without losing the rest of the '
              'report.'))
    return parser.parse_args()


def _existing_outputs_for(sim_name, output_dir):
    """True if a previous run's artifacts for this test are present.

    A test is considered cached if its composite viz PNG exists. The
    viz is produced unconditionally by ``run_composite_document`` and
    is the cheapest single signal that a test completed before.
    """
    plot_config = SIMULATIONS[sim_name].get('plot_config', {}) or {}
    base = plot_config.get('filename', sim_name)
    # The harness writes to './out/' regardless of --output; check both
    # so this stays correct if the harness is later parameterised.
    candidates = [
        os.path.join(output_dir, f'{base}_viz.png'),
        os.path.join('out', f'{base}_viz.png'),
    ]
    return any(os.path.exists(c) for c in candidates)


def main():
    args = parse_args()

    output_dir = args.output

    if not args.skip_existing:
        prepare_output_dir(output_dir)
    else:
        # Don't wipe — preserve whatever cached artifacts are there
        # so the report can rebuild from disk.
        os.makedirs(output_dir, exist_ok=True)

    core = allocate_core()

    total_sim_time = 0.0  # To track simulation time only
    runtimes = {}
    timing_details = {}  # {name: (process_time, framework_time)}

    test_names = list(SIMULATIONS.keys())
    tests_to_run = args.tests if args.tests else test_names
    print(f"\nSelected tests to run: {', '.join(tests_to_run)}\n")

    for name in tests_to_run:
        print(f"\n🚀 Running test: {name}")
        if name not in SIMULATIONS:
            print(f"Skipping unknown test: '{name}'")
            continue

        if args.skip_existing and _existing_outputs_for(name, output_dir):
            print(f"⏭️  Skipping '{name}' (cached outputs already exist)")
            continue

        sim_info = SIMULATIONS[name]

        print("Creating document...")
        if 'generator' in sim_info:
            entry = next(e for e in COMPOSITE_REGISTRY.values()
                         if e.name == sim_info['generator'])
            doc = build_generator(entry, overrides=sim_info.get('overrides', {}), core=core)
        else:
            config = sim_info.get('config', {})
            doc = sim_info['doc_func'](core=core, config=config)

        print("Sending document...")
        runtime = sim_info.get('time', DEFAULT_RUNTIME_LONG)
        # Per-test override for emit cadence; default auto-computes
        # in run_composite_document to target ~60 emits.
        emit_subsample = sim_info.get('emit_subsample')
        results, proc_time, fw_time = run_composite_document(
            doc, core=core, name=name, time=runtime,
            show_types=True, show_values=True,
            emit_subsample=emit_subsample)

        sim_elapsed = proc_time + fw_time
        runtimes[name] = sim_elapsed
        timing_details[name] = (proc_time, fw_time)
        total_sim_time += sim_elapsed

        print("Generating plots...")
        plot_config = sim_info.get('plot_config', {})
        sim_info['plot_func'](results, doc.get('state', doc), config=plot_config)

        print(f"✅ Completed: {name} in {sim_elapsed:.2f}s "
              f"(process: {proc_time:.2f}s, framework: {fw_time:.2f}s)")

        plt.close('all')
        del results, doc
        gc.collect()

    print(f"\nCompiling HTML report...")
    generate_html_report(
        output_dir,
        {k: v.get('config', v.get('overrides', {})) for k, v in SIMULATIONS.items()},
        {k: v.get('description', '') for k, v in SIMULATIONS.items()},
        runtimes,
        total_sim_time,
        timing_details=timing_details,
    )


if __name__ == '__main__':
    main()
