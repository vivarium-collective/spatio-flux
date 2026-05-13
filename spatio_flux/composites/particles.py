"""Composite generators — particles group. See spec at
docs/superpowers/specs/2026-05-12-composite-generator-convention.md."""
import numpy as np
from pbg_superpowers.composite_generator import composite_generator

from spatio_flux.processes import (
    get_fields, get_particles_state, get_newtonian_particles_state,
    get_brownian_movement_process, get_boundaries_process,
    get_particle_exchange_process, get_particle_divide_process,
    get_kinetic_particle_composition, get_dfba_particle_composition,
    DIVISION_MASS_THRESHOLD,
)
from spatio_flux.processes.monod_kinetics import MODEL_REGISTRY_KINETICS
from spatio_flux.composites._constants import (
    SQUARE_BINS, SQUARE_BOUNDS, DEFAULT_DIFFUSION, DEFAULT_ADVECTION,
    get_newtonian_particles_process,
)


@composite_generator(
    name="brownian_particles",
    description=(
        "Particle-only baseline: Brownian motion of agents with mass in "
        "continuous space (no fields, no metabolism). Checks integrator + "
        "particle state schema."
    ),
    parameters={
        "n_bins":         {"type": "object", "default": list(SQUARE_BINS)},
        "bounds":         {"type": "object", "default": list(SQUARE_BOUNDS)},
        "n_particles":    {"type": "int",    "default": 1},
        "time_interval":  {"type": "float",  "default": 0.1},
        "diffusion_rate": {"type": "float",  "default": DEFAULT_DIFFUSION},
        "add_rate":       {"type": "float",  "default": 0.01},
    },
)
def brownian_particles(core=None, *, n_bins=list(SQUARE_BINS),
                       bounds=list(SQUARE_BOUNDS), n_particles=1,
                       time_interval=0.1, diffusion_rate=DEFAULT_DIFFUSION,
                       add_rate=0.01):
    bounds_t = tuple(bounds)
    return {
        "state": {
            "particles": get_particles_state(
                n_particles=n_particles, bounds=bounds_t),
            "brownian_movement": get_brownian_movement_process(
                bounds=bounds_t, diffusion_rate=diffusion_rate,
                interval=time_interval),
            "enforce_boundaries": get_boundaries_process(
                particle_process_name="brownian_movement",
                bounds=bounds_t, add_rate=add_rate),
        },
    }


@composite_generator(
    name="br_particles_kinetics",
    description=(
        "Particle–field coupling (kinetics): Brownian agents sample local "
        "lattice concentrations and apply Monod-style exchange, modifying "
        "both particle mass and fields."
    ),
    parameters={
        "model_id":                 {"type": "string", "default": "overflow_metabolism"},
        "division_mass_threshold":  {"type": "float",  "default": DIVISION_MASS_THRESHOLD},
        "n_bins":                   {"type": "object", "default": list(SQUARE_BINS)},
        "bounds":                   {"type": "object", "default": list(SQUARE_BOUNDS)},
        "n_particles":              {"type": "int",    "default": 1},
        "diffusion_rate":           {"type": "float",  "default": DEFAULT_DIFFUSION / 2},
    },
)
def br_particles_kinetics(core=None, *, model_id="overflow_metabolism",
                          division_mass_threshold=DIVISION_MASS_THRESHOLD,
                          n_bins=list(SQUARE_BINS),
                          bounds=list(SQUARE_BOUNDS),
                          n_particles=1,
                          diffusion_rate=DEFAULT_DIFFUSION / 2):
    initial_min_max = {"glucose": (5.0, 10.0), "acetate": (0, 0)}
    particle_config = MODEL_REGISTRY_KINETICS[model_id]()
    mol_ids = list(initial_min_max.keys())
    n_bins_t, bounds_t = tuple(n_bins), tuple(bounds)
    return {
        "state": {
            "fields": get_fields(
                n_bins=n_bins_t, mol_ids=mol_ids,
                initial_min_max=initial_min_max),
            "particles": get_particles_state(
                n_particles=n_particles, bounds=bounds_t),
            "brownian_movement": get_brownian_movement_process(
                bounds=bounds_t, diffusion_rate=diffusion_rate,
                advection_rate=(0, 0)),
            "enforce_boundaries": get_boundaries_process(
                particle_process_name="brownian_movement",
                bounds=bounds_t, add_rate=0.0),
            "particle_exchange": get_particle_exchange_process(
                n_bins=n_bins_t, bounds=bounds_t),
            "particle_division": get_particle_divide_process(
                division_mass_threshold=division_mass_threshold),
        },
        "schema": get_kinetic_particle_composition(core=core,
                                                   config=particle_config),
    }


@composite_generator(
    name="br_particles_dfba",
    description=(
        "Particle-embedded metabolism: Brownian agents carry internal dFBA; "
        "uptake/secretion couples to fields and biomass accumulates into "
        "particle mass/size."
    ),
    parameters={
        "particle_model_id":        {"type": "string", "default": "ecoli core"},
        "division_mass_threshold":  {"type": "float",  "default": DIVISION_MASS_THRESHOLD},
        "add_rate":                 {"type": "float",  "default": 0.1},
        "n_bins":                   {"type": "object", "default": list(SQUARE_BINS)},
        "bounds":                   {"type": "object", "default": list(SQUARE_BOUNDS)},
        "n_particles":              {"type": "int",    "default": 1},
        "diffusion_rate":           {"type": "float",  "default": DEFAULT_DIFFUSION},
        "advection_rate":           {"type": "object", "default": list(DEFAULT_ADVECTION)},
    },
)
def br_particles_dfba(core=None, *, particle_model_id="ecoli core",
                      division_mass_threshold=DIVISION_MASS_THRESHOLD,
                      add_rate=0.1,
                      n_bins=list(SQUARE_BINS),
                      bounds=list(SQUARE_BOUNDS),
                      n_particles=1,
                      diffusion_rate=DEFAULT_DIFFUSION,
                      advection_rate=list(DEFAULT_ADVECTION)):
    mol_ids = ["glucose", "acetate"]
    n_bins_t, bounds_t = tuple(n_bins), tuple(bounds)
    advection_rate_t = tuple(advection_rate)
    nx, ny = n_bins_t
    acetate_field = np.zeros((ny, nx), dtype=float)
    glc_y = np.linspace(0.01, 10.0, ny, dtype=float)[:, None]
    glc_field = np.repeat(glc_y, nx, axis=1)
    initial_fields = {"glucose": glc_field, "acetate": acetate_field}
    return {
        "state": {
            "fields": get_fields(n_bins=n_bins_t, mol_ids=mol_ids,
                                 initial_fields=initial_fields),
            "particles": get_particles_state(
                n_particles=n_particles, bounds=bounds_t),
            "brownian_movement": get_brownian_movement_process(
                bounds=bounds_t, diffusion_rate=diffusion_rate,
                advection_rate=advection_rate_t),
            "enforce_boundaries": get_boundaries_process(
                particle_process_name="brownian_movement",
                bounds=bounds_t, add_rate=add_rate),
            "particle_exchange": get_particle_exchange_process(
                n_bins=n_bins_t, bounds=bounds_t),
            "particle_division": get_particle_divide_process(
                division_mass_threshold=division_mass_threshold),
        },
        "schema": get_dfba_particle_composition(model_file=particle_model_id),
    }


@composite_generator(
    name="newtonian_particles",
    description=(
        "Physics-only baseline (Pymunk): rigid-body particles with "
        "collisions/crowding in continuous space. Use to validate contact "
        "dynamics + boundary enforcement."
    ),
    parameters={
        "n_particles": {"type": "int",    "default": 1},
        "gravity":     {"type": "float",  "default": -0.2},
        "elasticity":  {"type": "float",  "default": 0.1},
        "add_rate":    {"type": "float",  "default": 0.02},
        "bounds":      {"type": "object", "default": list(SQUARE_BOUNDS)},
    },
)
def newtonian_particles(core=None, *, n_particles=1, gravity=-0.2,
                        elasticity=0.1, add_rate=0.02,
                        bounds=list(SQUARE_BOUNDS)):
    bounds_t = tuple(bounds)
    physics = {
        "gravity": gravity, "elasticity": elasticity, "bounds": bounds_t,
        "jitter_per_second": 0.5, "damping_per_second": 0.998,
    }
    return {
        "state": {
            "particles": get_newtonian_particles_state(
                n_particles=n_particles, bounds=bounds_t),
            "newtonian_particles": get_newtonian_particles_process(
                config=physics),
            "enforce_boundaries": get_boundaries_process(
                particle_process_name="newtonian_particles",
                bounds=bounds_t, add_rate=add_rate),
        },
    }
