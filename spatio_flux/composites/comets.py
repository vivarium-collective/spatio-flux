"""Composite generators — comets group. See spec at
docs/superpowers/specs/2026-05-12-composite-generator-convention.md."""
import numpy as np
from pbg_superpowers.composite_generator import composite_generator

from spatio_flux.processes import (
    MODEL_REGISTRY_DFBA, DIVISION_MASS_THRESHOLD,
    get_fields, get_fields_with_schema, get_particles_state,
    get_newtonian_particles_state, get_brownian_movement_process,
    get_boundaries_process, get_particle_exchange_process,
    get_particle_divide_process, get_diffusion_advection_process,
    get_spatial_many_dfba, get_spatial_dFBA_process,
    get_spatial_many_kinetics, get_kinetic_particle_composition,
    get_dfba_particle_composition,
)
from spatio_flux.processes.monod_kinetics import MODEL_REGISTRY_KINETICS
from spatio_flux.composites._constants import (
    SQUARE_BINS, SQUARE_BOUNDS, DEFAULT_DIFFUSION, DEFAULT_ADVECTION,
    DEFAULT_INITIAL_MIN_MAX, get_newtonian_particles_process,
)


@composite_generator(
    name="comets_diffusion",
    description=(
        "Two-phase coupling (no particles): spatial dFBA on a lattice with "
        "advection–diffusion of substrates + biomass. The classic COMETS "
        "field-only scenario."
    ),
    parameters={
        "dissolved_model_id": {"type": "string", "default": "ecoli core"},
        "n_bins":             {"type": "object", "default": list(SQUARE_BINS)},
        "bounds":             {"type": "object", "default": list(SQUARE_BOUNDS)},
        "advection_rate":     {"type": "object", "default": list(DEFAULT_ADVECTION)},
    },
)
def comets_diffusion(core=None, *, dissolved_model_id="ecoli core",
                     n_bins=list(SQUARE_BINS),
                     bounds=list(SQUARE_BOUNDS),
                     advection_rate=list(DEFAULT_ADVECTION)):
    mol_ids = ["glucose", "acetate", "dissolved biomass"]
    n_bins_t, bounds_t = tuple(n_bins), tuple(bounds)
    advection_rate_t = tuple(advection_rate)
    diffusion_coeffs = {"glucose": 0.0, "acetate": 1e-1,
                        "dissolved biomass": 1e-2}
    advection_coeffs = {"dissolved biomass": advection_rate_t}
    nx, ny = n_bins_t
    shape = (ny, nx)
    acetate_field = np.zeros(shape, dtype=float)
    glc_y = np.linspace(0.01, 10.0, ny, dtype=float)[:, None]
    glc_field = np.repeat(glc_y, nx, axis=1)
    biomass_field = np.zeros(shape, dtype=float)
    x0 = nx // 2
    biomass_field[0:1, x0 - 1:x0 + 1] = 0.1
    initial_fields = {
        "dissolved biomass": biomass_field,
        "glucose": glc_field, "acetate": acetate_field,
    }
    spatial_dfba = get_spatial_many_dfba(
        model_id=dissolved_model_id, mol_ids=mol_ids,
        n_bins=n_bins_t, path=["fields"])
    return {
        **spatial_dfba,
        "fields": get_fields_with_schema(
            n_bins=n_bins_t, mol_ids=mol_ids, initial_fields=initial_fields),
        "diffusion": get_diffusion_advection_process(
            bounds=bounds_t, n_bins=n_bins_t, mol_ids=mol_ids,
            advection_coeffs=advection_coeffs,
            diffusion_coeffs=diffusion_coeffs),
    }


@composite_generator(
    name="comets_br_particles_kinetics",
    description=(
        "Three-way coupling: spatial dFBA fields + Brownian particle "
        "agents that run Monod kinetics locally + diffusion. End-to-end "
        "smoke test of the COMETS-style composite stack with kinetic agents."
    ),
    parameters={
        "division_mass_threshold": {"type": "float",  "default": DIVISION_MASS_THRESHOLD},
        "dissolved_model_id":      {"type": "string", "default": "ecoli core"},
        "kinetic_model_id":        {"type": "string", "default": "overflow_metabolism"},
        "n_particles":             {"type": "int",    "default": 1},
        "add_rate":                {"type": "float",  "default": 0.1},
        "n_bins":                  {"type": "object", "default": list(SQUARE_BINS)},
        "bounds":                  {"type": "object", "default": list(SQUARE_BOUNDS)},
        "diffusion_rate":          {"type": "float",  "default": DEFAULT_DIFFUSION},
        "advection_rate":          {"type": "object", "default": list(DEFAULT_ADVECTION)},
        "initial_min_max":         {"type": "object", "default": dict(DEFAULT_INITIAL_MIN_MAX)},
    },
)
def comets_br_particles_kinetics(
        core=None, *, division_mass_threshold=DIVISION_MASS_THRESHOLD,
        dissolved_model_id="ecoli core",
        kinetic_model_id="overflow_metabolism",
        n_particles=1, add_rate=0.1,
        n_bins=list(SQUARE_BINS),
        bounds=list(SQUARE_BOUNDS),
        diffusion_rate=DEFAULT_DIFFUSION,
        advection_rate=list(DEFAULT_ADVECTION),
        initial_min_max=dict(DEFAULT_INITIAL_MIN_MAX)):
    mol_ids = ["glucose", "acetate", "dissolved biomass"]
    particle_config = MODEL_REGISTRY_KINETICS[kinetic_model_id]()
    n_bins_t, bounds_t = tuple(n_bins), tuple(bounds)
    advection_rate_t = tuple(advection_rate)
    fields = get_fields(
        n_bins=n_bins_t, mol_ids=mol_ids,
        initial_min_max=initial_min_max)
    n_grid = (n_bins_t[1], n_bins_t[0])
    fields["dissolved biomass"] = np.zeros(n_grid)
    fields["dissolved biomass"][0, int(n_grid[0] / 4):int(3 * n_grid[0] / 4)] = 0.1
    spatial_dFBA_config = {
        "n_bins": n_bins_t, "models": MODEL_REGISTRY_DFBA, "mol_ids": mol_ids,
    }
    return {
        "state": {
            "fields": fields,
            "particles": get_particles_state(
                n_particles=n_particles, bounds=bounds_t,
                mass_range=(1e0, 1e1)),
            "spatial_dFBA": get_spatial_dFBA_process(
                config=spatial_dFBA_config, model_id=dissolved_model_id),
            "diffusion": get_diffusion_advection_process(
                bounds=bounds_t, n_bins=n_bins_t, mol_ids=mol_ids),
            "brownian_movement": get_brownian_movement_process(
                bounds=bounds_t, advection_rate=advection_rate_t,
                diffusion_rate=diffusion_rate),
            "enforce_boundaries": get_boundaries_process(
                particle_process_name="brownian_movement",
                bounds=bounds_t, add_rate=add_rate),
            "particle_exchange": get_particle_exchange_process(
                n_bins=n_bins_t, bounds=bounds_t),
            "particle_division": get_particle_divide_process(
                division_mass_threshold=division_mass_threshold),
        },
        "schema": get_kinetic_particle_composition(core, config=particle_config),
    }


@composite_generator(
    name="comets_br_particles_dfba",
    description=(
        "Full COMETS-style spatial composite: lattice dFBA + advection–"
        "diffusion + Brownian particles with internal dFBA + division. "
        "End-to-end coupled scenario."
    ),
    parameters={
        "particle_model_id":       {"type": "string", "default": "ecoli core"},
        "dissolved_model_id":      {"type": "string", "default": "ecoli core"},
        "division_mass_threshold": {"type": "float",  "default": 3},
        "n_particles":             {"type": "int",    "default": 1},
        "add_rate":                {"type": "float",  "default": 0.3},
        "n_bins":                  {"type": "object", "default": list(SQUARE_BINS)},
        "bounds":                  {"type": "object", "default": list(SQUARE_BOUNDS)},
        "diffusion_rate":          {"type": "float",  "default": DEFAULT_DIFFUSION},
        "field_advection_rate":    {"type": "object", "default": list(DEFAULT_ADVECTION)},
    },
)
def comets_br_particles_dfba(
        core=None, *, particle_model_id="ecoli core",
        dissolved_model_id="ecoli core", division_mass_threshold=3,
        n_particles=1, add_rate=0.3,
        n_bins=list(SQUARE_BINS),
        bounds=list(SQUARE_BOUNDS),
        diffusion_rate=DEFAULT_DIFFUSION,
        field_advection_rate=list(DEFAULT_ADVECTION)):
    mol_ids = ["glucose", "acetate", "dissolved biomass"]
    n_bins_t, bounds_t = tuple(n_bins), tuple(bounds)
    field_advection_rate_t = tuple(field_advection_rate)
    nx, ny = n_bins_t
    shape = (ny, nx)
    acetate_field = np.zeros(shape, dtype=float)
    glc_y = np.linspace(0.01, 10.0, ny, dtype=float)[:, None]
    glc_field = np.repeat(glc_y, nx, axis=1)
    biomass_field = np.zeros(shape, dtype=float)
    x0, x1 = nx // 4, 3 * nx // 4
    biomass_field[0:1, x0:x1] = 0.1
    initial_fields = {"dissolved biomass": biomass_field,
                      "glucose": glc_field, "acetate": acetate_field}
    advection_coeffs = {"dissolved biomass": field_advection_rate_t}
    spatial_dfba = get_spatial_many_dfba(
        n_bins=n_bins_t, model_id=dissolved_model_id,
        mol_ids=mol_ids, path=["fields"])
    state = {
        **spatial_dfba,
        "fields": get_fields_with_schema(
            n_bins=n_bins_t, mol_ids=mol_ids, initial_fields=initial_fields),
        "diffusion": get_diffusion_advection_process(
            bounds=bounds_t, n_bins=n_bins_t, mol_ids=mol_ids,
            advection_coeffs=advection_coeffs),
        "particles": get_particles_state(
            n_particles=n_particles, bounds=bounds_t),
        "brownian_movement": get_brownian_movement_process(
            bounds=bounds_t, advection_rate=(0, -0.2),
            diffusion_rate=diffusion_rate),
        "enforce_boundaries": get_boundaries_process(
            particle_process_name="brownian_movement",
            bounds=bounds_t, add_rate=add_rate),
        "particle_exchange": get_particle_exchange_process(
            n_bins=n_bins_t, bounds=bounds_t),
        "particle_division": get_particle_divide_process(
            division_mass_threshold=division_mass_threshold),
    }
    return {
        "state": state,
        "schema": get_dfba_particle_composition(model_file=particle_model_id),
    }


@composite_generator(
    name="comets_nt_particles_dfba",
    description=(
        "Mechanochemical + metabolic coupling: Pymunk particles move/"
        "collide while COMETS fields diffuse; particles run metabolism "
        "(via exchange) against local concentrations."
    ),
    parameters={
        "particle_model_id":  {"type": "string", "default": "ecoli core"},
        "dissolved_model_id": {"type": "string", "default": "ecoli core"},
        "n_particles":        {"type": "int",    "default": 2},
        "bounds":             {"type": "object", "default": list(SQUARE_BOUNDS)},
        "n_bins":             {"type": "object",
                               "default": [n * 2 for n in SQUARE_BINS]},
        "advection_rate":     {"type": "object", "default": list(DEFAULT_ADVECTION)},
    },
)
def comets_nt_particles_dfba(
        core=None, *, particle_model_id="ecoli core",
        dissolved_model_id="ecoli core", n_particles=2,
        bounds=list(SQUARE_BOUNDS),
        n_bins=[n * 2 for n in SQUARE_BINS],
        advection_rate=list(DEFAULT_ADVECTION)):
    bounds_t, n_bins_t = tuple(bounds), tuple(n_bins)
    advection_rate_t = tuple(advection_rate)
    mol_ids = ["glucose", "acetate", "dissolved biomass"]
    initial_min_max = {"glucose": (0.5, 2), "acetate": (0, 0),
                       "dissolved biomass": (0, 0.1)}
    advection_coeffs = {"dissolved biomass": advection_rate_t}
    fields = get_fields(n_bins=n_bins_t, mol_ids=mol_ids,
                        initial_min_max=initial_min_max)
    particle_config = {
        "gravity": -1.0, "elasticity": 0.1, "bounds": bounds_t,
        "jitter_per_second": 1e-1, "damping_per_second": 1e-1,
    }
    return {
        "state": {
            "fields": fields,
            "diffusion": get_diffusion_advection_process(
                bounds=bounds_t, n_bins=n_bins_t, mol_ids=mol_ids,
                advection_coeffs=advection_coeffs),
            "spatial_kinetics": get_spatial_many_kinetics(
                model_id="single_substrate_assimilation",
                n_bins=n_bins_t, mol_ids=mol_ids),
            "particles": get_newtonian_particles_state(
                n_particles=n_particles, bounds=bounds_t),
            "newtonian_particles": get_newtonian_particles_process(
                config=particle_config),
            "particle_exchange": get_particle_exchange_process(
                n_bins=n_bins_t, bounds=bounds_t),
            "particle_division": get_particle_divide_process(
                division_mass_threshold=0.5),
            "enforce_boundaries": get_boundaries_process(
                particle_process_name="newtonian_particles",
                bounds=bounds_t, boundary_to_add=("top",),
                add_rate=0.01, mass_range=(1e-3, 1e-2)),
        },
        "schema": get_dfba_particle_composition(model_file=particle_model_id),
    }
