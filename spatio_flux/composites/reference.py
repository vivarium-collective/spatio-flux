"""Composite generators — reference group. See spec at
docs/superpowers/specs/2026-05-12-composite-generator-convention.md."""
from pbg_superpowers.composite_generator import composite_generator

from spatio_flux.processes import (
    get_fields, get_newtonian_particles_state,
    get_diffusion_advection_process, get_spatial_many_kinetics,
    get_particle_exchange_process, get_particle_divide_process,
    get_boundaries_process, get_community_dfba_particle_composition,
)
from spatio_flux.composites._constants import (
    SQUARE_BINS, SQUARE_BOUNDS, get_newtonian_particles_process,
)


@composite_generator(
    name="spatioflux_reference_demo",
    description=(
        "The flagship SpatioFlux reference model: Newtonian particles, each "
        "carrying a two-strain E. coli community with its own internal dFBA, "
        "swim through diffusing glucose and acetate fields, exchange substrate "
        "locally, and divide by splitting sub-masses between daughters."
    ),
    parameters={
        "bounds":      {"type": "object", "default": list(SQUARE_BOUNDS)},
        "n_bins":      {"type": "object", "default": list(SQUARE_BINS)},
        "depth":       {"type": "float",  "default": 1 / 25},
        "n_particles": {"type": "int",    "default": 1},
    },
)
def spatioflux_reference_demo(
        core=None, *, bounds=list(SQUARE_BOUNDS),
        n_bins=list(SQUARE_BINS), depth=1 / 25, n_particles=1):
    bounds_t, n_bins_t = tuple(bounds), tuple(n_bins)
    division_mass_threshold = 0.4
    add_rate = 0.0
    initial_submasses = {"ecoli_1": 0.1, "ecoli_2": 0.1}
    glucose_level = 5.0
    biomass_id = "dissolved biomass"
    mol_ids = ["glucose", "acetate", biomass_id]
    initial_min_max = {
        "glucose": (glucose_level, glucose_level),
        "acetate": (0.0, 0.0),
        biomass_id: (0.1, 0.2),
    }
    diffusion_coeffs = {"glucose": 1e-1, "acetate": 1e-1, biomass_id: 1e-1}
    diffusion_boundary_config = {
        "default": {"x": {"type": "periodic"},
                    "y": {"type": "neumann"}},
        "glucose": {"top": {"type": "dirichlet", "value": glucose_level}},
        "acetate": {"bottom": {"type": "dirichlet", "value": glucose_level}},
    }
    physics_cfg = {"gravity": -1.0, "elasticity": 0.1, "bounds": bounds_t,
                   "jitter_per_second": 1e-2, "damping_per_second": 0.95,
                   "friction": 0.9}
    models = {
        "ecoli_1": {
            "model_file": "textbook",
            "substrate_update_reactions": {
                "glucose": "EX_glc__D_e", "acetate": "EX_ac_e"},
            "kinetic_params": {"glucose": (0.1, 2), "acetate": (1.0, 0.1)},
            "bounds": {
                "EX_o2_e": {"lower": -2, "upper": None},
                "ATPM": {"lower": 3, "upper": 3},
            },
        },
        "ecoli_2": {
            "model_file": "textbook",
            "substrate_update_reactions": {
                "glucose": "EX_glc__D_e", "acetate": "EX_ac_e"},
            "kinetic_params": {"glucose": (1.0, 0.1), "acetate": (0.01, 1)},
            "bounds": {
                "EX_o2_e": {"lower": -2, "upper": None},
                "ATPM": {"lower": 1, "upper": 1},
            },
        },
    }
    fields = get_fields(n_bins=n_bins_t, mol_ids=mol_ids,
                        initial_min_max=initial_min_max)
    particles = get_newtonian_particles_state(
        n_particles=n_particles, bounds=bounds_t)
    for pid, internal in particles.items():
        internal["sub_masses"] = initial_submasses.copy()
    diffusion = get_diffusion_advection_process(
        bounds=bounds_t, n_bins=n_bins_t, mol_ids=mol_ids,
        diffusion_coeffs=diffusion_coeffs, advection_coeffs={},
        boundary_conditions=diffusion_boundary_config)
    spatial_kinetics = get_spatial_many_kinetics(
        model_id="low_yield_glucose_overflow", biomass_id=biomass_id,
        n_bins=n_bins_t, mol_ids=mol_ids, path=["fields"])
    newtonian = get_newtonian_particles_process(config=physics_cfg)
    particle_division = get_particle_divide_process(
        division_mass_threshold=division_mass_threshold,
        submass_split_mode="random")
    enforce_boundaries = get_boundaries_process(
        particle_process_name="newtonian_particles",
        bounds=bounds_t, add_rate=add_rate)
    particle_exchange = get_particle_exchange_process(
        n_bins=n_bins_t, bounds=bounds_t, depth=depth)
    schema = get_community_dfba_particle_composition(models=models)
    return {
        "state": {
            **spatial_kinetics,
            "fields": fields,
            "diffusion": diffusion,
            "particles": particles,
            "particle_exchange": particle_exchange,
            "particle_division": particle_division,
            "enforce_boundaries": enforce_boundaries,
            "newtonian_particles": newtonian,
        },
        "schema": schema,
    }


# reference_demo_x2y2 is the same generator with doubled grid resolution —
# expose it as a SECOND entry to keep the SIMULATIONS catalog identical to
# what existed before. The body lives in spatioflux_reference_demo; this
# wrapper just declares the alternate default.

@composite_generator(
    name="reference_demo_x2y2",
    description=(
        "The SpatioFlux reference model on a finer field lattice with doubled "
        "bin counts in x and y: the same Newtonian particles carrying a "
        "two-strain internal-dFBA community, with local substrate exchange, "
        "diffusing glucose/acetate fields, and sub-mass-splitting division."
    ),
    parameters={
        "bounds":      {"type": "object", "default": list(SQUARE_BOUNDS)},
        "n_bins":      {"type": "object",
                        "default": [n * 2 for n in SQUARE_BINS]},
        "depth":       {"type": "float",  "default": 1 / 25},
        "n_particles": {"type": "int",    "default": 1},
    },
)
def reference_demo_x2y2(core=None, **kwargs):
    return spatioflux_reference_demo(core=core, **kwargs)
