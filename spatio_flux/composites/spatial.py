"""Composite generators — spatial group. See spec at
docs/superpowers/specs/2026-05-12-composite-generator-convention.md."""
import numpy as np
from pbg_superpowers.composite_generator import composite_generator

from spatio_flux.processes import (
    MODEL_REGISTRY_DFBA, get_fields, get_fields_with_schema,
    get_spatial_many_dfba, get_spatial_dFBA_process,
    get_diffusion_advection_process,
)
from spatio_flux.composites._constants import (
    DEFAULT_BINS, DEFAULT_BOUNDS, DEFAULT_BINS_SMALL,
    DEFAULT_ADVECTION, DEFAULT_DIFFUSION,
    build_model_grid,
)


@composite_generator(
    name="spatial_many_dfba",
    description=(
        "Spatial microenvironment (sitewise dFBA): a lattice where each site "
        "runs its own dFBA instance. Useful for validating lattice indexing "
        "+ per-site state isolation."
    ),
    parameters={
        "model_id": {"type": "string", "default": "ecoli core"},
        "n_bins":   {"type": "object", "default": list(DEFAULT_BINS_SMALL)},
    },
)
def spatial_many_dfba(core=None, *, model_id="ecoli core",
                      n_bins=list(DEFAULT_BINS_SMALL)):
    n_bins_t = tuple(n_bins)
    mol_ids = ["glucose", "acetate", "dissolved biomass"]
    initial_min_max = {"glucose": (0, 20), "acetate": (0, 0),
                       "dissolved biomass": (0, 0.1)}
    return {
        "fields": get_fields_with_schema(
            n_bins=n_bins_t, mol_ids=mol_ids, initial_min_max=initial_min_max),
        "spatial_dFBA": get_spatial_many_dfba(
            model_id=model_id, mol_ids=mol_ids, n_bins=n_bins_t),
    }


@composite_generator(
    name="spatial_dfba_process",
    description=(
        "Spatial microenvironment (vectorized dFBA): one spatial dFBA "
        "process updates all lattice sites as a single structured state. "
        "Demonstrates batched execution + field coupling."
    ),
    parameters={
        "n_bins": {"type": "object", "default": [5, 6]},
    },
)
def spatial_dfba_process(core=None, *, n_bins=[5, 6]):
    n_bins_t = tuple(n_bins)
    mol_ids = ["glucose", "acetate", "glycolate", "ammonium", "formate",
               "glutamate", "serine", "dissolved biomass"]
    initial_min_max = {
        "glucose": (10, 10), "glycolate": (10, 10), "ammonium": (10, 10),
        "formate": (10, 10), "glutamate": (10, 10), "serine": (0, 0),
        "acetate": (0, 0), "dissolved biomass": (0.1, 0.1),
    }
    fields = get_fields(n_bins_t, mol_ids, initial_min_max, {})
    bins_x, bins_y = n_bins_t
    horizontal_gradient = np.linspace(0, 20, bins_x).reshape(1, -1)
    fields["glucose"] = np.repeat(horizontal_gradient, bins_y, axis=0)
    model_positions = {
        "ecoli core": [(x, 0) for x in range(bins_x)],
        "ecoli":      [(x, 1) for x in range(bins_x)],
        "cdiff":      [(x, 2) for x in range(bins_x)],
        "pputida":    [(x, 3) for x in range(bins_x)],
        "yeast":      [(x, 4) for x in range(bins_x)],
        "llactis":    [(x, 5) for x in range(bins_x)],
    }
    model_grid = build_model_grid(n_bins=n_bins_t,
                                  model_positions=model_positions)
    return {
        "fields": fields,
        "spatial_dFBA": get_spatial_dFBA_process(config={
            "n_bins": n_bins_t,
            "models": MODEL_REGISTRY_DFBA,
            "model_grid": model_grid,
            "mol_ids": mol_ids,
        }),
    }


@composite_generator(
    name="diffusion_process",
    description=(
        "Field transport primitive: finite-volume diffusion/advection on a "
        "2D lattice. Use to validate boundary conditions, stability, and "
        "transport timescales."
    ),
    parameters={
        "n_bins": {"type": "object", "default": list(DEFAULT_BINS)},
        "bounds": {"type": "object", "default": list(DEFAULT_BOUNDS)},
    },
)
def diffusion_process(core=None, *, n_bins=list(DEFAULT_BINS),
                      bounds=list(DEFAULT_BOUNDS)):
    n_bins_t, bounds_t = tuple(n_bins), tuple(bounds)
    mol_ids = ["glucose", "dissolved biomass"]
    advection_coeffs = {"dissolved biomass": DEFAULT_ADVECTION}
    diffusion_coeffs = {
        "glucose": DEFAULT_DIFFUSION / 10,
        "dissolved biomass": DEFAULT_DIFFUSION / 10,
    }
    glc_field = np.random.uniform(
        low=0.1, high=2, size=(n_bins_t[1], n_bins_t[0]))
    biomass_field = np.zeros((n_bins_t[1], n_bins_t[0]))
    biomass_field[4:5, :] = 10
    return {
        "fields": {"dissolved biomass": biomass_field,
                   "glucose": glc_field},
        "diffusion": get_diffusion_advection_process(
            bounds=bounds_t, n_bins=n_bins_t, mol_ids=mol_ids,
            diffusion_coeffs=diffusion_coeffs,
            advection_coeffs=advection_coeffs),
    }
