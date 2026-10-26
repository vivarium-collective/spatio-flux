"""Novel visually-rich composites for the discovery-engine studies.

- ``spatial_cross_feeding``: strain A (glucose side) secretes acetate that diffuses
  to strain B (no-glucose side), which grows on it — an emergent spatial partnership.
- ``comets_washout``: a dFBA biomass field advected toward an OUTFLOW boundary, so a
  colony persists below a critical advection and washes out above it.

Both wire existing real processes (spatial dFBA + advection–diffusion) in new
spatial arrangements — no new science, no mocks.
"""
import numpy as np
from pbg_superpowers.composite_generator import composite_generator

from spatio_flux.processes import (
    get_fields_with_schema,
    get_diffusion_advection_process,
    get_spatial_many_dfba,
    get_spatial_many_kinetics,
)
from spatio_flux.composites._constants import SQUARE_BINS, SQUARE_BOUNDS


@composite_generator(
    name="spatial_cross_feeding",
    description=(
        "Spatial cross-feeding: strain A grows on glucose (seeded on one side) and "
        "secretes acetate that diffuses across the lattice to strain B (seeded on the "
        "no-glucose side), which grows on the acetate. An emergent spatial partnership "
        "— B cannot grow without A's secreted acetate bridge."
    ),
    parameters={
        "model_id":  {"type": "string", "default": "ecoli core"},
        "n_bins":    {"type": "object", "default": list(SQUARE_BINS)},
        "bounds":    {"type": "object", "default": list(SQUARE_BOUNDS)},
        "a_absent":  {"type": "boolean", "default": False},  # control: no strain A
    },
)
def spatial_cross_feeding(core=None, *, model_id="ecoli core",
                          n_bins=list(SQUARE_BINS), bounds=list(SQUARE_BOUNDS),
                          a_absent=False):
    mol_ids = ["glucose", "acetate", "biomass_A", "biomass_B"]
    n_bins_t, bounds_t = tuple(n_bins), tuple(bounds)
    nx, ny = n_bins_t
    shape = (ny, nx)
    half = nx // 2
    # glucose only on strain A's side (left); acetate starts at zero everywhere.
    glucose = np.zeros(shape, dtype=float)
    glucose[:, :half] = 10.0
    acetate = np.zeros(shape, dtype=float)
    biomass_A = np.zeros(shape, dtype=float)
    if not a_absent:
        biomass_A[:, :half] = 0.1           # A on the glucose side
    biomass_B = np.zeros(shape, dtype=float)
    biomass_B[:, half:] = 0.1               # B on the no-glucose side
    initial_fields = {"glucose": glucose, "acetate": acetate,
                      "biomass_A": biomass_A, "biomass_B": biomass_B}
    # acetate diffuses fast (the bridge); biomass fields barely diffuse; glucose fixed.
    diffusion_coeffs = {"glucose": 0.0, "acetate": 2e-1, "biomass_A": 1e-3, "biomass_B": 1e-3}
    return {
        # strain A: per-site dFBA on glucose → biomass_A, secreting acetate (keys dFBA[x,y])
        **get_spatial_many_dfba(model_id=model_id, mol_ids=["glucose", "acetate", "biomass_A"],
                                n_bins=n_bins_t, biomass_id="biomass_A", path=["fields"]),
        # strain B: per-site acetate-only Monod kinetics → biomass_B (keys monod_kinetics[i,j])
        **get_spatial_many_kinetics(model_id="acetate_only", mol_ids=["glucose", "acetate", "biomass_B"],
                                    n_bins=n_bins_t, biomass_id="biomass_B", path=["fields"]),
        "fields": get_fields_with_schema(n_bins=n_bins_t, mol_ids=mol_ids, initial_fields=initial_fields),
        "diffusion": get_diffusion_advection_process(
            bounds=bounds_t, n_bins=n_bins_t, mol_ids=mol_ids, diffusion_coeffs=diffusion_coeffs),
    }
