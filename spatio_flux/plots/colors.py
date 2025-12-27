"""
Color palette and rules for assigning colors to plot nodes based on their paths.

TODO -- simplify and integrate with bigraph-viz
"""

import numpy as np


# -----------------------------
# Palette
# -----------------------------

COLORS = {
    # PARTICLES (soft sage → slightly bluer sage)
    "particles_state":             "#C3E1D6",
    "particles_process":           "#C3E1D6",

    # PARTICLE DIVISION (teal, slightly more blue)
    "particle_graph_rewrite":      "#9FD8DF",

    # NEWTONIAN PARTICLES (deeper cool green)
    "newtonian_particles_state":   "#7FBFA8",
    "newtonian_particles_process": "#7FBFA8",

    # FIELDS (warm light yellow, reduced green tint)
    "fields":                      "#F4E6B8",
    "mass":                        "#E4CF77",

    # DIFFUSION (olive → more mustard, less green)
    "diffusion":                   "#D1BE56",

    # dFBA / kinetics (coral → mauve, red reduced)
    "kinetic_process":             "#D9A4A6",
    "dfba_process":                "#A24E56",

    # EXCHANGE (amber, slightly darker for contrast)
    "exchange":                    "#EEC06F",

    # PARTICLE ↔ FIELD ADAPTER (yellow-green → chartreuse-leaning)
    "exchange_adapter":            "#C6D780",

    # ORCHESTRATION (unchanged neutral)
    "orchestration":               "#C9CED6"
}


# -----------------------------
# Helpers
# -----------------------------

def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#" + "".join(f"{max(0, min(255, v)):02x}" for v in rgb)


def _darken(h: str, factor: float = 0.78) -> str:  # ~22% darker for borders
    r, g, b = _hex_to_rgb(h)
    return _rgb_to_hex((int(r * factor), int(g * factor), int(b * factor)))


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _set_many(dst: dict, keys, color_hex: str):
    for k in keys:
        dst[k] = color_hex


# -----------------------------
# Defaults (override in build_plot_settings)
# -----------------------------

DEFAULT_FIELD_SPECIES = [
    "glucose",
    "oxygen",
    "acetate",
    "formate",
    "ammonium",
    "serine",
    "lactate",
    "glycolate",
    "glutamate",
    "detritus",
    "biomass",
    "dissolved biomass",
    "substrate_id",
    "dfba_biomass",
    "kinetic_biomass",
    "monod_biomass",
    "ecoli_biomass",
    "yeast_biomass",
    "dfba biomass",
    "kinetic biomass",
]

DEFAULT_FIELD_BIOMASS_SPECIES = [
    "pputida",
    "llactis",
    "ecoli",
    "ecoli core",
    "yeast",
    "cdiff",
    "biomass",
    "dissolved biomass",
    "ecoli_1",
    "ecoli_2",
    "monod biomass",
]


# -----------------------------
# Main
# -----------------------------

def build_plot_settings(
    particle_ids=None,
    field_species=None,
    field_biomass_species=None,
):
    particle_ids = _as_list(particle_ids)
    field_species = list(field_species) if field_species is not None else list(DEFAULT_FIELD_SPECIES)
    field_biomass_species = (
        list(field_biomass_species) if field_biomass_species is not None else list(DEFAULT_FIELD_BIOMASS_SPECIES)
    )
    remove_paths = []
    fills = {}

    # ---- fixed (non-generated) rules ----
    _set_many(fills, [("particles",), ("brownian_movement",), ("BrownianMovement",)
                      ], COLORS["particles_state"])

    _set_many(fills, [("brownian_movement",), ("BrownianMovement",)
                      ], COLORS["particles_process"])

    _set_many(fills, [
        ("particle_division",),
        ("ParticleDivision",),
        ("enforce_boundaries",),
        ("ManageBoundaries",),
    ], COLORS["particle_graph_rewrite"])

    _set_many(fills, [("particle_exchange",), ("ParticleExchange",)
                      ], COLORS["exchange_adapter"])

    # interval
    _set_many(fills, [("brownian_movement", "interval"), ("newtonian_particles", "interval")
                       ], COLORS["orchestration"])

    # newtonian particle family
    _set_many(fills, [("newtonian_particles",), ("PymunkParticleMovement",)
                      ], COLORS["newtonian_particles_process"])

    # containers
    _set_many(fills, [("fields",), ("concentration",), ("substrate",), ("lattice",), ("lattice", "fields"), ("lattice", "bin_volume"),
                      ], COLORS["fields"])

    _set_many(fills, [("diffusion",),  ("DiffusionAdvection",), ("lattice", "diffusion"),
                      ], COLORS["diffusion"])

    _set_many(fills, [("lattice", "exchanges"),], COLORS["exchange"])

    fills[("lattice", "conc_count_adapter")] = COLORS["exchange_adapter"]
    fills[("lattice", "monod_kinetics[0,0]")] = COLORS["kinetic_process"]

    # dFBA / kinetics family
    _set_many(fills, [("spatial_dFBA",), ("SpatialDFBA",), ("spatial_dFBA", "dFBA[0,0]"), ("dFBA",), ("dFBA[0,0]",), ("DynamicFBA",)
                      ], COLORS["dfba_process"])

    _set_many(fills, [("spatial_kinetics",), ("spatial_kinetics", "monod_kinetics[0,0]"),
                      ("monod_kinetics",), ("monod_kinetics[0,0]"), ("MonodKinetics",),
                      ], COLORS["kinetic_process"])

    # ---- auto-generate field species ----
    for s in field_species:
        _set_many(fills, [
            ("fields", s), ("fields", f"{s} biomass"), ("lattice", "fields", s),("lattice", "exchanges", s)
        ], COLORS["fields"])
        fills[("lattice", "exchanges", s)] = COLORS["exchange"]  # override (matches original intent)

    # ---- explicitly named biomass fields ----
    for s in field_biomass_species:
        _set_many(fills, [
            ("fields", s), ("fields", f"{s} biomass"), ("lattice", "fields", s), ("lattice", "fields", f"{s} biomass"),
        ], COLORS["fields"])

        _set_many(fills, [(f"{s} dFBA",), ("lattice", f"{s} dFBA")], COLORS["dfba_process"])

        _set_many(fills, [("lattice", "exchanges", s), ("lattice", "exchanges", f"{s} biomass"),
                          ], COLORS["exchange"])

    # ---- particle-specific mappings (preserve exact keys) ----
    particle_scalar_keys = (
        "id", "position", "mass", "sub_masses",
        "shape", "velocity", "inertia", "radius", "elasticity", "friction",
    )

    # keys that exist only under ('particles', pid, ...)
    particles_only = {
        "local": COLORS["fields"],
        "exchange": COLORS["exchange"],
        "dFBA": COLORS["dfba_process"],
        "monod_kinetics": COLORS["kinetic_process"],
        "aggregate_mass": COLORS["exchange_adapter"],
    }

    # special hardcoded sub_masses children preserved from original
    submass_children = ("ecoli_1", "ecoli_2")

    for pid in particle_ids:
        particle_id = ["particles", pid, "id"]
        remove_paths.append(particle_id)

        # ('particles', pid, ...)
        fills[("particles", pid)] = COLORS["particles_state"]
        for k in particle_scalar_keys:
            if k in ("shape", "velocity", "inertia", "radius", "elasticity", "friction"):
                fills[("particles", pid, k)] = COLORS["newtonian_particles_state"]
            else:
                fills[("particles", pid, k)] = COLORS["particles_state"]

        for child in submass_children:
            fills[("particles", pid, "sub_masses", child)] = COLORS["particles_state"]

        for k, c in particles_only.items():
            fills[("particles", pid, k)] = c
            # Missing per-particle dFBA keys from original
            fills[("particles", pid, "ecoli_1 dFBA")] = COLORS["dfba_process"]
            fills[("particles", pid, "ecoli_2 dFBA")] = COLORS["dfba_process"]

            # Optional: be robust to alternate monod kinetics key spellings under particles
            fills[("particles", pid, "monod_kinetics")] = COLORS["kinetic_process"]

        # (pid, ...)
        fills[(pid,)] = COLORS["particles_state"]
        for k in particle_scalar_keys:
            if k in ("shape", "velocity", "inertia", "radius", "elasticity", "friction"):
                fills[(pid, k)] = COLORS["newtonian_particles_state"]
            else:
                fills[(pid, k)] = COLORS["particles_state"]

        # (pid, 'local'/'exchange' etc) — these existed in the original second block
        fills[(pid, "local")] = COLORS["fields"]
        fills[(pid, "exchange")] = COLORS["exchange"]
        fills[(pid, "dFBA")] = COLORS["dfba_process"]
        fills[(pid, "monod_kinetics")] = COLORS["kinetic_process"]

        # per-species under local/exchange
        for species in field_species:
            fills[("particles", pid, "local", species)] = COLORS["fields"]
            fills[("particles", pid, "exchange", species)] = COLORS["exchange"]

    borders = {k: _darken(v) for k, v in fills.items()}

    return {
        "node_fill_colors": fills,
        "node_border_colors": borders,
        "remove_paths": remove_paths
    }
