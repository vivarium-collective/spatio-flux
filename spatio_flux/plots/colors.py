"""
Color palette and rules for assigning colors to plot nodes based on their paths.

TODO -- simplify and integrate with bigraph-viz
"""

COLORS = {
    # PARTICLES (soft, readable green)
    "particles_state":             "#C6E3D0",
    "particles_process":           "#C6E3D0",

    # PARTICLE DIVISION (clean teal, distinct rule-op)
    "particle_graph_rewrite":      "#A9DDE3",

    # NEWTONIAN PARTICLES (slightly deeper, more physical)
    "newtonian_particles_state":   "#8EC09A",
    "newtonian_particles_process": "#8EC09A",

    # FIELDS (lighter, more neutral yellow)
    "fields":                      "#F3E8B3",
    "mass":                        "#E1D07F",

    # DIFFUSION (field-adjacent, darker, clearly active)
    "diffusion":                   "#D6C35F",

    # dFBA / kinetics (warm coral → deeper red)
    "kinetic_process":             "#E3A295",
    "dfba_process":                "#B34A44",

    # EXCHANGE (warm amber, not too orange)
    "exchange":                    "#F1C27A",

    # PARTICLE ↔ FIELD ADAPTER (yellow-green bridge)
    "exchange_adapter":            "#CBDD8A",

    "orchestration": "#C9CED6"
}


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#" + "".join(f"{max(0, min(255, v)):02x}" for v in rgb)


def _darken(h, factor=0.78):  # ~22% darker for borders
    r, g, b = _hex_to_rgb(h)
    return _rgb_to_hex((int(r*factor), int(g*factor), int(b*factor)))


# ---- default species lists (can be overridden in build_plot_settings) ----
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


def build_plot_settings(
    particle_ids=None,
    n_bins=(2, 2),
    field_species=None,
    field_biomass_species=None,
    conc_type_species=None,
):
    # Normalize particle_ids into a list
    if particle_ids is None:
        particle_ids = []
    elif isinstance(particle_ids, str):
        particle_ids = [particle_ids]
    else:
        particle_ids = list(particle_ids)

    # Use defaults if not provided
    field_species = field_species or list(DEFAULT_FIELD_SPECIES)
    field_biomass_species = field_biomass_species or list(DEFAULT_FIELD_BIOMASS_SPECIES)

    fills = {
        # particle family
        ("particles",):                 COLORS["particles_state"],
        ("brownian_movement",):         COLORS["particles_process"],
        ("BrownianMovement",):          COLORS["particles_process"],
        ("particle_division",):         COLORS["particle_graph_rewrite"],
        ("ParticleDivision",):          COLORS["particle_graph_rewrite"],
        ("enforce_boundaries",):        COLORS["particle_graph_rewrite"],
        ("ManageBoundaries",):          COLORS["particle_graph_rewrite"],
        ("particle_exchange",):         COLORS["exchange_adapter"],
        ("ParticleExchange",):          COLORS["exchange_adapter"],

        # interval
        ("brownian_movement", "interval"):   COLORS["orchestration"],
        ("newtonian_particles", "interval"): COLORS["orchestration"],

        # newtonian particle family
        ("newtonian_particles",):       COLORS["newtonian_particles_process"],
        ("PymunkParticleMovement",):    COLORS["newtonian_particles_process"],

        # containers
        ("fields",):                    COLORS["fields"],
        ("concentration",):             COLORS["fields"],
        ("substrate",):                 COLORS["fields"],
        ("diffusion",):                 COLORS["diffusion"],
        ("DiffusionAdvection",):        COLORS["diffusion"],

        # dFBA / kinetics family
        ("spatial_dFBA",):              COLORS["dfba_process"],
        ("SpatialDFBA",):               COLORS["dfba_process"],
        ("spatial_dFBA", "dFBA[0,0]"):  COLORS["dfba_process"],
        ("dFBA",):                      COLORS["dfba_process"],
        ("dFBA[0,0]",):                 COLORS["dfba_process"],
        ("DynamicFBA",):                COLORS["dfba_process"],

        ("spatial_kinetics",):          COLORS["kinetic_process"],
        ("spatial_kinetics", "monod_kinetics[0,0]"): COLORS["kinetic_process"],
        ("monod_kinetics",):            COLORS["kinetic_process"],
        ("monod_kinetics[0,0]",):       COLORS["kinetic_process"],
        ("MonodKinetics",):             COLORS["kinetic_process"],
    }

    # --- auto-generate field species ---
    for s in field_species:
        # (fields, species)
        fills[("fields", s)] =          COLORS["fields"]

        # (fields, species biomass) – for things like "pputida", "llactis", etc.
        biomass_name = f"{s} biomass"
        fills[("fields", biomass_name)] = COLORS["fields"]

    for s in conc_type_species or []:
        fills[("fields", s,)] =                   COLORS["fields"]
        fills[("fields", s, "concentration",)] =  COLORS["fields"]
        fills[("fields", s, "count",)] =          COLORS["fields"]
        fills[("fields", s, "volume",)] =         COLORS["fields"]

    # --- explicitly named biomass fields ---
    for s in field_biomass_species:
        fills[("fields", s)] =              COLORS["fields"]
        fills[("fields", f'{s} biomass')] = COLORS["fields"]
        fills[(f'{s} dFBA',)] =             COLORS["dfba_process"]

    # ---- particle-specific stuff unchanged, example: ----
    for pid in particle_ids:
        fills.update({
            ("particles", pid):             COLORS["particles_state"],
            ("particles", pid, "id"):       COLORS["particles_state"],
            ("particles", pid, "position"): COLORS["particles_state"],
            ("particles", pid, "mass"):     COLORS["particles_state"],
            ("particles", pid, "sub_masses"): COLORS["particles_state"],
            ('particles', pid, 'local'):    COLORS["fields"],
            ('particles', pid, 'exchange'): COLORS["exchange"],
            ('particles', pid, 'dFBA'):     COLORS["dfba_process"],
            ('particles', pid, 'monod_kinetics'): COLORS["kinetic_process"],
            ('particles', pid, 'shape'):    COLORS["newtonian_particles_state"],
            ('particles', pid, 'velocity'): COLORS["newtonian_particles_state"],
            ('particles', pid, 'inertia'):  COLORS["newtonian_particles_state"],
            ('particles', pid, 'radius'):   COLORS["newtonian_particles_state"],
            ('particles', pid, 'elasticity'): COLORS["newtonian_particles_state"],
            ('particles', pid, 'friction'): COLORS["newtonian_particles_state"],
            ('particles', pid, 'dFBA_ecoli_1'): COLORS["dfba_process"],
            ('particles', pid, 'dFBA_ecoli_2'): COLORS["dfba_process"],
            ('particles', pid, 'aggregate_mass'): COLORS["exchange_adapter"],
        })
        fills.update({
            (pid,):             COLORS["particles_state"],
            (pid, "id"):        COLORS["particles_state"],
            (pid, "position"):  COLORS["particles_state"],
            (pid, "mass"):      COLORS["particles_state"],
            (pid, 'local'):     COLORS["fields"],
            (pid, 'exchange'):  COLORS["exchange"],
            (pid, 'dFBA'):      COLORS["dfba_process"],
            (pid, 'monod_kinetics'): COLORS["kinetic_process"],
            (pid, 'shape'):     COLORS["newtonian_particles_state"],
            (pid, 'velocity'):  COLORS["newtonian_particles_state"],
            (pid, 'inertia'):   COLORS["newtonian_particles_state"],
            (pid, 'radius'):    COLORS["newtonian_particles_state"],
            (pid, 'elasticity'): COLORS["newtonian_particles_state"],
            (pid, 'friction'):  COLORS["newtonian_particles_state"],
        })

        for species in field_species:
            fills[('particles', pid, 'local', species)] =    COLORS["fields"]
            fills[('particles', pid, 'exchange', species)] = COLORS["exchange"]

    borders = {k: _darken(v) for k, v in fills.items()}

    return {
        "node_fill_colors": fills,
        "node_border_colors": borders,
    }
