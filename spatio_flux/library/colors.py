COLORS = {
    # --- PARTICLES (greens) ---
    "particles_base":        "#C9DCCF",
    "particles_process":     "#7FA889",

    # --- NEWTONIAN PARTICLES (teal greens) ---
    "newtonian_particles_process": "#3F716F",
    "newtonian_particles_state":   "#8DBFBA",

    # --- FIELDS (LB media yellows) ---
    "fields":      "#E8DFAF",
    "diffusion":   "#C8A837",

    # --- dFBA (metabolic reds) ---
    "dfba_base":   "#B84C48",
    "dfba_light":  "#E6B4B0",

    # --- LOCAL / EXCHANGE (fields ↔ dFBA bridge: yellow → orange) ---
    "local":       "#E9C88B",
    "exchange":    "#D9A56F",

    # --- PARTICLE ↔ FIELD BRIDGE (green → yellow olive) ---
    "particle_exchange_bridge": "#BFC78F",
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
]


def build_plot_settings(
    particle_ids=None,
    n_bins=(2, 2),
    field_species=None,
    field_biomass_species=None,
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
        ("particles",):             COLORS["particles_base"],
        ("brownian_movement",):     COLORS["particles_process"],
        ("particle_division",):     COLORS["particles_process"],
        ("particle_exchange",):     COLORS["particle_exchange_bridge"],

        # newtonian particle family
        ("newtonian_particles",):           COLORS["newtonian_particles_process"],

        # containers
        ("fields",):                COLORS["fields"],
        ("diffusion",):             COLORS["diffusion"],

        # dFBA
        ("spatial_dFBA",):          COLORS["dfba_light"],
        ("spatial_dFBA", "dFBA[0,0]"): COLORS["dfba_base"],
        ("dFBA",):                  COLORS["dfba_base"],
        ("monod_kinetics",):            COLORS["dfba_base"],
    }

    # --- auto-generate field species ---
    for s in field_species:
        # (fields, species)
        fills[("fields", s)] = COLORS["fields"]

        # (fields, species biomass) – for things like "pputida", "llactis", etc.
        biomass_name = f"{s} biomass"
        fills[("fields", biomass_name)] = COLORS["fields"]

    # --- explicitly named biomass fields ---
    for s in field_biomass_species:
        fills[("fields", s)] = COLORS["fields"]
        fills[("fields", f'{s} biomass')] = COLORS["fields"]
        fills[(f'{s} dFBA',)] = COLORS["dfba_base"]

    # ---- particle-specific stuff unchanged, example: ----
    for pid in particle_ids:
        fills.update({
            ("particles", pid):             COLORS["particles_base"],
            ("particles", pid, "id"):       COLORS["particles_base"],
            ("particles", pid, "position"): COLORS["particles_base"],
            ("particles", pid, "mass"):     COLORS["particles_base"],
            ('particles', pid, 'local'):    COLORS["local"],
            ('particles', pid, 'exchange'): COLORS["exchange"],
            ('particles', pid, 'dFBA'):     COLORS["dfba_base"],
            ('particles', pid, 'monod_kinetics'):  COLORS["dfba_base"],
            ('particles', pid, 'shape'):  COLORS["newtonian_particles_state"],
            ('particles', pid, 'velocity'):  COLORS["newtonian_particles_state"],
            ('particles', pid, 'inertia'):  COLORS["newtonian_particles_state"],
            ('particles', pid, 'radius'): COLORS["newtonian_particles_state"],
            ('particles', pid, 'elasticity'): COLORS["newtonian_particles_state"],
            ('particles', pid, 'friction'): COLORS["newtonian_particles_state"],
        })
        fills.update({
            (pid,):             COLORS["particles_base"],
            (pid, "id"):       COLORS["particles_base"],
            (pid, "position"): COLORS["particles_base"],
            (pid, "mass"):     COLORS["particles_base"],
            (pid, 'local'):    COLORS["local"],
            (pid, 'exchange'): COLORS["exchange"],
            (pid, 'dFBA'):     COLORS["dfba_base"],
            (pid, 'monod_kinetics'):  COLORS["dfba_base"],
            (pid, 'shape'):  COLORS["newtonian_particles_state"],
            (pid, 'velocity'):  COLORS["newtonian_particles_state"],
            (pid, 'inertia'):  COLORS["newtonian_particles_state"],
            (pid, 'radius'): COLORS["newtonian_particles_state"],
            (pid, 'elasticity'): COLORS["newtonian_particles_state"],
            (pid, 'friction'): COLORS["newtonian_particles_state"],
        })

        for species in field_species:
            fills[('particles', pid, 'local', species)] = COLORS["local"]
            fills[('particles', pid, 'exchange', species)] = COLORS["exchange"]

    borders = {k: _darken(v) for k, v in fills.items()}

    return {
        "node_fill_colors": fills,
        "node_border_colors": borders,
    }
