COLORS = {
    # --- PARTICLES FAMILY (sage greens + darker processes) ---
    "particles_base":  "#B8D0C0",   # soft light sage for container and states
    "particles_process": "#6F9C81", # muted green for movement/division

    # --- dFBA FAMILY (cool desaturated blues) ---
    "dfba_base":   "#5C7FA0",       # medium desaturated blue
    "dfba_light":  "#C3D5E4",       # pale blue-gray

    # --- FIELDS FAMILY (warm muted rose & red) ---
    "fields":      "#D1918C",       # dusty rose
    "diffusion":   "#B7504D",       # muted brick red (stronger red balance)

    # --- CROSS-DOMAIN / BRIDGES ---
    "particle_exchange_bridge": "#B4B899",  # olive-sage bridge tone

    # --- dfBA-LIKE SUPPORT FAMILIES (cool neutrals) ---
    "local":       "#D6DDF0",       # pale periwinkle
    "exchange":    "#B6D0D8",       # cool gray-cyan
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
        ("particle_movement",):     COLORS["particles_process"],
        ("particle_division",):     COLORS["particles_process"],
        ("particle_exchange",):     COLORS["particle_exchange_bridge"],

        # containers
        ("fields",):                COLORS["fields"],
        ("diffusion",):             COLORS["diffusion"],

        # dFBA
        ("spatial_dfba",):          COLORS["dfba_light"],
        ("spatial_dfba", "dFBA[0,0]"): COLORS["dfba_base"],
        ("dFBA",):                  COLORS["dfba_base"],
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
            # ...
        })

    borders = {k: _darken(v) for k, v in fills.items()}

    return {
        "node_fill_colors": fills,
        "node_border_colors": borders,
    }
