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


def build_plot_settings(particle_ids=None, n_bins=(2, 2)):
    # Normalize particle_ids into a list
    if particle_ids is None:
        particle_ids = []
    elif isinstance(particle_ids, str):
        particle_ids = [particle_ids]
    else:
        particle_ids = list(particle_ids)

    # ---- fills (family-first) ----
    fills = {
        # particle family (greens)
        ('particles',):                          COLORS["particles_base"],
        ('particle_movement',):                  COLORS["particles_process"],
        ('particle_division',):                  COLORS["particles_process"],
        ('particle_exchange',):                  COLORS["particle_exchange_bridge"],

        # fields & diffusion + field substrates
        ('fields',):                             COLORS["fields"],
        ('fields', 'glucose'):                   COLORS["fields"],
        ('fields', 'acetate'):                   COLORS["fields"],
        ('fields', 'dissolved biomass'):         COLORS["fields"],
        ('diffusion',):                          COLORS["diffusion"],

        # dFBA family (blues)
        ('spatial_dfba',):                       COLORS["dfba_light"],
        ('spatial_dfba', 'dFBA[0,0]'):           COLORS["dfba_base"],
        ('dFBA',):                               COLORS["dfba_base"],
    }

    # ---- particle-specific fills (repeat per particle) ----
    for pid in particle_ids:
        fills.update({
            ('particles', pid):                           COLORS["particles_base"],
            ('particles', pid, 'id'):                     COLORS["particles_base"],
            ('particles', pid, 'position'):               COLORS["particles_base"],

            ('particles', pid, 'local'):                  COLORS["local"],
            ('particles', pid, 'exchange'):               COLORS["exchange"],

            ('particles', pid, 'local', 'acetate'):        COLORS["local"],
            ('particles', pid, 'local', 'glucose'):        COLORS["local"],
            ('particles', pid, 'local', 'dissolved biomass'): COLORS["local"],

            ('particles', pid, 'exchange', 'acetate'):     COLORS["exchange"],
            ('particles', pid, 'exchange', 'glucose'):     COLORS["exchange"],
            ('particles', pid, 'exchange', 'dissolved biomass'): COLORS["exchange"],

            # particle-scoped dFBA
            ('particles', pid, 'dFBA'):                    COLORS["dfba_base"],

            # mass/biomass bridge under particle
            ('particles', pid, 'mass'):                    COLORS["particles_base"],
        })

    # ---- borders = darker versions of fills ----
    borders = {k: _darken(v) for k, v in fills.items()}

    return {
        'node_fill_colors': fills,
        'node_border_colors': borders,
    }
