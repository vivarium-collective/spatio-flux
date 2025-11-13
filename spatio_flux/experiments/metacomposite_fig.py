import os
import json

from spatio_flux.experiments.test_suite import DEFAULT_BOUNDS, DEFAULT_BINS, DEFAULT_ADVECTION
from spatio_flux.library.helpers import get_standard_emitter, inverse_tuple
from spatio_flux import register_types
from vivarium.vivarium import VivariumTypes, Composite
from bigraph_viz import plot_bigraph

from spatio_flux.processes import DIVISION_MASS_THRESHOLD, get_fields, get_diffusion_advection_process, \
    get_spatial_many_dfba, get_particles_state, get_particle_movement_process, get_particle_exchange_process, \
    get_particle_divide_process, get_dfba_particle_composition


def get_particle_multi_dfba_comets_doc(core=None, config=None):
    config = config or {}
    particle_model_id = config.get('particle_model_id', 'ecoli core')
    dissolved_model_id = config.get('dissolved_model_id', 'ecoli core')
    division_mass_threshold=config.get('division_mass_threshold', DIVISION_MASS_THRESHOLD) # divide at mass 5.0

    mol_ids = ['glucose', 'acetate', 'biomass']
    initial_min_max = {'glucose': (1, 5), 'acetate': (0, 0), 'biomass': (0, 0.1)}
    bounds = DEFAULT_BOUNDS
    n_bins = config.get('n_bins', DEFAULT_BINS)
    advection_coeffs = {'biomass': inverse_tuple(DEFAULT_ADVECTION)}
    n_particles = 1
    add_probability = 0.2
    particle_advection = DEFAULT_ADVECTION
    fields = get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max)

    # spatial dfba config
    spatial_dfba_config = {'mol_ids': mol_ids, 'n_bins': n_bins}

    doc = {
        'state': {
            'fields': fields,
            'diffusion': get_diffusion_advection_process(bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, advection_coeffs=advection_coeffs),
            'spatial_dfba': get_spatial_many_dfba(model_file=dissolved_model_id, mol_ids=mol_ids, n_bins=n_bins),
            # 'spatial_dfba': get_spatial_dfba_process(model_id=dissolved_model_id, config=spatial_dfba_config),
            'particles': get_particles_state(n_particles=n_particles, n_bins=n_bins, bounds=bounds, fields=fields),
            'particle_movement': get_particle_movement_process(
                n_bins=n_bins, bounds=bounds, advection_rate=particle_advection, add_probability=add_probability),
            'particle_exchange': get_particle_exchange_process(n_bins=n_bins, bounds=bounds),
            'particle_division': get_particle_divide_process(division_mass_threshold=division_mass_threshold),
        },
        'composition': get_dfba_particle_composition(model_file=particle_model_id)
    }
    return doc


COLORS = {
    # --- PARTICLES FAMILY (sage greens, lighter states + darker processes) ---
    # States (lighter)
    "particles_base":  "#B8D0C0",   # soft light sage for container and states
    "particles_mid":   "#D9E7DF",   # very pale tint for id/position
    # Processes (darker)
    "particles_process": "#6F9C81", # muted green for movement/division

    # --- dFBA FAMILY (cool desaturated blues) ---
    "dfba_base":   "#5C7FA0",       # medium desaturated blue
    "dfba_light":  "#C3D5E4",       # pale blue-gray

    # --- FIELDS FAMILY (warm muted rose & red) ---
    "fields":      "#D1918C",       # dusty rose
    "diffusion":   "#B7504D",       # muted brick red (stronger red balance)

    # --- CROSS-DOMAIN / BRIDGES ---
    "particle_exchange_bridge": "#B4B899",  # olive-sage bridge tone (particles ↔ fields)

    # --- dfBA-LIKE SUPPORT FAMILIES (cool neutrals) ---
    "local":       "#D6DDF0",       # pale periwinkle
    "exchange":    "#B6D0D8",       # cool gray-cyan
    "mass":        "#93B7B4",       # cooler blue-green gray (distinct from particles)
}






# --- simple color utilities ---
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return "#" + "".join(f"{max(0, min(255, v)):02x}" for v in rgb)

def _darken(h, factor=0.78):  # ~22% darker for borders
    r, g, b = _hex_to_rgb(h)
    return _rgb_to_hex((int(r*factor), int(g*factor), int(b*factor)))

def build_plot_settings(particle_id=None, n_bins=(2, 2)):
    """
    Returns plot_settings with calm palette, auto-derived borders, and modular family rules.
    - Particles: greens (container, processes, id/position)
    - dFBA: blues (family + all dFBA[i,j] + particle-scoped dFBA)
    - Fields/diffusion/substrates: warm tones
    - Mass/Biomass unified 'bridge' color across families
    - Local/Exchange children colored consistently (both generic and particle-scoped)
    """
    # enumerate all dFBA nodes from n_bins
    dFBA_nodes = [('spatial_dfba', f'dFBA[{i},{j}]')
                  for i in range(n_bins[0]) for j in range(n_bins[1])]

    # ---- fills (family-first) ----
    fills = {
        # particle family (greens)
        ('particles',):                          COLORS["particles_base"],
        ('particle_movement',):                  COLORS["particles_process"],
        ('particle_division',):                  COLORS["particles_process"],
        ('particle_exchange',):                  COLORS["exchange"],

        # shared color for mass/biomass across families
        ('fields', 'biomass'):                   COLORS["mass"],

        # fields & diffusion + field substrates
        ('fields',):                             COLORS["fields"],
        ('fields', 'glucose'):                   COLORS["fields"],
        ('fields', 'acetate'):                   COLORS["fields"],
        ('diffusion',):                          COLORS["diffusion"],

        # dFBA family (blues)
        ('spatial_dfba',):                       COLORS["dfba_light"],
        **{node: COLORS["dfba_base"] for node in dFBA_nodes},
    }

    # unique particle + particle-owned states (guarded)
    if particle_id:
        fills.update({
            ('particles', particle_id):               COLORS["particles_mid"],
            ('particles', particle_id, 'id'):         COLORS["particles_mid"],
            ('particles', particle_id, 'position'):   COLORS["particles_mid"],
            ('particles', particle_id, 'local'):      COLORS["local"],
            ('particles', particle_id, 'exchange'):   COLORS["exchange"],
            ('particles', particle_id, 'local', 'acetate'):  COLORS["local"],
            ('particles', particle_id, 'local', 'glucose'):  COLORS["local"],
            ('particles', particle_id, 'exchange', 'acetate'): COLORS["exchange"],
            ('particles', particle_id, 'exchange', 'glucose'): COLORS["exchange"],
            # particle-scoped dFBA
            ('particles', particle_id, 'dFBA'):       COLORS["dfba_base"],
            # mass/biomass bridge under particle
            ('particles', particle_id, 'mass'):       COLORS["mass"],
        })

    # ---- borders (auto = darker of fill; override here if needed) ----
    borders = {k: _darken(v) for k, v in fills.items()}

    # Return only the color maps (you can add dpi/show_types externally if desired)
    return {
        'node_fill_colors': fills,
        'node_border_colors': borders,
    }


def main():
    name = "metacomposite"
    outdir = "out"
    n_bins = (1, 3)

    document = get_particle_multi_dfba_comets_doc(config={'n_bins': n_bins})
    core = VivariumTypes()
    core = register_types(core)
    document = {'state': document} if 'state' not in document else document

    # make the composite
    sim = Composite(document, core=core)

    # Save composition JSON
    sim.save(filename=f"{name}.json", outdir=outdir)

    # Save representation string (human-readable schema summary)
    rep = core.representation(document)
    rep_file = os.path.join(outdir, f"{name}_schema.txt")
    with open(rep_file, "w") as f:
        f.write(rep)
    print(f"💾 Saved schema representation → {rep_file}")

    # Save the underlying schema as JSON (machine-readable)
    schema_file = os.path.join(outdir, f"{name}_schema.json")
    try:
        with open(schema_file, "w") as f:
            json.dump(document, f, indent=2)
        print(f"💾 Saved schema JSON → {schema_file}")
    except Exception as e:
        print(f"⚠ Could not save schema JSON: {e}")

    # Visualize initial composition
    plot_state = {k: v for k, v in sim.state.items() if k not in ['global_time', 'emitter']}
    plot_schema = {k: v for k, v in sim.composition.items() if k not in ['global_time', 'emitter']}

    particle_id = list(plot_state['particles'].keys())[0]

    # plot
    plot_settings = build_plot_settings(particle_id=particle_id, n_bins=n_bins)
    plot_settings.update(dict(
        dpi='300',
        show_types=True,
        collapse_redundant_processes={
            'exclude': [('particle_movement',), ('particle_division',)]}
    ))

    plot_bigraph(
        state=plot_state,
        schema=plot_schema,
        core=core,
        out_dir=outdir,
        filename=f"{name}_viz",
        **plot_settings,
    )



if __name__ == "__main__":
    main()


