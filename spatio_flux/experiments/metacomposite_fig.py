import os
import json

from spatio_flux.experiments.test_suite import DEFAULT_BOUNDS, DEFAULT_BINS, DEFAULT_ADVECTION
from spatio_flux.library.helpers import get_standard_emitter, inverse_tuple, reversed_tuple
from spatio_flux import register_types
from vivarium.vivarium import VivariumTypes, Composite
from bigraph_viz import plot_bigraph

from spatio_flux.processes import DIVISION_MASS_THRESHOLD, get_fields, get_diffusion_advection_process, \
    get_spatial_many_dfba, get_particles_state, get_particle_movement_process, get_particle_exchange_process, \
    get_particle_divide_process, get_dfba_particle_composition, get_fields_with_schema, get_single_dfba_process, \
    get_dfba_process_from_registry


# -------------------------------------------------------------------
# shared configuration for local fields
# -------------------------------------------------------------------
MOL_IDS = ['glucose', 'acetate', 'dissolved biomass']
INITIAL_MIN_MAX = {
    'glucose': (1, 5),
    'acetate': (0, 0),
    'dissolved biomass': (0, 0.1),
}


def get_particle_multi_dfba_comets_doc(core=None, config=None):
    config = config or {}
    particle_model_id = config.get('particle_model_id', 'ecoli core')
    dissolved_model_id = config.get('dissolved_model_id', 'ecoli core')
    division_mass_threshold = config.get('division_mass_threshold', DIVISION_MASS_THRESHOLD)

    mol_ids = MOL_IDS
    initial_min_max = INITIAL_MIN_MAX
    bounds = DEFAULT_BOUNDS
    n_bins = config.get('n_bins', DEFAULT_BINS)
    advection_coeffs = {'dissolved biomass': inverse_tuple(DEFAULT_ADVECTION)}
    n_particles = config.get('n_particles', 1)
    add_probability = config.get('add_probability', 0.2)
    particle_advection = config.get('particle_advection', DEFAULT_ADVECTION)
    fields = get_fields(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max)

    # spatial dfba config
    spatial_dfba_config = {'mol_ids': mol_ids, 'n_bins': n_bins}

    doc = {
        'state': {
            'fields': fields,
            'diffusion': get_diffusion_advection_process(
                bounds=bounds, n_bins=n_bins, mol_ids=mol_ids, advection_coeffs=advection_coeffs),
            'spatial_dfba': get_spatial_many_dfba(
                model_file=dissolved_model_id, mol_ids=mol_ids, n_bins=n_bins, biomass_id="dissolved biomass"),
            # 'spatial_dfba': get_spatial_dfba_process(model_id=dissolved_model_id, config=spatial_dfba_config),
            'particles': get_particles_state(
                n_particles=n_particles, n_bins=n_bins, bounds=bounds, fields=fields),
            'particle_movement': get_particle_movement_process(
                n_bins=n_bins, bounds=bounds, advection_rate=particle_advection,
                add_probability=add_probability),
            'particle_exchange': get_particle_exchange_process(
                n_bins=n_bins, bounds=bounds),
            'particle_division': get_particle_divide_process(
                division_mass_threshold=division_mass_threshold),
        },
        'composition': get_dfba_particle_composition(model_file=particle_model_id)
    }
    return doc


# -------------------------------------------------------------------
# 1) just particle_movement on particles
# -------------------------------------------------------------------
def get_particles_movement_doc(core=None, config=None):
    """
    Particles + particle_movement process (no diffusion, no spatial dFBA, no particle dFBA).
    """
    config = config or {}
    bounds = config.get('bounds', DEFAULT_BOUNDS)
    n_bins = config.get('n_bins', DEFAULT_BINS)
    n_particles = config.get('n_particles', 1)
    add_probability = config.get('add_probability', 0.2)
    particle_advection = config.get('particle_advection', DEFAULT_ADVECTION)
    state = {
        'particles': get_particles_state(
            n_particles=n_particles, n_bins=n_bins, bounds=bounds),
        'particle_movement': get_particle_movement_process(
            n_bins=n_bins, bounds=bounds,
            advection_rate=particle_advection,
            add_probability=add_probability),
    }

    return {
        'state': state,
        'composition': {},   # no extra composition; only movement
    }


# -------------------------------------------------------------------
# 2) just diffusion on fields
# -------------------------------------------------------------------
def get_fields_diffusion_doc(core=None, config=None):
    """
    Fields + diffusion/advection only (no particles, no dFBA).
    """
    config = config or {}
    bounds = config.get('bounds', DEFAULT_BOUNDS)
    n_bins = config.get('n_bins', DEFAULT_BINS)
    mol_ids = MOL_IDS
    initial_min_max = INITIAL_MIN_MAX

    fields = get_fields(
        n_bins=n_bins,
        mol_ids=mol_ids,
        initial_min_max=initial_min_max,
    )
    advection_coeffs = {'dissolved biomass': inverse_tuple(DEFAULT_ADVECTION)}

    state = {
        'fields': fields,
        'diffusion': get_diffusion_advection_process(
            bounds=bounds,
            n_bins=n_bins,
            mol_ids=mol_ids,
            advection_coeffs=advection_coeffs),
    }

    return {
        'state': state,
        'composition': {},   # only diffusion process
    }


# -------------------------------------------------------------------
# 4) just spatial dFBA on the field grid
# -------------------------------------------------------------------
def get_spatial_dfba_doc(core=None, config=None):
    dissolved_model_file = 'ecoli core'
    mol_ids = ['glucose', 'acetate', 'dissolved biomass']
    initial_min_max = {'glucose': (0, 20), 'acetate': (0, 0), 'dissolved biomass': (0, 0.1)}
    n_bins = config.get('n_bins', DEFAULT_BINS)
    n_bins = reversed_tuple(n_bins)
    return {
        'fields': get_fields_with_schema(n_bins=n_bins, mol_ids=mol_ids, initial_min_max=initial_min_max),
        'spatial_dfba': get_spatial_many_dfba(model_file=dissolved_model_file, mol_ids=mol_ids, n_bins=n_bins)
    }

def get_single_dfba_doc(core=None, config=None):
    model_id = config.get('model_id', 'ecoli core')
    biomass_id = config.get('biomass_id', f'dissolved biomass')
    dfba_process = get_dfba_process_from_registry(
        model_id=model_id,
        biomass_id=biomass_id,
        path=['fields']
    )
    substrates = list(dfba_process['inputs']['substrates'].keys())
    initial_fields = config.get('initial_fields', {'glucose': 2, 'acetate': 0})
    if biomass_id not in initial_fields:
        initial_fields[biomass_id] = 0.1
    for substrate in substrates:
        if substrate not in initial_fields:
            initial_fields[substrate] = 10.0
    doc = {
        f'dFBA': dfba_process,
        'fields': initial_fields
    }
    return doc


DOC_BUILDERS = {
    "metacomposite": get_particle_multi_dfba_comets_doc,
    "particles_movement": get_particles_movement_doc,
    "fields_diffusion": get_fields_diffusion_doc,
    "spatial_dfba": get_spatial_dfba_doc,
    "single_dfba": get_single_dfba_doc,
}


# -------------
# plot settings
# -------------


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
    # # enumerate all dFBA nodes from n_bins
    # dFBA_nodes = [('spatial_dfba', f'dFBA[{i},{j}]')
    #               for i in range(n_bins[0]) for j in range(n_bins[1])]

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
        # **{node: COLORS["dfba_base"] for node in dFBA_nodes},
    }

    # unique particle + particle-owned states (guarded)
    if particle_id:
        fills.update({
            ('particles', particle_id):               COLORS["particles_base"],
            ('particles', particle_id, 'id'):         COLORS["particles_base"],
            ('particles', particle_id, 'position'):   COLORS["particles_base"],
            ('particles', particle_id, 'local'):      COLORS["local"],
            ('particles', particle_id, 'exchange'):   COLORS["exchange"],
            ('particles', particle_id, 'local', 'acetate'):  COLORS["local"],
            ('particles', particle_id, 'local', 'glucose'):  COLORS["local"],
            ('particles', particle_id, 'local', 'dissolved biomass'): COLORS["local"],
            ('particles', particle_id, 'exchange', 'acetate'): COLORS["exchange"],
            ('particles', particle_id, 'exchange', 'glucose'): COLORS["exchange"],
            ('particles', particle_id, 'exchange', 'dissolved biomass'): COLORS["exchange"],
            # particle-scoped dFBA
            ('particles', particle_id, 'dFBA'):       COLORS["dfba_base"],
            # mass/biomass bridge under particle
            ('particles', particle_id, 'mass'):       COLORS["particles_base"],
        })

    # ---- borders (auto = darker of fill; override here if needed) ----
    borders = {k: _darken(v) for k, v in fills.items()}

    # Return only the color maps (you can add dpi/show_types externally if desired)
    return {
        'node_fill_colors': fills,
        'node_border_colors': borders,
    }


def main():
    outdir = "out"
    n_bins = (10, 5)
    config = {'n_bins': n_bins}

    # shared core for all docs
    core = VivariumTypes()
    core = register_types(core)

    os.makedirs(outdir, exist_ok=True)

    for name, doc_fn in DOC_BUILDERS.items():
        print(f"\n=== Building document: {name} ===")

        # build document
        document = doc_fn(config=config)
        # ensure top-level has 'state'
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
            print(f"⚠ Could not save schema JSON for {name}: {e}")

        # Visualize initial composition
        plot_state = {
            k: v for k, v in sim.state.items()
            if k not in ['global_time', 'emitter']
        }
        plot_schema = {
            k: v for k, v in sim.composition.items()
            if k not in ['global_time', 'emitter']
        }

        # particle_id is optional (some docs have no particles)
        particle_id = None
        if 'particles' in plot_state and plot_state['particles']:
            particle_id = list(plot_state['particles'].keys())[0]

        # plot settings
        plot_settings = build_plot_settings(particle_id=particle_id, n_bins=n_bins)
        plot_settings.update(dict(
            dpi='300',
            show_values=True,
            show_types=True,
            collapse_redundant_processes={
                'exclude': [('particle_movement',), ('particle_division',)]
            },
            value_char_limit=20,
            type_char_limit=40,
            # label_margin='0.01',
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
