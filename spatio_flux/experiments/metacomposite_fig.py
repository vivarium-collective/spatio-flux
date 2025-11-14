import os
import json

from spatio_flux.experiments.test_suite import DEFAULT_BOUNDS, DEFAULT_BINS, DEFAULT_ADVECTION
from spatio_flux.library.colors import build_plot_settings
from spatio_flux.library.helpers import inverse_tuple, reversed_tuple
from spatio_flux import register_types
from vivarium.vivarium import VivariumTypes, Composite
from bigraph_viz import plot_bigraph

from spatio_flux.processes import DIVISION_MASS_THRESHOLD, get_fields, get_diffusion_advection_process, \
    get_spatial_many_dfba, get_particles_state, get_particle_movement_process, get_particle_exchange_process, \
    get_particle_divide_process, get_dfba_particle_composition, get_fields_with_schema, get_dfba_process_from_registry


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
    "metacomposite": {
        'doc_func': get_particle_multi_dfba_comets_doc,
        'plot_func': None,
    },
    "particles_movement": {
        'doc_func': get_particles_movement_doc,
        'plot_func': None,
    },
    "fields_diffusion": {
        'doc_func': get_fields_diffusion_doc,
        'plot_func': None,
    },
    "spatial_dfba": {
        'doc_func': get_spatial_dfba_doc,
        'plot_func': None,
    },
    "single_dfba": {
        'doc_func': get_single_dfba_doc,
        'plot_func': None,
    }
}


def main():
    outdir = "out"
    n_bins = (10, 5)
    config = {'n_bins': n_bins}

    # shared core for all docs
    core = VivariumTypes()
    core = register_types(core)

    os.makedirs(outdir, exist_ok=True)

    for name, doc_spec in DOC_BUILDERS.items():
        print(f"\n=== Building document: {name} ===")
        doc_fn = doc_spec['doc_func']

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
        plot_settings = build_plot_settings(particle_ids=particle_id, n_bins=n_bins)
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
