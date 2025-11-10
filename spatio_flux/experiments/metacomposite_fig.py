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


def main():
    name = "metacomposite"
    outdir = "out"

    document = get_particle_multi_dfba_comets_doc(config={'n_bins': (2, 2)})
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

    plot_bigraph(
        state=plot_state,
        schema=plot_schema,
        core=core,
        out_dir=outdir,
        filename=f"{name}_viz",
        dpi="300",
        show_types=True,
        # collapse_redundant_processes=True,
    )



if __name__ == "__main__":
    main()


