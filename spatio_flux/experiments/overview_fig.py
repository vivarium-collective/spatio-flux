from process_bigraph import register_types as register_process_types
from vivarium.vivarium import VivariumTypes
from bigraph_viz import plot_bigraph
from spatio_flux import register_types, TYPES_DICT


# ---- Single-process docs (one node per plot) ------------------------

def get_dfba_single_doc(core=None, config=None):
    # Minimal DynamicFBA process (no wiring needed for a figure)
    return {
        'dfba': {
            "_type": "process",
            "address": "local:DynamicFBA",
            "config": {"model_file": "textbook"},
        }
    }


def get_spatial_dfba_doc(core=None, config=None):
    # Minimal SpatialDFBA process — only n_bins is required for the figure
    return {
        'spatial_dfba': {
            "_type": "process",
            "address": "local:SpatialDFBA",
            "config": {
                "n_bins": (5, 10)
                # no models/default model needed just to render a single-process node
            },
        }
    }


def get_diffusion_advection_doc(core=None, config=None):
    # Minimal DiffusionAdvection process — supply required config keys
    return {
        'diffusion_advection': {
            "_type": "process",
            "address": "local:DiffusionAdvection",
            "config": {
                "n_bins": (5, 10),
                "bounds": (5.0, 10.0),
                # optional/defaulted keys:
                "default_diffusion_rate": 1e-1,
                "default_diffusion_dt": 1e-1,
                # leave explicit per-species maps empty for a pure process node diagram
                "diffusion_coeffs": {},
                "advection_coeffs": {},
            },
        }
    }


def get_particles_doc(core=None, config=None):
    # Minimal Particles process — add_probability has no default, set it explicitly
    return {
        'particles': {
            "_type": "process",
            "address": "local:Particles",
            "config": {
                "n_bins": (5, 10),
                "bounds": (5.0, 10.0),
                "diffusion_rate": 1e-1,
                "advection_rate": (0.0, -0.1),
                "add_probability": 0.0,         # required (no default in schema)
                # other keys have defaults in the process schema
            },
        }
    }


def get_minimal_kinetic_doc(core=None, config=None):
    # MinimalParticle process (renamed “minimal_kinetic” for the filename)
    # Uses schema defaults for reactions/kinetics unless you wish to override.
    return {
        'minimal_kinetic': {
            "_type": "process",
            "address": "local:MinimalParticle",
            "config": {
                # leave empty to use defaults, or customize:
                # 'reactions': {'grow': {...}, 'release': {...}},
                # 'kinetic_params': {'glucose': (0.5, 0.01), 'mass': (1.0, 0.001)},
            },
        }
    }


PROCESS_DOCS = {
    'dfba_single': get_dfba_single_doc,
    'spatial_dfba_single': get_spatial_dfba_doc,
    'diffusion_advection_single': get_diffusion_advection_doc,
    'particles_single': get_particles_doc,
    'minimal_kinetic_single': get_minimal_kinetic_doc,
}


def main():
    outdir = 'out'
    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    # plot the processes
    for name, get_doc in PROCESS_DOCS.items():
        document = get_doc(core=core)

        plot_bigraph(
            state=document,
            # schema=plot_schema,
            core=core,
            out_dir=outdir,
            filename=f'{name}_process',
            dpi='300',
            collapse_redundant_processes=True
        )

    # plot the types
    for type_name, type_class in TYPES_DICT.items():
        plot_bigraph(
            state={type_name: type_class},
            # schema=,
            show_types=True,
            core=core,
            out_dir=outdir,
            filename=f'{type_name}_type',
            dpi='300',
            collapse_redundant_processes=True
        )



if __name__ == '__main__':
    main()
