"""
COMETS composite made of dFBAs and diffusion-advection processes.
"""
from process_bigraph import Composite
from bigraph_viz import plot_bigraph
from spatio_flux import core
from spatio_flux.viz.plot import plot_time_series, plot_species_distributions_to_gif

# TODO -- need to do this to register???
from spatio_flux.processes.dfba import get_spatial_dfba_state
from spatio_flux.processes.diffusion_advection import get_diffusion_advection_spec

default_config = {
    'total_time': 60.0,
    # environment size
    'bounds': (10.0, 10.0),
    'n_bins': (10, 10),
    # set fields
    'mol_ids': ['glucose', 'acetate', 'biomass'],
    'initial_min_max': {'glucose': (0, 10), 'acetate': (0, 0), 'biomass': (0, 0.1)},
}

## TODO -- maybe we need to make specific composites
class COMETS(Composite):
    """
    This needs to declare what types of processes are in the composite.
    """
    config_schema = {
        'n_bins': 'tuple',
    }

    def __init__(self, config, core=None):
        # set up the document here
        state = {
            'dFBA': {
                'config': {
                    'n_bins': config['n_bins'],
                }
            },
            'diffusion': {
                'config': {
                    'something_else': config['n_bins'],
                }
            }
        }

        super().__init__(config, core=core)

    # TODO -- this could be in Process.
    def get_default(self):
        return self.core.default(self.config_schema)


if __name__ == '__main__':
    run_comets(**default_config)
