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


def run_comets(
        total_time=10.0,
        bounds=(10.0, 10.0),
        n_bins=(10, 10),
        mol_ids=None,
        initial_min_max=None,
):
    mol_ids = mol_ids or default_config['mol_ids']
    initial_min_max = initial_min_max or default_config['initial_min_max']

    # make the composite state
    composite_state = get_spatial_dfba_state(
        n_bins=n_bins,
        mol_ids=mol_ids,
        initial_min_max=initial_min_max,
    )
    composite_state['diffusion'] = get_diffusion_advection_spec(
        bounds=bounds,
        n_bins=n_bins,
        mol_ids=mol_ids,
        default_diffusion_rate=1e-1,
        default_advection_rate=(0, 0),
        diffusion_coeffs=None,  # TODO add all these config options
        advection_coeffs=None,
    )

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'},
    }, core=core)

    # save the document
    sim.save(filename='comets.json', outdir='out', include_schema=True)

    # # save a viz figure of the initial state
    # plot_bigraph(
    #     state=sim.state,
    #     schema=sim.composition,
    #     core=core,
    #     out_dir='out',
    #     filename='comets_viz'
    # )

    # simulate
    print('Simulating...')
    sim.update({}, total_time)
    comets_results = sim.gather_results()
    # print(comets_results)

    print('Plotting results...')
    # plot timeseries
    plot_time_series(
        comets_results,
        coordinates=[(0, 0), (5, 5)],
        out_dir='out',
        filename='comets_timeseries.png',
    )

    # plot 2d video
    plot_species_distributions_to_gif(
        comets_results,
        out_dir='out',
        filename='comets_results.gif',
        title='',
        skip_frames=1)


if __name__ == '__main__':
    run_comets(**default_config)
