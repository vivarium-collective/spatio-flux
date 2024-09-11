"""
COMETS composite made of dFBAs and diffusion-advection processes.
"""

from process_bigraph import Composite
from processes import core
from viz.plot import plot_time_series, plot_species_distributions_to_gif


# TODO -- need to do this to register???
from processes.dfba import DynamicFBA, get_spatial_dfba_state
from processes.diffusion_advection import DiffusionAdvection, get_diffusion_advection_spec


def run_comets(
        total_time=60.0,
        bounds=(10.0, 10.0),
        n_bins=(10, 10),
        mol_ids=None,
):
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']

    # make the composite state
    composite_state = get_spatial_dfba_state(
        n_bins=n_bins,
        mol_ids=mol_ids,
        initial_max={
            'glucose': 20,
            'acetate': 0,
            'biomass': 0.1
        }
    )
    composite_state['diffusion'] = get_diffusion_advection_spec(
        bounds=bounds,
        n_bins=n_bins,
        mol_ids=mol_ids,
        default_diffusion_rate=1e-1,
        default_advection_rate=(0, 0),
        diffusion_coeffs=None,
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

    # TODO -- save a viz figure of the initial state

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
    run_comets()
