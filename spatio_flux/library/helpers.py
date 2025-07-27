from datetime import datetime

import numpy as np
from bigraph_viz import plot_bigraph
from process_bigraph import Composite, gather_emitter_results
from process_bigraph.emitter import emitter_from_wires
from vivarium.vivarium import VivariumTypes


def initialize_fields(n_bins, initial_min_max=None):
    initial_min_max = initial_min_max or {}
    fields = {}
    for field, minmax in initial_min_max.items():
        fields[field] = np.random.uniform(low=minmax[0], high=minmax[1], size=n_bins)
    return fields


def get_standard_emitter():
    """
    Returns a standard emitter specification for capturing global time and fields.
    """
    return emitter_from_wires({
        'global_time': ['global_time'],
        'fields': ['fields'],
        'particles': ['particles'],
    })


def run_composite_document(document, time=None, core=None, name=None):
    """
    Instantiates and runs a Composite simulation.

    Args:
        document (dict): Composition document with initial state and optional schema.
        time (float): Simulation duration.
        core (VivariumTypes): Core schema registration object.
        name (str): Output name prefix.

    Returns:
        dict: Simulation results emitted during the run.
    """
    time = time or 60
    if core is None:
        from spatio_flux import register_types
        core = VivariumTypes()
        core = register_types(core)
    if name is None:
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'spatio_flux_{date}'

    document = {'state': document} if 'state' not in document else document
    if 'emitter' not in document['state']:
        document['state']['emitter'] = get_standard_emitter()

    print('Making the composite...')
    sim = Composite(document, core=core)

    # Save composition JSON
    sim.save(filename=f'{name}.json', outdir='out')

    # Save visualization of the initial composition
    plot_bigraph(
        state=sim.state,
        schema=sim.composition,
        core=core,
        out_dir='out',
        filename=f'{name}_viz'
    )

    print('Simulating...')
    sim.run(time)
    results = gather_emitter_results(sim)
    return results[('emitter',)]
