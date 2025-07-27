
import numpy as np
from spatio_flux import SpatioFluxVivarium
from spatio_flux.processes import get_minimal_particle_composition


def run_vivarium_particles():
    bounds = (10.0, 20.0)  # Bounds of the environment
    n_bins = (20, 40)  # Number of bins in the x and y directions

    v6 = SpatioFluxVivarium()

    # make two fields
    v6.add_object(name='glucose',
                  path=['fields'],
                  value=np.ones((n_bins[0], n_bins[1])))
    v6.add_object(
        name='acetate',
        path=['fields'],
        value=np.zeros((n_bins[0], n_bins[1])))

    v6.add_process(
        name='particle_movement',
        process_id='Particles',
        config={
            'n_bins': n_bins,
            'bounds': bounds,
            'diffusion_rate': 0.1,
            'advection_rate': (0, -0.1),
            'add_probability': 0.3,
            'boundary_to_add': ['top']})
    v6.connect_process(
        name='particle_movement',
        inputs={
            'fields': ['fields'],
            'particles': ['particles']},
        outputs={
            'fields': ['fields'],
            'particles': ['particles']})

    # add a process into each particles schema
    minimal_particle_config = {
        'reactions': {
            'grow': {
                'glucose': {
                    'vmax': 0.01,
                    'kcat': 0.01,
                    'role': 'reactant'},
                'acetate': {
                    'vmax': 0.001,
                    'kcat': 0.001,
                    'role': 'product'}
            }}}
    particle_schema = get_minimal_particle_composition(v6.core, minimal_particle_config)
    v6.merge_schema(path=['particles'], schema=particle_schema['particles'])

    # add particles to the initial state
    v6.initialize_process(
        path='particle_movement',
        config={'n_particles': 1})

    v6.diagram(dpi='70')

    v6.run(60)
    v6_results = v6.get_results()

    v6.plot_particles_snapshots(skip_frames=3)


if __name__ == '__main__':
    run_vivarium_particles()
