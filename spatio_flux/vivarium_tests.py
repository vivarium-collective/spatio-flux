from vivarium import Vivarium
from spatio_flux import PROCESS_DICT, TYPES_DICT

def run_vivarium_dfba():
    bounds = (10.0, 20.0)  # Bounds of the environment
    n_bins = (20, 40)  # Number of bins in the x and y directions

    v5 = Vivarium(processes=PROCESS_DICT, types=TYPES_DICT)

    v5.add_object(
        name='glucose',
        path=['fields'],
        value=np.random.rand(n_bins[0], n_bins[1])
    )
    # v5.add_object(
    #     type='particle',  # TODO -- registered particle type?
    #     path=['particles']
    # )

    v5.add_process(
        name='particle_movement',
        process_id='Particles',
        config={
            'n_bins': n_bins,
            'bounds': bounds,
            'diffusion_rate': 0.1,
            'advection_rate': (0, -0.1),
            'add_probability': 0.4,
            'boundary_to_add': ['top']
        },
    )
    v5.connect_process(
        name='particle_movement',
        inputs={
            'fields': ['fields'],
            'particles': ['particles']
        },
        outputs={
            'fields': ['fields'],
            'particles': ['particles']
        }
    )

    v5.initialize_process(
        name='particle_movement',
        config={'n_particles': 20}
    )
    breakpoint()


if __name__ == '__main__':
    run_vivarium_dfba()
