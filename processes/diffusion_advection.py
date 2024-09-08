import numpy as np
from scipy.ndimage import convolve
from process_bigraph import Process, Composite
from processes import core  # import the core from the processes package
from plot.plot import plot_species_distributions_to_gif


# Laplacian for 2D diffusion
LAPLACIAN_2D = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])


class DiffusionAdvection(Process):
    config_schema = {
        'n_bins': 'tuple[integer,integer]',
        'bounds': 'tuple[float,float]',
        'default_diffusion_rate': {'_type': 'float', '_default': 1e-1},
        'default_diffusion_dt': {'_type': 'float', '_default': 1e-1},
        'diffusion_coeffs': 'map[float]',
        'advection_coeffs': 'map[tuple[float,float]]',
    }

    def __init__(self, config, core):
        super().__init__(config, core)

        # get diffusion rates
        bins_x = self.config['n_bins'][0]
        bins_y = self.config['n_bins'][1]
        length_x = self.config['bounds'][0]
        length_y = self.config['bounds'][1]
        dx = length_x / bins_x
        dy = length_y / bins_y
        dx2 = dx * dy

        # general diffusion rate
        diffusion_rate = self.config['default_diffusion_rate']
        self.diffusion_rate = diffusion_rate / dx2

        # diffusion rates for each individual molecules
        self.molecule_specific_diffusion = {
            mol_id: diff_rate / dx2
            for mol_id, diff_rate in self.config['diffusion_coeffs'].items()}

        # get diffusion timestep
        diffusion_dt = 0.5 * dx ** 2 * dy ** 2 / (2 * diffusion_rate * (dx ** 2 + dy ** 2))
        self.diffusion_dt = min(diffusion_dt, self.config['default_diffusion_dt'])

    def inputs(self):
        return {
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'positive_float'
                },
            }
        }

    def outputs(self):
        return {
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'positive_float'
                },
            }
        }

    def update(self, state, interval):
        fields = state['fields']

        fields_update = {}
        for species, field in fields.items():
            fields_update[species] = self.diffusion_delta(
                field,
                interval,
                diffusion_coeff=self.config['diffusion_coeffs'][species],
                advection_coeff=self.config['advection_coeffs'][species]
            )

        return {
            'fields': fields_update
        }

    def diffusion_delta(self, state, interval, diffusion_coeff, advection_coeff):
        t = 0.0
        dt = min(interval, self.diffusion_dt)
        updated_state = state.copy()

        while t < interval:

            # Diffusion
            laplacian = convolve(
                updated_state,
                LAPLACIAN_2D,
                mode='reflect',
            ) * diffusion_coeff

            # Advection
            advective_flux_x = convolve(
                updated_state,
                np.array([[-1, 0, 1]]),
                mode='reflect',
            ) * advection_coeff[0]
            advective_flux_y = convolve(
                updated_state,
                np.array([[-1], [0], [1]]),
                mode='reflect',
            ) * advection_coeff[1]

            # Update the current state
            updated_state += (laplacian + advective_flux_x + advective_flux_y) * dt

            # # Ensure non-negativity
            # current_states[species] = np.maximum(updated_state, 0)

            # Update time
            t += dt

        return updated_state - state

core.register_process('DiffusionAdvection', DiffusionAdvection)


def run_diffusion_process():
    n_bins = (10, 10)

    initial_glucose = np.random.uniform(low=0, high=20, size=n_bins)
    initial_acetate = np.random.uniform(low=0, high=0, size=n_bins)
    initial_biomass = np.random.uniform(low=0, high=0.1, size=n_bins)

    composite_state = {
        'fields': {
            'glucose': initial_glucose,
            'acetate': initial_acetate,
            'biomass': initial_biomass,
        },
        'diffusion': {
            '_type': 'process',
            'address': 'local:DiffusionAdvection',
            'config': {
                'n_bins': n_bins,
                'bounds': (10, 10),
                'default_diffusion_rate': 1e-1,
                'default_diffusion_dt': 1e-1,
                'diffusion_coeffs': {
                    'glucose': 1e-1,
                    'acetate': 1e-1,
                    'biomass': 1e-1,
                },
                'advection_coeffs': {
                    'glucose': (0, 0),
                    'acetate': (0, 0),
                    'biomass': (0, 0),
                },
            },
            'inputs': {
                'fields': ['fields']
            },
            'outputs': {
                'fields': ['fields']
            }
        }
    }

    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'},
    }, core=core)
    sim.update({}, 60.0)

    diffadv_results = sim.gather_results()
    # print(diffadv_results)

    # plot 2d video
    plot_species_distributions_to_gif(
        diffadv_results,
        out_dir='out',
        filename='diffadv_results.gif',
        title='',
        skip_frames=1
    )


if __name__ == '__main__':
    run_diffusion_process()

