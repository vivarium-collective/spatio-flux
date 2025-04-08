"""
Diffusion-Advection Process
===========================

This process is a simple 2D diffusion-advection process. It takes a 2D field as input and returns a 2D field as output.
"""
import numpy as np
from scipy.ndimage import convolve
from process_bigraph import Process, Composite
from spatio_flux.viz.plot import plot_species_distributions_to_gif


# Laplacian for 2D diffusion
LAPLACIAN_2D = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])


class DiffusionAdvection(Process):
    config_schema = {
        'n_bins': 'tuple[integer,integer]',
        'bounds': 'tuple[float,float]',
        'default_diffusion_rate': {'_type': 'float', '_default': 1e-2},
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

            # Update time
            t += dt

        return updated_state - state


# Helper functions to get specs and states
def get_diffusion_advection_spec(
        bounds=(10.0, 10.0),
        n_bins=(5, 5),
        mol_ids=None,
        default_diffusion_rate=1e-2,
        default_advection_rate=(0, 0),
        diffusion_coeffs=None,
        advection_coeffs=None,
):
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']
    if diffusion_coeffs is None:
        diffusion_coeffs = {}
    if advection_coeffs is None:
        advection_coeffs = {}

    # fill in the missing diffusion and advection rates
    diffusion_coeffs_all = {
        mol_id: diffusion_coeffs.get(mol_id, default_diffusion_rate)
        for mol_id in mol_ids
    }
    advection_coeffs_all = {
        mol_id: advection_coeffs.get(mol_id, default_advection_rate)
        for mol_id in mol_ids
    }

    return {
            '_type': 'process',
            'address': 'local:DiffusionAdvection',
            'config': {
                'n_bins': n_bins,
                'bounds': bounds,
                'default_diffusion_rate': 1e-1,
                'default_diffusion_dt': 1e-1,
                'diffusion_coeffs': diffusion_coeffs_all,
                'advection_coeffs': advection_coeffs_all,
            },
            'inputs': {
                'fields': ['fields']
            },
            'outputs': {
                'fields': ['fields']
            }
        }


def get_diffusion_advection_state(
        bounds=(10.0, 10.0),
        n_bins=(5, 5),
        mol_ids=None,
        initial_max=None,
        default_diffusion_rate=1e-1,
        default_advection_rate=(0, 0),
        diffusion_coeffs=None,
        advection_coeffs=None,
):
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']
    if initial_max is None:
        initial_max = {
            'glucose': 20,
            'acetate': 0,
            'biomass': 0.1
        }
    initial_fields = {
        mol_id: np.random.uniform(low=0, high=initial_max[mol_id], size=n_bins)
        for mol_id in mol_ids}

    return {
        'fields': {
            '_type': 'map',
            '_value': {
                '_type': 'array',
                '_shape': n_bins,
                '_data': 'positive_float'
            },
            **initial_fields,
        },
        'diffusion': get_diffusion_advection_spec(
            bounds=bounds,
            n_bins=n_bins,
            mol_ids=mol_ids,
            default_diffusion_rate=default_diffusion_rate,
            default_advection_rate=default_advection_rate,
            diffusion_coeffs=diffusion_coeffs,
            advection_coeffs=advection_coeffs,
        ),
    }


