"""
Diffusion-Advection Process
===========================

Simulates 2D diffusion and advection on scalar fields using finite differences.

Each species field evolves over time according to:
- Diffusion (via a Laplacian kernel)
- Advection (via gradient approximation in x and y directions)
"""

import numpy as np
from scipy.ndimage import convolve
from process_bigraph import Process

# Kernels
LAPLACIAN_2D = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])

KERNEL_DX = np.array([[-1, 0, 1]])  # central difference in x
KERNEL_DY = np.array([[-1], [0], [1]])  # central difference in y


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

        # Grid geometry
        nx, ny = self.config['n_bins']
        xmax, ymax = self.config['bounds']
        dx = xmax / nx
        dy = ymax / ny
        cell_area = dx * dy

        # Default diffusion rate, normalized
        D = self.config['default_diffusion_rate']
        self.default_D = D / cell_area

        # Per-species diffusion rates
        self.species_D = {
            mol_id: d / cell_area for mol_id, d in self.config['diffusion_coeffs'].items()
        }

        # Stability condition: max dt for stable explicit update
        stable_dt = 0.5 * dx ** 2 * dy ** 2 / (2 * D * (dx ** 2 + dy ** 2))
        self.diffusion_dt = min(stable_dt, self.config['default_diffusion_dt'])

    def inputs(self):
        return {
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'concentration'
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
                    '_data': 'concentration'
                },
            }
        }

    def update(self, state, interval):
        updated = {}

        for species, field in state['fields'].items():
            D = self.species_D.get(species, self.default_D)
            vx, vy = self.config['advection_coeffs'].get(species, (0.0, 0.0))
            updated[species] = self.diffuse_advect(field, interval, D, vx, vy)

        return {'fields': updated}

    def diffuse_advect(self, field, interval, diffusion_coeff, vx, vy):
        """
        Evolve a 2D field over time using explicit Euler steps of diffusion and advection.
        """
        state = field.copy()
        t = 0.0
        dt = min(interval, self.diffusion_dt)

        while t < interval:
            # Diffusion term (Laplacian)
            diffusion = convolve(state, LAPLACIAN_2D, mode='reflect') * diffusion_coeff

            # Advection terms (central differences)
            adv_x = convolve(state, KERNEL_DX, mode='reflect') * vx
            adv_y = convolve(state, KERNEL_DY, mode='reflect') * vy

            # Euler update
            state += (diffusion + adv_x + adv_y) * dt
            t += dt

        return state - field  # Return delta for this interval