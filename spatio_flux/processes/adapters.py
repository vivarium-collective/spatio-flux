"""
Adapters for converting between different representations.
"""
import numpy as np
from process_bigraph import Step


def get_conc_count_adapter(
        n_bins,
        bounds,
        depth=1.0,
        conc_path=None,
        exchange_path=None,
        volume_path=None,
):
    if volume_path is None:
        volume_path = ['bin_volume']
    if exchange_path is None:
        exchange_path = ['exchanges']
    if conc_path is None:
        conc_path = ['fields']

    return {
        '_type': 'step',
        'address': 'local:CountConcAdapter',
        'config': {
            'n_bins': n_bins,
            'bounds': bounds,
            'depth': depth,
        },
        'inputs': {
            'exchanges': exchange_path,
            # 'volume': volume_path,
        },
        'outputs': {
            'concentrations': conc_path,
        },
    }


class CountConcAdapter(Step):
    """
    Convert per-bin counts into concentrations by dividing by bin volume.

    Config:
      - n_bins: (nx, ny)
      - bounds: (xmax, ymax)  (assumes xmin=ymin=0)
      - depth:  scalar thickness

    bin_volume = (xmax/nx) * (ymax/ny) * depth
    """

    config_schema = {
        'n_bins': 'tuple[integer{1},integer{1}]',
        'bounds': 'tuple[float{1.0},float{1.0}]',  # (xmax, ymax)
        'depth':  'float{1}',
    }

    def initialize(self, config):
        nx, ny = tuple(self.config['n_bins'])

        bounds = self.config.get('bounds', None)
        if bounds is None:
            raise KeyError("CountConcAdapter requires config['bounds'] = (xmax, ymax)")
        if len(bounds) != 2:
            raise ValueError("CountConcAdapter config['bounds'] must be a 2-tuple: (xmax, ymax)")

        xmax, ymax = bounds
        xmax = float(xmax)
        ymax = float(ymax)

        depth = float(self.config.get('depth', None))
        if depth <= 0.0:
            raise ValueError(f"CountConcAdapter requires depth > 0, got {depth}")

        if nx <= 0 or ny <= 0:
            raise ValueError(f"CountConcAdapter requires n_bins > 0, got {(nx, ny)}")

        if xmax <= 0.0 or ymax <= 0.0:
            raise ValueError(f"CountConcAdapter requires bounds > 0, got (xmax={xmax}, ymax={ymax})")

        dx = xmax / float(nx)
        dy = ymax / float(ny)

        self.bin_volume = dx * dy * depth
        if self.bin_volume <= 0.0:
            raise ValueError(f"Computed bin_volume must be > 0, got {self.bin_volume}")

    def inputs(self):
        return {
            'exchanges': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'float',
                }
            }
        }

    def outputs(self):
        return {
            'concentrations': 'map[array]',
        }

    def update(self, state):
        exchanges = state.get('exchanges', {}) or {}

        vol = float(getattr(self, "bin_volume", 0.0))
        if vol <= 0.0:
            raise ValueError(f"CountConcAdapter internal bin_volume invalid: {vol}")

        concentrations = {}
        for mol_id, counts in exchanges.items():
            arr = np.asarray(counts, dtype=float)
            concentrations[mol_id] = arr / vol

        return {'concentrations': concentrations}

