"""
Adapters for converting between different representations.
"""
import numpy as np
from process_bigraph import Step


def get_conc_count_adapter(
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
        'config': {},
        'inputs': {
            'exchanges': exchange_path,
            'volume': volume_path,
        },
        'outputs': {
            'concentrations': conc_path,
        },
    }


class CountConcAdapter(Step):
    config_schema = {
        'n_bins': 'tuple[integer{1},integer{1}]',
        'bin_volume': 'float{1}',
    }

    def initialize(self, config):
        pass


    def inputs(self):
        volume = self.config['bin_volume']
        return {
            'exchanges': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'float',
                }
            },
            'volume': 'float'
            # 'volume': f'float{{{volume}}}',
        }

    def outputs(self):
        return {
            'concentrations': 'map[array]',
        }


    def update(self, state):
        exchanges = state.get('exchanges', {}) or {}
        volume = state.get('volume', None)

        if volume is None:
            raise KeyError("CountConcAdapter requires 'volume' input")

        vol = float(volume)
        if vol <= 0.0:
            raise ValueError(f"CountConcAdapter requires volume > 0, got {vol}")

        concentrations = {}

        for mol_id, counts in exchanges.items():
            # ensure array semantics
            arr = np.asarray(counts, dtype=float)
            concentrations[mol_id] = arr / vol

        # empty update is still meaningful here
        return {
            'concentrations': concentrations
        }
