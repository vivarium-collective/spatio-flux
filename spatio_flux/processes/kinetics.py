"""
Monod Kinetics Process (reaction-level params + yield + biomass-proportional uptake)

Scheme:
- Only `reactions` in config.
- Each reaction carries its own Monod params: km, vmax.
- All reactions use Monod/MM form: rate = vmax * conc / (km + conc)
- `vmax` is interpreted as per-unit-time, so we apply `flux = rate * interval`.
- Each reaction may optionally include `yield` (default 1.0), which scales PRODUCT formation
  relative to reactant consumption.
- Biomass-proportional uptake/growth:
    For reactions whose reactant is NOT 'mass', we multiply flux by current biomass:
        flux *= biomass
    This makes uptake / production capacity scale with biomass.

Notes on yield:
- Reactant consumption is always `-flux`.
- Product formation is `+flux * yield`.
- This does not enforce mass/element conservation; it’s a simple knob for biomass yield.
"""

import numpy as np
from process_bigraph import Process
from spatio_flux.library.tools import build_path


# ---------------------------------------------------------------------
# Model configs (registry)
# ---------------------------------------------------------------------

def get_single_substrate_assimilation_kinetics_config():
    return {
        'reactions': {
            'assimilate_glucose': {
                'reactant': 'glucose',
                'product': 'mass',
                'km': 0.5,
                'vmax': 0.03,  # higher vmax = faster growth potential
                'yield': 1.0,
            },
            'maintenance_turnover': {
                'reactant': 'mass',
                'product': 'acetate',
                'km': 1.0,
                'vmax': 0.001,
                'yield': 1.0,
            },
        }
    }


def get_two_substrate_assimilation_kinetics_config():
    return {
        'reactions': {
            'assimilate_glucose': {
                'reactant': 'glucose',
                'product': 'mass',
                'km': 0.3,
                'vmax': 0.02,
                'yield': 1.0,
            },
            'assimilate_acetate': {
                'reactant': 'acetate',
                'product': 'mass',
                'km': 0.2,
                'vmax': 0.015,
                'yield': 1.0,
            },
            'maintenance_turnover': {
                'reactant': 'mass',
                'product': 'detritus',
                'km': 1.0,
                'vmax': 0.001,
                'yield': 1.0,
            },
        }
    }


def get_overflow_metabolism_kinetics_config():
    # KEEPING YOUR CURRENT PARAMETERS EXACTLY (km/vmax/yield)
    return {
        'reactions': {
            # direct growth on glucose
            'assimilate_glucose': {
                'reactant': 'glucose',
                'product': 'mass',
                'km': 0.5,
                'vmax': 0.4,
                'yield': 0.2,
            },
            # overflow channel (glucose -> acetate)
            'overflow_to_acetate': {
                'reactant': 'glucose',
                'product': 'acetate',
                'km': 0.5,
                'vmax': 0.4,
                'yield': 1.0,
            },
            # growth on acetate
            'assimilate_acetate': {
                'reactant': 'acetate',
                'product': 'mass',
                'km': 0.5,
                'vmax': 0.2,
                'yield': 0.2,
            },
            # maintenance
            'maintenance_turnover': {
                'reactant': 'mass',
                'product': 'detritus',
                'km': 1.0,
                'vmax': 0.01,
                'yield': 1.0,
            },
        }
    }


def get_cross_feeding_kinetics_config():
    return {
        'reactions': {
            'convert_lactate_to_propionate': {
                'reactant': 'lactate',
                'product': 'propionate',
                'km': 0.4,
                'vmax': 0.02,
                'yield': 1.0,
            },
            'assimilate_lactate': {
                'reactant': 'lactate',
                'product': 'mass',
                'km': 0.4,
                'vmax': 0.02,
                'yield': 1.0,
            },
            'maintenance_turnover': {
                'reactant': 'mass',
                'product': 'detritus',
                'km': 1.0,
                'vmax': 0.001,
                'yield': 1.0,
            },
        }
    }


def get_autotrophic_kinetics_config():
    return {
        'reactions': {
            'assimilate_ammonium': {
                'reactant': 'ammonium',
                'product': 'mass',
                'km': 0.2,
                'vmax': 0.01,
                'yield': 1.0,
            },
            'maintenance_turnover': {
                'reactant': 'mass',
                'product': 'detritus',
                'km': 1.0,
                'vmax': 0.001,
                'yield': 1.0,
            },
        }
    }


MODEL_REGISTRY_KINETICS = {
    'single_substrate_assimilation': get_single_substrate_assimilation_kinetics_config,
    'two_substrate_assimilation': get_two_substrate_assimilation_kinetics_config,
    'overflow_metabolism': get_overflow_metabolism_kinetics_config,
    'cross_feeding': get_cross_feeding_kinetics_config,
    'autotrophic': get_autotrophic_kinetics_config,
}


def get_kinetics_process_from_registry(
    model_id,
    path,
    mol_ids=None,
    biomass_id=None,
    i=None,
    j=None,
):
    model_config = MODEL_REGISTRY_KINETICS[model_id]()
    biomass_id = biomass_id or 'biomass'

    if mol_ids is not None:
        mol_ids = [m for m in mol_ids if m != biomass_id]

    return {
        "_type": "process",
        "address": "local:MonodKinetics",
        "config": model_config,
        "inputs": {
            "substrates": {mol_id: build_path(path, mol_id, j, i) for mol_id in mol_ids},  # note j,i order for rows,cols
            "biomass": build_path(path, biomass_id, j, i),
        },
        "outputs": {
            "substrates": {mol_id: build_path(path, mol_id, j, i) for mol_id in mol_ids},
            "biomass": build_path(path, biomass_id, j, i),
        }
    }


# ---------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------

class MonodKinetics(Process):
    """
    Monod/MM kinetics with reaction-level params + yield + biomass-proportional uptake.

    Each reaction is:
      {reactant: str, product: str, km: float, vmax: float, yield?: float}

    Biomass-proportional rule:
      - If reactant != 'mass' (i.e., a substrate-driven reaction), treat the Monod rate as
        a specific rate and multiply by current biomass:
            flux = rate * biomass * interval
      - If reactant == 'mass' (maintenance/turnover), do NOT multiply by biomass again
        because the "concentration term" is already biomass.

    Output semantics:
      - Returns deltas (amounts over the interval)
      - Reactant is consumed by `flux`
      - Product is produced by `flux * yield`
    """

    config_schema = {
        'reactions': {
            '_type': 'map[reaction]',
            '_default': {
                'assimilate_glucose': {
                    'reactant': 'glucose',
                    'product': 'mass',
                    'km': 0.5,
                    'vmax': 0.01,
                    'yield': 1.0,
                },
                'maintenance_turnover': {
                    'reactant': 'mass',
                    'product': 'detritus',
                    'km': 1.0,
                    'vmax': 0.001,
                    'yield': 1.0,
                },
            }
        }
    }

    def initialize(self, config):
        self.reactions = config.get('reactions', {}) or {}

    def inputs(self):
        return {
            'biomass': 'concentration',
            'substrates': 'map[concentration]',
        }

    def outputs(self):
        return {
            'biomass': 'concentration',
            'substrates': 'map[concentration]',
        }

    @staticmethod
    def _monod_rate(conc, km, vmax):
        conc = float(conc)
        km = float(km)
        vmax = float(vmax)
        denom = km + conc
        return (vmax * conc / denom) if denom > 0.0 else 0.0

    def update(self, state, interval):
        substrates = state.get('substrates', {}) or {}
        biomass = float(state.get('biomass', 0.0))
        dt = float(interval)

        delta_biomass = 0.0
        delta_substrates = {}

        for rxn_name, rxn in self.reactions.items():
            reactant = rxn.get('reactant')
            product = rxn.get('product')

            if reactant is None or product is None:
                raise ValueError(f"Reaction '{rxn_name}' must define 'reactant' and 'product'.")

            if 'km' not in rxn or 'vmax' not in rxn:
                raise ValueError(f"Reaction '{rxn_name}' must define 'km' and 'vmax'.")

            km = rxn['km']
            vmax = rxn['vmax']
            yld = float(rxn.get('yield', 1.0))

            # concentration term for Monod/MM
            conc = biomass if reactant == 'mass' else float(substrates.get(reactant, 0.0))

            # rate is per time; interpret vmax as a specific rate for substrate-driven reactions
            rate = self._monod_rate(conc, km, vmax)

            # biomass-proportional uptake/growth for substrate-driven reactions
            if reactant != 'mass':
                flux = rate * biomass * dt
            else:
                # turnover already depends on biomass via conc=biomass
                flux = rate * dt

            # consume reactant (unscaled)
            if reactant == 'mass':
                delta_biomass -= flux
            else:
                delta_substrates[reactant] = delta_substrates.get(reactant, 0.0) - flux

            # produce product (scaled by yield)
            prod_flux = flux * yld
            if product == 'mass':
                delta_biomass += prod_flux
            else:
                delta_substrates[product] = delta_substrates.get(product, 0.0) + prod_flux

        return {
            'biomass': delta_biomass,
            'substrates': delta_substrates,
        }
