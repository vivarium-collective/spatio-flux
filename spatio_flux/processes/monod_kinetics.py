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


def get_monod_kinetics_process_from_config(
    *,
    model_config,
    fields_key='fields',
    biomass_id='biomass',
):
    """
    Build a MonodKinetics process spec by inspecting the kinetic model config.
    """
    substrates, has_biomass = extract_kinetic_fields(model_config)

    substrate_wires = {
        s: [fields_key, s]
        for s in sorted(substrates)
    }

    inputs = {'substrates': substrate_wires}
    outputs = {'substrates': substrate_wires}

    if has_biomass:
        inputs['biomass'] = [fields_key, biomass_id]
        outputs['biomass'] = [fields_key, biomass_id]

    return {
        '_type': 'process',
        'address': 'local:MonodKinetics',
        'config': model_config,
        'inputs': inputs,
        'outputs': outputs,
    }


def extract_kinetic_fields(kinetic_model_config):
    """
    Extract substrate and biomass field names from a kinetic model config.
    """
    reactions = kinetic_model_config.get('reactions', {})

    substrates = set()
    has_biomass = False

    for rxn in reactions.values():
        reactant = rxn.get('reactant')
        product = rxn.get('product')

        for species in (reactant, product):
            if species is None:
                continue
            if species == 'mass':
                has_biomass = True
            else:
                substrates.add(species)

    return substrates, has_biomass


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
                'vmax': 0.03,
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


# -------------------------
# Overflow metabolism
# -------------------------

def get_overflow_metabolism_kinetics_config():
    return {
        'reactions': {
            'assimilate_glucose': {
                'reactant': 'glucose',
                'product': 'mass',
                'km': 0.5,
                'vmax': 0.4,
                'yield': 0.2,
            },
            'overflow_to_acetate': {
                'reactant': 'glucose',
                'product': 'acetate',
                'km': 0.5,
                'vmax': 0.3,
                'yield': 0.5,
            },
            'assimilate_acetate': {
                'reactant': 'acetate',
                'product': 'mass',
                'km': 0.6,
                'vmax': 0.3,
                'yield': 0.2,
            },
        }
    }


# -------------------------
# Glucose-only and Acetate-only
# -------------------------

def get_glucose_only_kinetics_config(*, include_maintenance=False):
    """
    Glucose -> mass only.
    Uses overflow_metabolism's assimilate_glucose parameters.
    """
    reactions = {
        'assimilate_glucose': {
            'reactant': 'glucose',
            'product': 'mass',
            'km': 0.5,
            'vmax': 0.4,
            'yield': 0.2,
        },
    }

    if include_maintenance:
        # Choose your preferred byproduct here; detritus is a safe sink.
        reactions['maintenance_turnover'] = {
            'reactant': 'mass',
            'product': 'detritus',
            'km': 1.0,
            'vmax': 0.001,
            'yield': 1.0,
        }

    return {'reactions': reactions}


def get_acetate_only_kinetics_config(*, include_maintenance=False):
    """
    Acetate -> mass only.
    Uses overflow_metabolism's assimilate_acetate parameters.
    """
    reactions = {
        'assimilate_acetate': {
            'reactant': 'acetate',
            'product': 'mass',
            'km': 0.5,
            'vmax': 0.4,
            'yield': 1.0,
        },
    }

    if include_maintenance:
        reactions['maintenance_turnover'] = {
            'reactant': 'mass',
            'product': 'detritus',
            'km': 1.0,
            'vmax': 0.001,
            'yield': 1.0,
        }

    return {'reactions': reactions}


# -------------------------
# Others
# -------------------------

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
    'glucose_only': get_glucose_only_kinetics_config,
    'acetate_only': get_acetate_only_kinetics_config,
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
            'biomass': 'count',
            'substrates': 'map[count]',
        }

    @staticmethod
    def _monod_rate(conc, km, vmax):
        conc = float(conc)
        km = float(km)
        vmax = float(vmax)
        denom = km + conc
        return (vmax * conc / denom) if denom > 0.0 else 0.0

    def update(self, state, interval):
        substrates = state.get('substrates', {})
        biomass = float(state.get('biomass', 0.0))
        # substrates_dict = state.get('substrates', {}) or {}
        # substrates = {sid: float(s.get('concentration', 0.0)) for sid, s in substrates_dict.items()}
        # biomass = float(state.get('biomass', {}).get('concentration', 0.0))
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
        # return {
        #     'biomass': {'count': delta_biomass},
        #     'substrates': {sid: {'count': v} for sid, v in delta_substrates.items()},
        # }


def get_kinetics_single_doc(
        core=None,
        config=None,
):
    config = config or {}
    model_id = config.get('model_id', 'overflow_metabolism')
    model_config = MODEL_REGISTRY_KINETICS[model_id]()
    doc = {
        'monod_kinetics': {
            '_type': 'process',
            'address': 'local:MonodKinetics',
            'config': model_config,
            'inputs': {
                'substrates': {
                    'glucose': ['fields', 'glucose', 'concentration'],
                    'acetate': ['fields', 'acetate', 'concentration'],
                },
                'biomass': ['fields', 'monod_biomass', 'concentration'],
            },
            'outputs': {
                'substrates': {
                    'glucose': ['fields', 'glucose', 'count'],
                    'acetate': ['fields', 'acetate', 'count'],
                },
                'biomass': ['fields', 'monod_biomass', 'count'],
            },
        },
        'fields': {
            '_type': 'map[conc_count]',
            'glucose': {'concentration': 10},
            'acetate': {'concentration': 0},
            'monod_biomass': {'concentration': 0.1}
        }
    }
    return {'state': doc}

if __name__ == '__main__':
    from process_bigraph import allocate_core, Composite
    core = allocate_core()
    doc = get_kinetics_single_doc()
    sim = Composite(doc, core=core)
    sim.run(interval=10)

    import ipdb; ipdb.set_trace()
