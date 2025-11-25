"""
Monod Kinetics Process
"""

from process_bigraph import Process
from spatio_flux.library.tools import build_path


def get_single_substrate_assimilation_kinetics_config():
    return {
        'reactions': {
            'assimilate_glucose': {'reactant': 'glucose', 'product': 'mass'},
            'maintenance_turnover': {'reactant': 'mass', 'product': 'acetate'},
        },
        'kinetic_params': {
            'glucose': (0.5, 0.03),  # higher Vmax = faster growth potential
            'mass': (1.0, 0.001),
        }
    }

def get_two_substrate_assimilation_kinetics_config():
    return {
        'reactions': {
            'assimilate_glucose': {'reactant': 'glucose', 'product': 'mass'},
            'assimilate_acetate': {'reactant': 'acetate', 'product': 'mass'},
            'maintenance_turnover': {'reactant': 'mass', 'product': 'detritus'},
        },
        'kinetic_params': {
            'glucose': (0.3, 0.02),
            'acetate': (0.2, 0.015),
            'mass': (1.0, 0.001),
        }
    }

def get_overflow_metabolism_kinetics_config():
    return {
        'reactions': {
        'overflow_to_acetate': {'reactant': 'glucose', 'product': 'acetate'},
        'assimilate_acetate':  {'reactant': 'acetate', 'product': 'mass'},
        'maintenance_turnover': {'reactant': 'mass', 'product': 'detritus'},
        },
        'kinetic_params': {
        'glucose': (0.5, 0.02),   # overflow production scales with biomass
        'acetate': (0.3, 0.015),  # growth on acetate
        'mass':    (1.0, 0.001),
        }
    }

def get_cross_feeding_kinetics_config():
    return {
        'reactions': {
        'convert_lactate_to_propionate': {'reactant': 'lactate', 'product': 'propionate'},
        'assimilate_lactate':            {'reactant': 'lactate', 'product': 'mass'},
        'maintenance_turnover':          {'reactant': 'mass', 'product': 'detritus'},
        },
        'kinetic_params': {
        'lactate': (0.4, 0.02),
        'mass':    (1.0, 0.001),
        }
    }

def get_autotrophic_kinetics_config():
    return {
        'reactions': {
        'assimilate_ammonium': {'reactant': 'ammonium', 'product': 'mass'},
        'maintenance_turnover': {'reactant': 'mass', 'product': 'detritus'},
        },
        'kinetic_params': {
        'ammonium': (0.2, 0.01),
        'mass':     (1.0, 0.001),
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
    # mol_ids = model_config['substrate_update_reactions'].keys()
    biomass_id = biomass_id or 'biomass'
    if mol_ids is not None:
        mol_ids = [m for m in mol_ids if m != biomass_id]

    return {
        "_type": "process",
        "address": "local:MonodKinetics",
        "config": model_config,
        "inputs": {
            "substrates": {mol_id: build_path(path, mol_id, i, j) for mol_id in mol_ids},
            "biomass": build_path(path, biomass_id, i, j)
        },
        "outputs": {
            "substrates": {mol_id: build_path(path, mol_id, i, j) for mol_id in mol_ids},
            "biomass": build_path(path, biomass_id, i, j)
        }
    }



class MonodKinetics(Process):
    """
    A minimal particle that performs reactions based on Michaelis-Menten kinetics.

    Configuration:
    -------------
    - reactions (dict): {reaction_name: {'reactant': str, 'product': str}}
    - kinetic_params (dict): {reactant: (Km, Vmax)} for each substrate or 'mass'

    Inputs:
    -------
    - mass (float): current mass of the particle
    - substrates (map[concentration]): concentrations of external substrates

    Outputs:
    --------
    - mass (delta): net change in mass
    - substrates (map[counts]): net change in substrate concentrations

    Notes:
    ------
    - Supports reactions that use 'mass' as reactant or product (e.g., decay or growth).
    - Reaction rates follow: rate = Vmax * conc / (Km + conc)
    """

    config_schema = {
        'reactions': {
            '_type': 'map[reaction]',
            '_default': {
                # substrate uptake that adds to biomass
                'assimilate_glucose': {'reactant': 'glucose', 'product': 'mass'},
                # biomass turnover/maintenance loss to detritus
                'maintenance_turnover': {'reactant': 'mass', 'product': 'detritus'},
            }
        },
        'kinetic_params': {
            '_type': 'map[tuple[float,float]]',
            '_default': {
                'glucose': (0.5, 0.01),  # Km, Vmax
                'mass': (1.0, 0.001)  # Km, Vmax for mass-based turnover
            }
        }
    }

    def initialize(self, config):
        self.reactions = config['reactions']
        self.kinetic_params = config['kinetic_params']

    def inputs(self):
        return {
            'biomass': 'concentration',
            'substrates': 'map[concentration]'
        }

    def outputs(self):
        return {
            'biomass': 'counts',
            'substrates': 'map[counts]'
        }

    def update(self, state, interval):
        substrates = state['substrates']
        mass = state['biomass']

        delta_mass = 0.0
        delta_substrates = {mol_id: 0.0 for mol_id in substrates}

        for reaction in self.reactions.values():
            reactant = reaction['reactant']
            product = reaction['product']

            conc = mass if reactant == 'mass' else substrates.get(reactant, 0.0)

            if reactant not in self.kinetic_params:
                raise ValueError(f"Kinetic parameters not provided for reactant: {reactant}")

            Km, Vmax = self.kinetic_params[reactant]
            rate = Vmax * conc / (Km + conc) if Km + conc > 0 else 0.0

            # update mass
            if reactant == 'mass':
                delta_mass -= rate
            else:
                if reactant not in delta_substrates:
                    delta_substrates[reactant] = 0.0
                delta_substrates[reactant] -= rate

            if product == 'mass':
                delta_mass += rate
            else:
                delta_substrates[product] = delta_substrates.get(product, 0.0) + rate

        return {
            'biomass': delta_mass,
            'substrates': delta_substrates
        }
