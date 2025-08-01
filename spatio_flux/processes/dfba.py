"""
Dynamic FBA simulation
======================

A pluggable dynamic Flux Balance Analysis (dFBA) process.
Performs time-stepped metabolic modeling by combining COBRApy-based
optimization with kinetic uptake constraints.
"""
import os
import warnings
import numpy as np
from copy import deepcopy
import cobra
from cobra.io import load_model
from process_bigraph import Process

# Suppress known benign warnings from COBRApy
warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings("ignore", category=FutureWarning, module="cobra.medium.boundary_types")

# Define static bounds for specific models (can customize per model if needed)
default_bounds = {}

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

core_exchange_fluxes = [
    'EX_glc__D_e',  # universal carbon
    'EX_o2_e',      # aerobic respiration
    'EX_nh4_e',     # nitrogen source
    'EX_pi_e',      # phosphate
    'EX_co2_e',     # universal byproduct
    'EX_h2o_e',     # universal byproduct
    'EX_h_e',       # proton gradient / acid-base balance
    'EX_ac_e',      # fermentative overflow
    'EX_etoh_e',    # ethanol (esp. for yeast)
]

default_kinetics = {
    'substrate_update_reactions': {
        "glucose": "EX_glc__D_e",
        "acetate": "EX_ac_e"
    },
    'kinetic_params': {
        "glucose": (0.5, 1),
        "acetate": (0.5, 2)
    }
}

MODEL_REGISTRY_DFBA = {
    'textbook': {
        'filename': 'textbook',
        'bounds': {
            "EX_o2_e": {"lower": -2, "upper": None},
            "ATPM": {"lower": 1, "upper": 1}},
        'config': default_kinetics,
    },
    'ecoli': {
        'filename': 'iAF1260.xml',
        'config': default_kinetics,
    },
    'cdiff': {
        'filename': 'iCN900.xml',
        'config': default_kinetics,
    },
    'pputida': {
        'filename': 'iJN746.xml',
        'config': default_kinetics,
    },
    'yeast': {
        'filename': 'iMM904.xml',
        'config': default_kinetics,
    },
    'llactis': {
        'filename': 'iNF517.xml',
        'config': default_kinetics,
    },
}


def load_fba_model(model_file, bounds):
    """
    Load an SBML or named model and apply static bounds.

    Parameters:
    -----------
    - model_file: str, path to SBML or name of registered model
    - bounds: dict, {reaction_id: {'lower': val, 'upper': val}}

    Returns:
    --------
    cobra.Model instance with bounds applied
    """
    if model_file.endswith(".xml"):
        model = cobra.io.read_sbml_model(model_file)
    else:
        model = load_model(model_file)

    for rxn_id, limits in bounds.items():
        rxn = model.reactions.get_by_id(rxn_id)
        if limits.get("lower") is not None:
            rxn.lower_bound = limits["lower"]
        if limits.get("upper") is not None:
            rxn.upper_bound = limits["upper"]

    return model


def run_fba_update(model, config, substrates, biomass, interval):
    """
    Run a single FBA update step using uptake kinetics and biomass growth.

    Parameters:
    -----------
    - model: cobra.Model instance, modified in-place for bounds and optimization.
    - config: dict, must include:
        - 'kinetic_params': {substrate: (Km, Vmax)}
        - 'substrate_update_reactions': {substrate: reaction_id}
    - substrates: dict of external substrate concentrations {substrate: float}
    - biomass: current biomass (float)
    - interval: simulation time interval (float)

    Returns:
    --------
    dict with:
        - 'substrates': {substrate: delta_concentration}
        - 'biomass': delta_biomass
    """

    update_substrates = {}
    delta_biomass = 0.0

    # Set uptake bounds using Michaelis-Menten kinetics
    for substrate, reaction_id in config["substrate_update_reactions"].items():
        Km, Vmax = config["kinetic_params"][substrate]
        substrate_concentration = substrates[substrate]
        uptake_rate = Vmax * substrate_concentration / (Km + substrate_concentration)
        model.reactions.get_by_id(reaction_id).lower_bound = -uptake_rate

    # Run FBA optimization
    solution = model.optimize()

    if solution.status == "optimal":
        mu = solution.objective_value
        delta_biomass = mu * biomass * interval

        for substrate, rxn_id in config["substrate_update_reactions"].items():
            flux = solution.fluxes[rxn_id] * biomass * interval
            delta = max(flux, -substrates[substrate])  # prevent negative concentrations
            update_substrates[substrate] = delta
    else:
        for substrate in config["substrate_update_reactions"]:
            update_substrates[substrate] = 0.0

    return {
        "substrates": update_substrates,
        "biomass": delta_biomass,
    }


class DynamicFBA(Process):
    """
    A dynamic FBA process that integrates FBA optimization with substrate uptake kinetics.

    Configuration:
    -------------
    - model_file (str): Path to the SBML or named model to load.
    - kinetic_params (dict): {substrate: (Km, Vmax)} for uptake kinetics.
    - substrate_update_reactions (dict): {substrate: reaction_id} mapping each substrate to its uptake reaction.
    - bounds (dict): {reaction_id: {'lower': val, 'upper': val}} for setting static bounds.

    Inputs:
    -------
    - substrates (map[positive_float]): External concentrations of substrates.
    - biomass (positive_float): Current biomass level.

    Outputs:
    --------
    - substrates (map[float]): Changes in substrate concentrations.
    - biomass (float): Change in biomass.

    Notes:
    ------
    - Assumes units are consistent (e.g., mmol/L, gDW).
    - Negative fluxes represent uptake.
    """

    config_schema = {
        "model_file": "string",
        "kinetic_params": "map[tuple[float,float]]",
        "substrate_update_reactions": "map[string]",
        "bounds": "map[bounds]",
    }

    def initialize(self, config):
        self.model = load_fba_model(
            model_file=self.config["model_file"],
            bounds=self.config["bounds"]
        )

    def inputs(self):
        return {
            "substrates": "map[positive_float]",  # external concentrations
            "biomass": "positive_float",
        }

    def outputs(self):
        return {
            "substrates": "map[float]",   # deltas (not absolute concentrations)
            "biomass": "float",           # delta biomass
        }

    def update(self, inputs, interval):
        return run_fba_update(
            self.model,
            self.config,
            inputs["substrates"],
            inputs["biomass"],
            interval
        )


class SpatialDFBA(Process):
    """
    A spatial extension of DynamicFBA using one DFBA instance per bin.

    Configuration:
    --------------
    - n_bins (tuple[int, int]): Number of (x, y) bins.
    - Remaining config keys passed to each DynamicFBA.
    """

    config_schema = {
        'n_bins': 'tuple[integer,integer]',
        'model_file': 'string',
        'kinetic_params': 'map[tuple[float,float]]',
        'substrate_update_reactions': 'map[string]',
        'bounds': 'map[bounds]',
    }

    def initialize(self, config):
        self.model = load_fba_model(
            model_file=self.config["model_file"],
            bounds=self.config["bounds"]
        )
        self.n_bins = config["n_bins"]

    def inputs(self):
        return {
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'positive_array',
                    '_shape': self.n_bins,
                    '_data': 'float'
                },
            },
            'biomass': {
                '_type': 'array',
                '_shape': self.n_bins,
                '_data': 'float'
            }
        }

    def outputs(self):
        return {
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.n_bins,
                    '_data': 'float'
                },
            },
            'biomass': {
                '_type': 'array',
                '_shape': self.n_bins,
                '_data': 'float'
            }
        }

    def update(self, inputs, interval):
        substrate_fields = inputs['fields']
        biomass_field = inputs['biomass']
        x_bins, y_bins = self.n_bins

        # Initialize outputs with zero arrays
        delta_fields = {
            mol_id: np.zeros(self.n_bins)
            for mol_id in substrate_fields
        }
        delta_biomass = np.zeros(self.n_bins)

        # Loop through each grid cell
        for i in range(x_bins):
            for j in range(y_bins):
                bin_idx = i * y_bins + j
                # dfba = self.dfba_grid[bin_idx]

                # Extract local substrate concentrations
                local_substrates = {
                    mol_id: substrate_fields[mol_id][i, j]
                    for mol_id in substrate_fields
                }

                local_biomass = biomass_field[i, j]

                # Run DFBA update for this bin
                update = run_fba_update(
                    self.model,
                    self.config,
                    local_substrates,
                    local_biomass,
                    interval
                )

                # Accumulate updates into fields
                for mol_id, delta in update['substrates'].items():
                    delta_fields[mol_id][i, j] = delta
                delta_biomass[i, j] = update['biomass']

        return {
            'fields': delta_fields,
            'biomass': delta_biomass
        }


def analyze_fba_model_minimal_media(model_key, config, model_dir, flux_epsilon=1e-1):
    """
    Load model, apply bounds and kinetic constraints, run FBA, and print important exchange reactions.
    """
    print(f"\n=== Analyzing model: {model_key} ===")

    filename = config['filename']
    if filename.endswith('.xml'):
        model_path = os.path.join(model_dir, filename)
        model = load_fba_model(model_path, config.get('bounds', {}))
    else:
        model = load_fba_model(filename, config.get('bounds', {}))  # named model

    # Apply kinetic substrate constraints if available
    substrate_updates = config.get('substrate_update_reactions', {})
    kinetic_params = config.get('kinetic_params', {})
    for substrate, rxn_id in substrate_updates.items():
        if rxn_id in model.reactions:
            Km, Vmax = kinetic_params.get(substrate, (0.5, 1.0))
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = -Vmax
            rxn.upper_bound = 0
            print(f"  Applied kinetic bound to {rxn_id}: [-{Vmax}, 0]")

    # Run FBA
    solution = model.optimize()
    if solution.status != 'optimal':
        print(f"  ⚠ Optimization not optimal (status: {solution.status})")
    else:
        print(f"  ✅ Objective value: {solution.objective_value:.4f}")

    # Print only important exchange reactions
    print("  Important exchange reactions:")
    user_constrained_rxns = set(config.get('bounds', {}).keys()) | set(substrate_updates.values())
    for rxn in model.exchanges:
        flux = solution.fluxes.get(rxn.id, 0.0)
        default_bounds = (-1000.0, 1000.0) if rxn.reversibility else (0.0, 1000.0)
        is_user_defined = rxn.id in user_constrained_rxns
        is_constrained = (rxn.lower_bound, rxn.upper_bound) != default_bounds
        is_active = abs(flux) > flux_epsilon

        if is_user_defined or is_constrained or is_active:
            print(f"    {rxn.id:20s} Bounds: ({rxn.lower_bound:6.1f}, {rxn.upper_bound:6.1f})  Flux: {flux:7.3f}")


# Example usage
if __name__ == "__main__":
    for model_key, config in MODEL_REGISTRY_DFBA.items():
        analyze_fba_model_minimal_media(model_key, config, MODEL_DIR)
