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
from spatio_flux.library.helpers import build_path

# Suppress known benign warnings from COBRApy
warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings("ignore", category=FutureWarning, module="cobra.medium.boundary_types")

# Define static bounds for specific models (can customize per model if needed)
default_bounds = {}

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

default_kinetics = {
    'substrate_update_reactions': {
        'glucose': 'EX_glc__D_e',
        'acetate': 'EX_ac_e'
    },
    'kinetic_params': {
        'glucose': (0.5, 1),
        'acetate': (0.5, 2)
    }
}

MODEL_REGISTRY_DFBA = {
    'textbook': {
        'model_file': 'textbook',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'acetate': 'EX_ac_e',
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            'acetate': (0.5, 2),
        },
        'bounds': {
            'EX_o2_e': {'lower': -2, 'upper': None},
            'ATPM': {'lower': 1, 'upper': 1}
        },
    },
    'ecoli': {
        'model_file': 'models/iAF1260.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            # 'acetate': 'EX_ac_e'
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            # 'acetate': (0.5, 2)
        },
        'bounds': {
            'EX_o2_e': {'lower': -2, 'upper': None},
            'ATPM': {'lower': 1, 'upper': 1}
        },
    },
    'cdiff': {
        'model_file': 'models/iCN900.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'acetate': 'EX_ac_e'
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            'acetate': (0.5, 2)
        }
    },
    'pputida': {
        'model_file': 'models/iJN746.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'glycerol': 'EX_gly_e'
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            'glycerol': (0.5, 2)
        }
    },
    'yeast': {
        'model_file': 'models/iMM904.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'ethanol': 'EX_etoh_e'
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            'ethanol': (0.5, 2)
        }
    },
    'llactis': {
        'model_file': 'models/iNF517.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'ammonium': 'EX_nh4_e',
            'glutamine': 'EX_gln__L_e',
            'arginine': 'EX_arg__L_e'
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            'ammonium': (0.5, 1),
            'glutamine': (0.5, 2),
            'arginine': (0.5, 2)
        }
    },
}


def get_dfba_process_from_registry(
    model_id,
    path,
    biomass_id=None,
    i=None,
    j=None,
):
    model_config = MODEL_REGISTRY_DFBA[model_id]
    mol_ids = model_config['substrate_update_reactions'].keys()
    biomass_id = biomass_id or 'biomass'

    return {
        "_type": "process",
        "address": "local:DynamicFBA",
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

def validate_model_registry_substrates(model_registry):
    """
    Validate that 'substrate_update_reactions' and 'kinetic_params' fields match for each model.
    Also returns the set of all substrate fields across all models.

    :param model_registry: A dictionary like MODEL_REGISTRY_DFBA.
    :return: A sorted list of unique substrate fields used across all models.
    """
    all_fields = set()

    for model_key, model_info in model_registry.items():
        config = model_info.get('config', {})
        reactions = config.get('substrate_update_reactions', {})
        kinetics = config.get('kinetic_params', {})

        fields_reactions = set(reactions.keys())
        fields_kinetics = set(kinetics.keys())

        if fields_reactions != fields_kinetics:
            raise AssertionError(
                f"Mismatch in substrate fields for model '{model_key}':\n"
                f"  In substrate_update_reactions: {sorted(fields_reactions)}\n"
                f"  In kinetic_params:             {sorted(fields_kinetics)}"
            )

        all_fields.update(fields_reactions)

    return sorted(all_fields)


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
    try:
        if model_file.endswith(".xml"):
            model = cobra.io.read_sbml_model(model_file)
        else:
            model = load_model(model_file)
    except:
        raise ValueError(f"Failed to load model from {model_file}. Ensure it is a valid SBML file or registered model name.")

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
        uptake_rate = -1 * Vmax * substrate_concentration / (Km + substrate_concentration)

        if model.reactions.get_by_id(reaction_id).upper_bound < uptake_rate:
            # If the current upper bound is lower than the calculated uptake rate, adjust it
            model.reactions.get_by_id(reaction_id).upper_bound = uptake_rate

        # set the lower bound
        model.reactions.get_by_id(reaction_id).lower_bound = uptake_rate

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
            model_file=config["model_file"],
            bounds=config["bounds"]
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

def restore_bounds_safely(rxn, lb, ub):
    """Restore bounds safely regardless of current invalid state."""
    # Set to dummy permissive bounds first to avoid intermediate errors
    rxn.lower_bound = -1000
    rxn.upper_bound = 1000
    rxn.lower_bound = lb
    rxn.upper_bound = ub


def analyze_fba_model_minimal_media(model_key, config, model_dir, flux_epsilon=1e-5, top_k=10):
    """
    Load model, apply bounds and kinetic constraints, run FBA, and print important exchange reactions,
    including minimal media, main sources of uptake, and top growth-limiting exchanges.
    """
    print(f"\n=== Analyzing model: {model_key} ===")

    model_file = config['model_file'].removeprefix('models/')
    if model_file.endswith('.xml'):
        model_path = os.path.join(model_dir, model_file)
        model = load_fba_model(model_path, config.get('bounds', {}))
    else:
        model = load_fba_model(model_file, config.get('bounds', {}))  # named model

    # Run base FBA
    solution = model.optimize()
    if solution.status != 'optimal':
        print(f"  ⚠ Optimization not optimal (status: {solution.status})")
        return
    baseline_growth = solution.objective_value
    print(f"  ✅ Objective value: {baseline_growth:.4f}")

    # Track exchange reaction data
    active_exchanges = []
    uptake_reactions = []

    print("\n  Important exchange reactions:")
    for rxn in model.exchanges:
        flux = solution.fluxes.get(rxn.id, 0.0)
        default_bounds = (-1000.0, 1000.0) if rxn.reversibility else (0.0, 1000.0)
        is_constrained = (rxn.lower_bound, rxn.upper_bound) != default_bounds
        is_active = abs(flux) > flux_epsilon

        if is_constrained or is_active:
            if flux > flux_epsilon:
                print(f"    {rxn.id:20s} Bounds: ({rxn.lower_bound:6.1f}, {rxn.upper_bound:6.1f})  Flux: {flux:12.6f}")
            elif flux < -flux_epsilon:
                uptake_reactions.append((rxn.id, flux))
                print(f"    {rxn.id:20s} Bounds: ({rxn.lower_bound:6.1f}, {rxn.upper_bound:6.1f})  Flux: {flux:12.6f}  [uptake]")

        if is_active:
            active_exchanges.append((rxn, flux))

    # --- Minimal Media Report ---
    minimal_media = sorted([(rxn_id, flux) for rxn_id, flux in uptake_reactions], key=lambda x: x[1])
    if minimal_media:
        print("\n  📦 Minimal media (required uptake):")
        for rxn_id, flux in minimal_media:
            print(f"    {rxn_id:20s}  Uptake flux: {flux:10.6f}")
    else:
        print("\n  ⚠ No uptake reactions identified for minimal media.")

    # --- Top Uptake Sources ---
    top_uptake = sorted(uptake_reactions, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    print(f"\n  🔋 Top {len(top_uptake)} sources of uptake:")
    for i, (rxn_id, flux) in enumerate(top_uptake, 1):
        print(f"    {i}. {rxn_id:20s}  Flux: {flux:10.6f}")

    # --- Knockout Analysis ---
    limiting_rxns = []
    for rxn, flux in active_exchanges:
        original_lb, original_ub = rxn.lower_bound, rxn.upper_bound

        # SAFE KNOCKOUT
        rxn.lower_bound = -1000.0
        rxn.upper_bound = 1000.0
        rxn.lower_bound = 0.0
        rxn.upper_bound = 0.0

        perturbed_sol = model.optimize()
        restore_bounds_safely(rxn, original_lb, original_ub)

        if perturbed_sol.status == 'optimal':
            drop = baseline_growth - perturbed_sol.objective_value
            if drop > 1e-12 and perturbed_sol.objective_value > 1e-12:
                limiting_rxns.append({
                    'id': rxn.id,
                    'flux': flux,
                    'drop': drop,
                    'new_growth': perturbed_sol.objective_value
                })
        else:
            print(f"    ⚠ Blocking {rxn.id} made the model infeasible")

    # --- Top Growth-Limiting Reactions ---
    if limiting_rxns:
        limiting_rxns.sort(key=lambda r: r['drop'], reverse=True)
        print(f"\n  🚨 Top {min(top_k, len(limiting_rxns))} growth-limiting exchanges:")
        for i, rxn in enumerate(limiting_rxns[:top_k], 1):
            print(
                f"    {i}. {rxn['id']:20s}  Flux: {rxn['flux']:9.4f}  Growth drop: {rxn['drop']:.6f} → {rxn['new_growth']:.6f}")
    else:
        print("\n  ⚠ No exchange flux significantly limits growth.")


# Example usage
if __name__ == "__main__":
    for model_key, config in MODEL_REGISTRY_DFBA.items():
        analyze_fba_model_minimal_media(model_key, config, MODEL_DIR)
