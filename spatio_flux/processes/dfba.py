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
from pathlib import Path

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
    'ecoli core': {
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
        'model_file': 'iAF1260.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'formate': 'EX_for_e',
            # 'acetate': 'EX_ac_e'
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            'formate': (0.5, 1),
            # 'acetate': (0.5, 2)
        },
        # 'bounds': {
        #     'EX_o2_e': {'lower': -2, 'upper': None},
        #     'ATPM': {'lower': 1, 'upper': 1}
        # },
    },
    'cdiff': {
        'model_file': 'iCN900.xml',
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
        'model_file': 'iJN746.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'ammonium': 'EX_nh4_e',
            'glycolate': 'EX_glyclt_e'
            # 'glycerol': 'EX_gly_e'
        },
        'kinetic_params': {
            'glucose': (1, 2),
            'ammonium': (2, 4),
            'glycolate': (0.5, 1)
            # 'glycerol': (0.5, 2)
        }
    },
    'yeast': {
        'model_file': 'iMM904.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'ammonium': 'EX_nh4_e',
            # 'ethanol': 'EX_etoh_e',
            # 'formate': 'EX_for_e',
        },
        'kinetic_params': {
            'glucose': (0.5, 1),
            'ammonium': (0.5, 1),
            # 'ethanol': (0.5, 2),
            # 'formate': (0.5, 1),
        }
    },
    'llactis': {
        'model_file': 'iNF517.xml',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'glutatmate': 'EX_glu__L_e',
            'serine': 'EX_ser__L_e',
            # 'ammonium': 'EX_nh4_e',
            # 'glutamine': 'EX_gln__L_e',
            # 'arginine': 'EX_arg__L_e'
        },
        'kinetic_params': {
            'glucose': (0.5, 1.25),
            'glutatmate': (0.05, 0.1),
            'serine': (0.05, 0.1),
            # 'ammonium': (0.5, 1),
            # 'glutamine': (0.5, 2),
            # 'arginine': (0.5, 2)
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
    if model_file in MODEL_REGISTRY_DFBA:
        # Load a named model from the registry
        model_config = MODEL_REGISTRY_DFBA[model_file]
        model_file = model_config['model_file']


    # Get the path to the directory containing *this* file
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / '..' / 'models'
    full_path = (models_dir / model_file).resolve()

    try:
        if model_file.endswith('.xml'):
            if not full_path.exists():
                raise FileNotFoundError(f"SBML file not found at: {full_path}")
            model = cobra.io.read_sbml_model(str(full_path))
        else:
            # Load a named model from registry
            model = load_model(model_file)
    except:
        raise ValueError(f"Failed to load model from {model_file}. Ensure it is a valid SBML file or registered model name.")

    for rxn_id, limits in bounds.items():
        rxn = model.reactions.get_by_id(rxn_id)
        lower = limits.get("lower")
        upper = limits.get("upper")
        if lower is not None and lower != {}:
            rxn.lower_bound = limits["lower"]
        if upper is not None and upper != {}:
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
        update = run_fba_update(
            self.model,
            self.config,
            inputs["substrates"],
            inputs["biomass"],
            interval
        )
        return update


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
        'model_file': 'maybe[string]',  # if provided, will fill unspecified model grid locations
        'models': 'any',  #map[string,map]',        # for multiple models
        'model_grid': 'list[list[string]]',  #'list[list[string]',  # grid of model IDs
        'kinetic_params': 'map[tuple[float,float]]',
        'substrate_update_reactions': 'map[string]',
        'bounds': 'map[bounds]',
    }

    def initialize(self, config):
        self.n_bins = config['n_bins']
        self.default_model_file = config.get("model_file")

        # Load models
        self.models = {}
        model_configs = config.get('models', {})
        for model_id, model_config in model_configs.items():
            self.models[model_id] = load_fba_model(
                model_file=model_config['model_file'],
                bounds=model_config.get('bounds', {})
            )

        # Register default model if provided
        if self.default_model_file and 'default' not in self.models:
            self.models['default'] = load_fba_model(
                model_file=self.default_model_file,
                bounds=config.get('bounds', {})
            )

        # Initialize model grid
        if self.default_model_file:
            # Fill with 'default' model if specified
            model_grid_array = np.full(self.n_bins, 'default', dtype='U20')
        else:
            model_grid_array = np.full(self.n_bins, '', dtype='U20')

        # Update from model_grid_config if provided
        model_grid_config = config.get('model_grid')
        if model_grid_config is not None:
            model_grid_config = np.array(model_grid_config, dtype='U20')

            # Update only where model_grid_config has non-empty strings
            for index, value in np.ndenumerate(model_grid_config):
                if value != '':
                    model_grid_array[index] = value

        # Validate all entries in final model_grid
        unique_ids = set(np.unique(model_grid_array))
        unique_ids.discard('')
        unknown_ids = unique_ids - set(self.models.keys())
        if unknown_ids:
            raise ValueError(f"Unknown model IDs in model_grid: {unknown_ids}")

        self.model_grid = model_grid_array

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
                # Extract local substrate concentrations
                local_substrates = {
                    mol_id: substrate_fields[mol_id][i, j]
                    for mol_id in substrate_fields
                }

                local_biomass = biomass_field[i, j]

                # get the local model from the model gri
                model_id = self.model_grid[i][j]
                if model_id == '':
                    continue  # skip empty cells

                model = self.models[model_id]

                # Run DFBA update for this bin
                update = run_fba_update(
                    model,
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


def get_field_names(model_registry):
    all_fields = set()
    for model_info in model_registry.values():
        config = model_info.get('config', {})
        all_fields.update(config.get('substrate_update_reactions', {}).keys())
        all_fields.update(config.get('kinetic_params', {}).keys())
    return sorted(all_fields)


def restore_bounds_safely(rxn, lb, ub):
    """Restore bounds safely regardless of current invalid state."""
    # Set to dummy permissive bounds first to avoid intermediate errors
    rxn.lower_bound = -1000
    rxn.upper_bound = 1000
    rxn.lower_bound = lb
    rxn.upper_bound = ub


from cobra.medium import minimal_medium

def analyze_fba_model_minimal_media(model_key, config, model_dir, flux_epsilon=1e-10, top_k=10, growth_threshold=1e-6):
    """
    Load model, apply bounds and kinetic constraints, run FBA, and print important exchange reactions,
    including minimal media, alternate nutrient sources, secreted byproducts, and growth-limiting exchanges.
    """
    print(f"\n=== Analyzing model: {model_key} ===")

    # --- Load Model ---
    model_file = config['model_file'].removeprefix('models/')
    if model_file.endswith('.xml'):
        model_path = os.path.join(model_dir, model_file)
        model = load_fba_model(model_path, config.get('bounds', {}))
    else:
        model = load_fba_model(model_file, config.get('bounds', {}))  # named model

    # --- Step 1: Full Media Optimization ---
    print("\n--- Step 1: Optimize with Full Media ---")
    print(f'Objective expression: {model.objective.expression}')
    full_solution = model.optimize()
    if full_solution.status != 'optimal':
        print(f"  ⚠ Optimization not optimal (status: {full_solution.status})")
        return
    baseline_growth = full_solution.objective_value
    print(f"  ✅ Baseline growth rate: {baseline_growth:.6f}")

    # --- Step 2: Active Exchange Reactions (Full Media) ---
    active_exchanges = []
    uptake_reactions = []
    secretion_reactions = []

    print("\n--- Step 2: Active Exchange Reactions ---")
    for rxn in model.exchanges:
        flux = full_solution.fluxes.get(rxn.id, 0.0)
        default_bounds = (-1000.0, 1000.0) if rxn.reversibility else (0.0, 1000.0)
        is_constrained = (rxn.lower_bound, rxn.upper_bound) != default_bounds
        is_active = abs(flux) > flux_epsilon

        if is_constrained or is_active:
            if flux < -flux_epsilon:
                uptake_reactions.append((rxn.id, flux))
                print(f"    {rxn.id:20s} Bounds: ({rxn.lower_bound:6.1f}, {rxn.upper_bound:6.1f})  Flux: {flux:12.6f}  [uptake]")
            elif flux > flux_epsilon:
                secretion_reactions.append((rxn.id, flux))
                print(f"    {rxn.id:20s} Bounds: ({rxn.lower_bound:6.1f}, {rxn.upper_bound:6.1f})  Flux: {flux:12.6f}  [secretion]")

        if is_active:
            active_exchanges.append((rxn, flux))

    # --- Step 3: Minimal Media Optimization ---
    print("\n--- Step 3: Minimal Media Analysis ---")
    if baseline_growth < growth_threshold:
        print(f"  ⚠ Baseline growth too low for minimal media analysis (growth: {baseline_growth:.6e})")
        min_solution = full_solution
        min_media = {}
    else:
        print("  🧪 Computing minimal media from baseline growth...")
        min_media = minimal_medium(model, baseline_growth * 0.99)

        # Block all uptakes
        for rxn in model.exchanges:
            if rxn.lower_bound < 0 and rxn.upper_bound > 0:
                rxn.lower_bound = 0.0

        # Set only the minimal uptakes
        for rxn_id, flux in min_media.items():
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = -abs(flux)

        # Re-optimize
        min_solution = model.optimize()
        if min_solution.status != 'optimal':
            print("  ❌ Optimization failed under minimal media.")
            return
        print(f"  ✅ Growth with minimal media: {min_solution.objective_value:.6f}")

    # --- Step 4: Minimal Media Uptakes ---
    minimal_uptakes = [
        (rxn_id, flux) for rxn_id, flux in min_solution.fluxes.items()
        if rxn_id in min_media and flux < -flux_epsilon
    ]
    if minimal_uptakes:
        print("\n  📦 Required Minimal Media Uptakes:")
        for rxn_id, flux in sorted(minimal_uptakes, key=lambda x: x[1]):
            print(f"    {rxn_id:20s}  Uptake flux: {flux:10.6f}")
    else:
        print("  ⚠ No active uptake reactions under minimal media.")

    # --- Step 5: Top Uptake Sources ---
    print(f"\n--- Step 4: Top {top_k} Uptake Sources ---")
    top_uptake = sorted(minimal_uptakes, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    for i, (rxn_id, flux) in enumerate(top_uptake, 1):
        print(f"    {i}. {rxn_id:20s}  Flux: {flux:10.6f}")

    # --- Step 6: Alternate Uptake Source Analysis ---
    print(f"\n--- Step 5: Alternate Uptake Sources ---")
    alternate_sources = []

    for rxn_id, flux in top_uptake:
        rxn = model.reactions.get_by_id(rxn_id)
        original_lb, original_ub = rxn.lower_bound, rxn.upper_bound
        restore_bounds_safely(rxn, 0.0, 0.0)  # knockout top uptake

        alt_sol = model.optimize()
        restore_bounds_safely(rxn, original_lb, original_ub)

        if alt_sol.status == 'optimal' and alt_sol.objective_value > growth_threshold:
            # Look for other uptake reactions that now carry flux
            alt_uptakes = [
                (r.id, alt_sol.fluxes.get(r.id, 0.0))
                for r in model.exchanges
                if alt_sol.fluxes.get(r.id, 0.0) < -flux_epsilon and r.id != rxn_id
            ]
            if alt_uptakes:
                alternate_sources.append((rxn_id, alt_sol.objective_value, alt_uptakes))
                print(f"  🔄 Alternate source(s) found for {rxn_id} (growth: {alt_sol.objective_value:.6f}):")
                for alt_id, alt_flux in sorted(alt_uptakes, key=lambda x: x[1]):
                    print(f"    → {alt_id:20s}  Flux: {alt_flux:10.6f}")
            else:
                print(f"  ⚠ {rxn_id}: growth sustained, but no new uptakes found")
        else:
            print(f"  🚫 {rxn_id}: growth not sustained after removal")

    # --- Step 7: Growth-Limiting Reaction Knockouts ---
    print("\n--- Step 6: Growth-Limiting Knockout Analysis ---")
    limiting_rxns = []
    for rxn, flux in active_exchanges:
        original_lb, original_ub = rxn.lower_bound, rxn.upper_bound
        restore_bounds_safely(rxn, 0.0, 0.0)

        perturbed_sol = model.optimize()
        restore_bounds_safely(rxn, original_lb, original_ub)

        if perturbed_sol.status == 'optimal':
            drop = min_solution.objective_value - perturbed_sol.objective_value
            if drop > 1e-12 and perturbed_sol.objective_value > 1e-12:
                limiting_rxns.append({
                    'id': rxn.id,
                    'flux': flux,
                    'drop': drop,
                    'new_growth': perturbed_sol.objective_value
                })
        else:
            print(f"    ⚠ Blocking {rxn.id} made the model infeasible")

    if limiting_rxns:
        limiting_rxns.sort(key=lambda r: r['drop'], reverse=True)
        print(f"\n  🚨 Top {min(top_k, len(limiting_rxns))} Growth-Limiting Exchanges:")
        for i, rxn in enumerate(limiting_rxns[:top_k], 1):
            print(
                f"    {i}. {rxn['id']:20s}  Flux: {rxn['flux']:9.4f}  Growth drop: {rxn['drop']:.6f} → {rxn['new_growth']:.6f}")
    else:
        print("  ⚠ No exchange flux significantly limits growth under minimal media.")


# Example usage
if __name__ == "__main__":
    for model_key, config in MODEL_REGISTRY_DFBA.items():
        analyze_fba_model_minimal_media(model_key, config, MODEL_DIR)
