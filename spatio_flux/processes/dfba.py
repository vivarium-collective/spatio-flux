"""
Dynamic FBA simulation
======================

A pluggable dynamic Flux Balance Analysis (dFBA) process.
Performs time-stepped metabolic modeling by combining COBRApy-based
optimization with kinetic uptake constraints.
"""

import warnings
import cobra
from cobra.io import load_model
from process_bigraph import Process

# Suppress known benign warnings from COBRApy
warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings("ignore", category=FutureWarning, module="cobra.medium.boundary_types")


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
        # Load model: named model or SBML file
        model_file = self.config["model_file"]
        if model_file.endswith(".xml"):
            self.model = cobra.io.read_sbml_model(model_file)
        else:
            self.model = load_model(model_file)

        # Set user-defined bounds
        for rxn_id, limits in self.config["bounds"].items():
            rxn = self.model.reactions.get_by_id(rxn_id)
            if limits.get("lower") is not None:
                rxn.lower_bound = limits["lower"]
            if limits.get("upper") is not None:
                rxn.upper_bound = limits["upper"]

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
        substrates = inputs["substrates"]
        biomass = inputs["biomass"]
        update_substrates = {}
        delta_biomass = 0.0

        for substrate, reaction_id in self.config["substrate_update_reactions"].items():
            Km, Vmax = self.config["kinetic_params"][substrate]
            substrate_concentration = substrates[substrate]
            uptake_rate = Vmax * substrate_concentration / (Km + substrate_concentration)
            self.model.reactions.get_by_id(reaction_id).lower_bound = -uptake_rate

        # Run FBA optimization
        solution = self.model.optimize()

        if solution.status == "optimal":
            mu = solution.objective_value
            delta_biomass = mu * biomass * interval

            for substrate, rxn_id in self.config["substrate_update_reactions"].items():
                flux = solution.fluxes[rxn_id] * biomass * interval
                delta = max(flux, -substrates[substrate])  # prevent negative concentrations
                update_substrates[substrate] = delta
        else:
            # No biomass growth or substrate consumption
            for substrate, reaction_id in self.config["substrate_update_reactions"].items():
                update_substrates[substrate] = 0

        return {
            "substrates": update_substrates,
            "biomass": delta_biomass,
        }
