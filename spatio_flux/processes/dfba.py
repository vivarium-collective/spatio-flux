"""
Dynamic FBA simulation
======================

Process for a pluggable dFBA simulation.
"""

import warnings

import cobra
from cobra.io import load_model
from process_bigraph import Process

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="cobra.medium.boundary_types"
)

class DynamicFBA(Process):
    """
    Performs dynamic FBA.

    Parameters:
    - model: The metabolic model for the simulation.
    - kinetic_params: Kinetic parameters (Km and Vmax) for each substrate.
    - substrate_update_reactions: A dictionary mapping substrates to their update reactions.
    - bounds: A dictionary of bounds for any reactions in the model.

    TODO -- check units
    """

    config_schema = {
        "model_file": "string",  # TODO -- register a "path" type
        "kinetic_params": "map[tuple[float,float]]",
        "substrate_update_reactions": "map[string]",
        "bounds": "map[bounds]",
    }

    def initialize(self, config):
        if not "xml" in self.config["model_file"]:
            # use the textbook model if no model file is provided
            # TODO: Also handle JSON or .mat model files
            self.model = load_model(self.config["model_file"])
        elif isinstance(self.config["model_file"], str):
            self.model = cobra.io.read_sbml_model(self.config["model_file"])
        else:
            # error handling
            raise ValueError("Invalid model file")

        for reaction_id, bounds in self.config["bounds"].items():
            if bounds["lower"] is not None:
                self.model.reactions.get_by_id(reaction_id).lower_bound = bounds["lower"]
            if bounds["upper"] is not None:
                self.model.reactions.get_by_id(reaction_id).upper_bound = bounds["upper"]

    def inputs(self):
        return {
            "substrates": "map[positive_float]",  # TODO this should be map[concentration]
            "biomass": "positive_float",
            # "enzymes": "map[positive_float]"  # TODO this should be map[concentration]
        }

    def outputs(self):
        return {
            "substrates": "map[positive_float]",
            "biomass": "positive_float",
        }

    # TODO -- can we just put the inputs/outputs directly in the function?
    def update(self, inputs, interval):
        substrates_input = inputs["substrates"]
        current_biomass = inputs["biomass"]

        for substrate, reaction_id in self.config["substrate_update_reactions"].items():
            Km, Vmax = self.config["kinetic_params"][substrate]
            substrate_concentration = substrates_input[substrate]
            uptake_rate = Vmax * substrate_concentration / (Km + substrate_concentration)
            self.model.reactions.get_by_id(reaction_id).lower_bound = -uptake_rate

        substrate_update = {}
        biomass_update = 0.0

        solution = self.model.optimize()
        if solution.status == "optimal":
            biomass_growth_rate = solution.objective_value
            biomass_update = biomass_growth_rate * current_biomass * interval

            for substrate, reaction_id in self.config["substrate_update_reactions"].items():
                flux = solution.fluxes[reaction_id] * current_biomass * interval
                old_concentration = substrates_input[substrate]
                new_concentration = max(old_concentration + flux, 0)  # keep above 0 -- TODO this should not happen
                substrate_update[substrate] = new_concentration - old_concentration
                # TODO -- assert not negative?
        else:
            # Handle non-optimal solutions if necessary
            # print("Non-optimal solution, skipping update")
            for substrate, reaction_id in self.config["substrate_update_reactions"].items():
                substrate_update[substrate] = 0

        print(f'dFBA update: {substrate_update}, biomass update: {biomass_update}')
        return {
            "substrates": substrate_update,
            "biomass": biomass_update,
        }
