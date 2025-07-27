"""
Dynamic FBA simulation
======================

Process for a pluggable dFBA simulation.
"""

import warnings

import cobra
from cobra.io import load_model
from process_bigraph import Process, Composite
from spatio_flux.library.functions import initialize_fields

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
    - biomass_identifier: The identifier for biomass in the current state.
    - bounds: A dictionary of bounds for any reactions in the model.

    TODO -- check units
    """

    config_schema = {
        "model_file": "string",  # TODO -- register a "path" type
        "kinetic_params": "map[tuple[float,float]]",
        "substrate_update_reactions": "map[string]",
        "biomass_identifier": "string",
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
            "substrates": "map[positive_float]"  # TODO this should be map[concentration]
            # "enzymes": "map[positive_float]"  # TODO this should be map[concentration]
        }

    def outputs(self):
        return {
            "substrates": "map[positive_float]"
        }

    # TODO -- can we just put the inputs/outputs directly in the function?
    def update(self, inputs, interval):
        substrates_input = inputs["substrates"]

        for substrate, reaction_id in self.config["substrate_update_reactions"].items():
            Km, Vmax = self.config["kinetic_params"][substrate]
            substrate_concentration = substrates_input[substrate]
            uptake_rate = Vmax * substrate_concentration / (Km + substrate_concentration)
            self.model.reactions.get_by_id(reaction_id).lower_bound = -uptake_rate

        substrate_update = {}

        solution = self.model.optimize()
        if solution.status == "optimal":
            current_biomass = substrates_input[self.config["biomass_identifier"]]
            biomass_growth_rate = solution.objective_value
            substrate_update[self.config["biomass_identifier"]] = biomass_growth_rate * current_biomass * interval

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

        return {
            "substrates": substrate_update,
        }


# Helper functions to get specs and states
def dfba_config(
        model_file="textbook",
        kinetic_params=None,
        substrate_update_reactions=None,
        biomass_identifier="biomass",
        bounds=None
):
    if substrate_update_reactions is None:
        substrate_update_reactions = {
            "glucose": "EX_glc__D_e",
            "acetate": "EX_ac_e"}
    if bounds is None:
        bounds = {
            "EX_o2_e": {"lower": -2, "upper": None},
            "ATPM": {"lower": 1, "upper": 1}}
    if kinetic_params is None:
        kinetic_params = {
            "glucose": (0.5, 1),
            "acetate": (0.5, 2)}
    return {
        "model_file": model_file,
        "kinetic_params": kinetic_params,
        "substrate_update_reactions": substrate_update_reactions,
        "biomass_identifier": biomass_identifier,
        "bounds": bounds
    }

def build_path(base_path, mol_id, i=None, j=None):
    """
    Constructs a path list for a molecule, optionally appending indices.

    Parameters:
        base_path (list of str): The base path prefix (e.g., ["..", "fields"]).
        mol_id (str): The molecule ID to insert in the path.
        i (int, optional): First index to append, if provided.
        j (int, optional): Second index to append, if provided.

    Returns:
        list: The full path as a list of path elements.
    """
    full_path = base_path + [mol_id]
    if i is not None:
        full_path.append(i)
    if j is not None:
        full_path.append(j)
    return full_path


def get_single_dfba_spec(
        model_file="textbook",
        mol_ids=None,
        path=None,
        i=None,
        j=None,
):
    """
    Constructs a configuration dictionary for a dynamic FBA process with optional path indices.
    """
    if path is None:
        path = ["..", "fields"]
    if mol_ids is None:
        mol_ids = ["glucose", "acetate", "biomass"]

    return {
        "_type": "process",
        "address": "local:DynamicFBA",
        "config": dfba_config(model_file=model_file),
        "inputs": {
            "substrates": {mol_id: build_path(path, mol_id, i, j) for mol_id in mol_ids}
        },
        "outputs": {
            "substrates": {mol_id: build_path(path, mol_id, i, j) for mol_id in mol_ids}
        }
    }


def get_spatial_dfba_spec(n_bins=(5, 5), mol_ids=None):
    if mol_ids is None:
        mol_ids = ["glucose", "acetate", "biomass"]
    dfba_processes_dict = {}
    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            dfba_processes_dict[f"[{i},{j}]"] = get_single_dfba_spec(mol_ids=mol_ids, path=["..", "fields"], i=i, j=j)
    return dfba_processes_dict


def get_spatial_dfba_state(
        n_bins=(5, 5),
        mol_ids=None,
        initial_min_max=None,  # {mol_id: (min, max)}
):
    if mol_ids is None:
        mol_ids = ["glucose", "acetate", "biomass"]

    initial_fields = initialize_fields(n_bins=n_bins, initial_min_max=initial_min_max)

    return {
        "fields": {
            "_type": "map",
            "_value": {
                "_type": "array",
                "_shape": n_bins,
                "_data": "positive_float"
            },
            **initial_fields,
        },
        "spatial_dfba": get_spatial_dfba_spec(n_bins=n_bins, mol_ids=mol_ids)
    }
