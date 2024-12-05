"""
Dynamic FBA simulation
======================

Process for a pluggable dFBA simulation.
"""

import warnings

import numpy as np
import cobra
from cobra.io import load_model
from process_bigraph import Process, Composite
from bigraph_viz import plot_bigraph
from spatio_flux.viz.plot import plot_time_series, plot_species_distributions_to_gif

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

    def __init__(self, config, core):
        super().__init__(config, core)

        if not "xml" in self.config["model_file"]:
            # use the textbook model if no model file is provided
            # TODO: Also handle JSON or .mat model files
            self.model = load_model(self.config["model_file"])
        elif isinstance(self.config["model_file"], str):
            self.model = cobra.io.read_sbml_model(self.config["model_file"])
        else:
            # error handling
            raise ValueError('Invalid model file')

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
            'substrates': 'map[positive_float]'
        }

    # TODO -- can we just put the inputs/outputs directly in the function?
    def update(self, state, interval):
        substrates_input = state["substrates"]

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


def get_single_dfba_spec(
        model_file="textbook",
        mol_ids=None,
        path=None,
        i=None,
        j=None,
):
    """
    Constructs a configuration dictionary for a dynamic FBA process with optional path indices.

    This function builds a process specification for use with a dynamic FBA system. It allows
    specification of substrate molecule IDs and optionally appends indices to the paths for those substrates.

    Parameters:
        mol_ids (list of str, optional): List of molecule IDs to include in the process. Defaults to
                                         ["glucose", "acetate", "biomass"].
        path (list of str, optional): The base path to prepend to each molecule ID. Defaults to ["..", "fields"].
        i (int, optional): The first index to append to the path for each molecule, if not None.
        j (int, optional): The second index to append to the path for each molecule, if not None.

    Returns:
        dict: A dictionary containing the process type, address, configuration, and paths for inputs
              and outputs based on the specified molecule IDs and indices.
    """
    if path is None:
        path = ["..", "fields"]
    if mol_ids is None:
        mol_ids = ["glucose", "acetate", "biomass"]

    # Function to build the path with optional indices
    def build_path(mol_id):
        base_path = path + [mol_id]
        if i is not None:
            base_path.append(i)
        if j is not None:
            base_path.append(j)
        return base_path

    return {
        "_type": "process",
        "address": "local:DynamicFBA",
        "config": dfba_config(model_file=model_file),
        "inputs": {
            "substrates": {mol_id: build_path(mol_id) for mol_id in mol_ids}
        },
        "outputs": {
            "substrates": {mol_id: build_path(mol_id) for mol_id in mol_ids}
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
    if initial_min_max is None:
        initial_min_max = {"glucose": (0, 20), "acetate": (0,0 ), "biomass": (0, 0.1)}

    initial_fields = {
        mol_id: np.random.uniform(low=initial_min_max[mol_id][0],
                                  high=initial_min_max[mol_id][1],
                                  size=n_bins)
        for mol_id in mol_ids}

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


def run_dfba_single(
        total_time=60,
        mol_ids=None,
):
    single_dfba_config = {
        'dfba': get_single_dfba_spec(path=['fields']),
        'fields': {
            'glucose': 10,
            'acetate': 0,
            'biomass': 0.1
        }
    }

    # make the simulation
    sim = Composite({
        'state': single_dfba_config,
        'emitter': {'mode': 'all'}
    }, core=core)

    # save the document
    sim.save(filename='single_dfba.json', outdir='out')

    # simulate
    print('Simulating...')
    sim.update({}, total_time)

    # gather results
    dfba_results = sim.gather_results()

    print('Plotting results...')
    # plot timeseries
    plot_time_series(
        dfba_results,
        # coordinates=[(0, 0), (1, 1), (2, 2)],
        out_dir='out',
        filename='dfba_single_timeseries.png',
    )


def run_dfba_spatial(
        total_time=60,
        n_bins=(3, 3),  # TODO -- why can't do (5, 10)??
        mol_ids=None,
):
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']
    composite_state = get_spatial_dfba_state(
        n_bins=n_bins,
        mol_ids=mol_ids,
    )

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'}
    }, core=core)

    # save the document
    sim.save(filename='spatial_dfba.json', outdir='out')

    # # save a viz figure of the initial state
    # plot_bigraph(
    #     state=sim.state,
    #     schema=sim.composition,
    #     core=core,
    #     out_dir='out',
    #     filename='dfba_spatial_viz'
    # )

    # simulate
    print('Simulating...')
    sim.update({}, total_time)

    # gather results
    dfba_results = sim.gather_results()

    print('Plotting results...')
    # plot timeseries
    plot_time_series(
        dfba_results,
        coordinates=[(0, 0), (1, 1), (2, 2)],
        out_dir='out',
        filename='dfba_timeseries.png',
    )

    # make video
    plot_species_distributions_to_gif(
        dfba_results,
        out_dir='out',
        filename='dfba_results.gif',
        title='',
        skip_frames=1
    )


if __name__ == '__main__':
    run_dfba_single()
    run_dfba_spatial(n_bins=(8,8))
