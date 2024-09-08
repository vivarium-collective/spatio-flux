"""
Dynamic FBA simulation in a spatial context.
"""

import numpy as np
import warnings
import cobra
from cobra.io import load_model
from process_bigraph import Process, Composite
from process_bigraph.processes.parameter_scan import RunProcess
from processes import core  # import the core from the processes package
from plot.fields import plot_time_series, plot_species_distributions_to_gif

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings("ignore", category=FutureWarning, module="cobra.medium.boundary_types")


# create new types
def apply_non_negative(schema, current, update, core):
    new_value = current + update
    return max(0, new_value)


positive_float = {
    '_type': 'positive_float',
    '_inherit': 'float',
    '_apply': apply_non_negative
}
core.register('positive_float', positive_float)

bounds_type = {
    'lower': 'maybe[float]',
    'upper': 'maybe[float]'
}
core.register_process('bounds', bounds_type)

# TODO -- check the function signature of the apply method and report missing keys upon registration

MODEL_FOR_TESTING = load_model('textbook')


class DynamicFBA(Process):
    """
    Performs dynamic FBA.

    Parameters:
    - model: The metabolic model for the simulation.
    - kinetic_params: Kinetic parameters (Km and Vmax) for each substrate.
    - biomass_reaction: The identifier for the biomass reaction in the model.
    - substrate_update_reactions: A dictionary mapping substrates to their update reactions.
    - biomass_identifier: The identifier for biomass in the current state.

    TODO -- check units
    """

    config_schema = {
        'model_file': 'string',
        'model': 'Any',
        'kinetic_params': 'map[tuple[float,float]]',
        'biomass_reaction': {
            '_type': 'string',
            '_default': 'Biomass_Ecoli_core'
        },
        'substrate_update_reactions': 'map[string]',
        'biomass_identifier': 'string',
        'bounds': 'map[bounds]',
    }

    def __init__(self, config, core):
        super().__init__(config, core)

        if self.config['model_file'] == 'TESTING':
            self.model = MODEL_FOR_TESTING
        elif not 'xml' in self.config['model_file']:
            # use the textbook model if no model file is provided
            self.model = load_model(self.config['model_file'])
        elif isinstance(self.config['model_file'], str):
            self.model = cobra.io.read_sbml_model(self.config['model_file'])
        else:
            # error handling
            raise ValueError('Invalid model file')

        for reaction_id, bounds in self.config['bounds'].items():
            if bounds['lower'] is not None:
                self.model.reactions.get_by_id(reaction_id).lower_bound = bounds['lower']
            if bounds['upper'] is not None:
                self.model.reactions.get_by_id(reaction_id).upper_bound = bounds['upper']

    def inputs(self):
        return {
            'substrates': 'map[positive_float]'
        }

    def outputs(self):
        return {
            'substrates': 'map[positive_float]'
        }

    # TODO -- can we just put the inputs/outputs directly in the function?
    def update(self, state, interval):
        substrates_input = state['substrates']

        for substrate, reaction_id in self.config['substrate_update_reactions'].items():
            Km, Vmax = self.config['kinetic_params'][substrate]
            substrate_concentration = substrates_input[substrate]
            uptake_rate = Vmax * substrate_concentration / (Km + substrate_concentration)
            self.model.reactions.get_by_id(reaction_id).lower_bound = -uptake_rate

        substrate_update = {}

        solution = self.model.optimize()
        if solution.status == 'optimal':
            current_biomass = substrates_input[self.config['biomass_identifier']]
            biomass_growth_rate = solution.fluxes[self.config['biomass_reaction']]
            substrate_update[self.config['biomass_identifier']] = biomass_growth_rate * current_biomass * interval

            for substrate, reaction_id in self.config['substrate_update_reactions'].items():
                flux = solution.fluxes[reaction_id]
                substrate_update[substrate] = flux * current_biomass * interval
                # TODO -- assert not negative?
        else:
            # Handle non-optimal solutions if necessary
            # print('Non-optimal solution, skipping update')
            for substrate, reaction_id in self.config['substrate_update_reactions'].items():
                substrate_update[substrate] = 0

        return {
            'substrates': substrate_update,
        }


# register the process
core.register_process('DynamicFBA', DynamicFBA)


def dfba_config(
        model_file='textbook',
        kinetic_params=None,
        biomass_reaction='Biomass_Ecoli_core',
        substrate_update_reactions=None,
        biomass_identifier='biomass',
        bounds=None
):
    if substrate_update_reactions is None:
        substrate_update_reactions = {
            'glucose': 'EX_glc__D_e',
            'acetate': 'EX_ac_e'}
    if bounds is None:
        bounds = {
            'EX_o2_e': {'lower': -2, 'upper': None},
            'ATPM': {'lower': 1, 'upper': 1}}
    if kinetic_params is None:
        kinetic_params = {
            'glucose': (0.5, 1),
            'acetate': (0.5, 2)}
    return {
        'model_file': model_file,
        'kinetic_params': kinetic_params,
        'biomass_reaction': biomass_reaction,
        'substrate_update_reactions': substrate_update_reactions,
        'biomass_identifier': biomass_identifier,
        'bounds': bounds
    }


def run_dfba_spatial():
    n_bins = (5, 5)

    initial_glucose = np.random.uniform(low=0, high=20, size=n_bins)
    initial_acetate = np.random.uniform(low=0, high=0, size=n_bins)
    initial_biomass = np.random.uniform(low=0, high=0.1, size=n_bins)

    dfba_processes_dict = {}
    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            dfba_processes_dict[f'[{i},{j}]'] = {
                '_type': 'process',
                'address': 'local:DynamicFBA',
                'config': dfba_config(),
                'inputs': {
                    'substrates': {
                        'glucose': ['..', 'fields', 'glucose', i, j],
                        'acetate': ['..', 'fields', 'acetate', i, j],
                        'biomass': ['..', 'fields', 'biomass', i, j],
                    }
                },
                'outputs': {
                    'substrates': {
                        'glucose': ['..', 'fields', 'glucose', i, j],
                        'acetate': ['..', 'fields', 'acetate', i, j],
                        'biomass': ['..', 'fields', 'biomass', i, j]
                    }
                }
            }

    composite_state = {
        'fields': {
            '_type': 'map',
            '_value': {
                '_type': 'array',
                '_shape': n_bins,
                '_data': 'positive_float'
            },
            'glucose': initial_glucose,
            'acetate': initial_acetate,
            'biomass': initial_biomass,
        },
        'spatial_dfba': dfba_processes_dict
    }

    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'}
    }, core=core)

    # save the document
    sim.save(filename='spatial_dfba.json', outdir='out')

    # simulate
    sim.update({}, 60.0)

    # gather results
    dfba_results = sim.gather_results()
    # print(dfba_results)

    # plot timeseries
    plot_time_series(
        dfba_results,
        coordinates=[(0, 0), (1, 1), (2, 2)],
    )

    # make video
    plot_species_distributions_to_gif(
        dfba_results,
        out_dir='out',
        filename='out/dfba_results.gif',
        title='',
        skip_frames=1
    )


if __name__ == '__main__':
    run_dfba_spatial()
