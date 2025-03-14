"""
Particles process
=================

A process for simulating the motion of particles in a 2D environment.
"""
import uuid
import numpy as np
from process_bigraph import Process, Composite, default
from bigraph_viz import plot_bigraph
from spatio_flux.viz.plot import plot_species_distributions_with_particles_to_gif, plot_particles


class Particles(Process):
    config_schema = {
        # environment size and resolution
        'bounds': 'tuple[float,float]',
        'n_bins': 'tuple[integer,integer]',

        # particle movement
        'diffusion_rate': {'_type': 'float', '_default': 1e-1},
        'advection_rate': {'_type': 'tuple[float,float]', '_default': (0, 0)},

        # adding/removing particles at boundaries
        'add_probability': {'_type': 'float', '_default': 0.0},  # TODO -- make probability type
        'boundary_to_add': {'_type': 'list', '_default': ['left', 'right']},  # which boundaries to add particles
        'boundary_to_remove': {'_type': 'list', '_default': ['left', 'right', 'top', 'bottom']},
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.env_size = (
            (0, self.config['bounds'][0]),
            (0, self.config['bounds'][1])
        )

    def inputs(self):
        return {
            'particles': 'map[particle]',
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'positive_float'
                },
            }
        }

    def outputs(self):
        return {
            'particles': 'map[particle]',
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'positive_float'
                },
            }
        }

    @staticmethod
    def initialize_particles(
            n_particles,
            bounds,
            fields,
            size_range=(10, 100),
    ):
        """
        Initialize particle positions for multiple species.
        """
        mol_ids = fields.keys()

        # get n_bins from the shape of the first field array
        n_bins = fields[list(fields.keys())[0]].shape

        # advection_rates = advection_rates or [(0.0, 0.0) for _ in range(len(n_particles_per_species))]
        particles = {}
        for _ in range(n_particles):
            id = str(uuid.uuid4())
            position = tuple(np.random.uniform(low=[0, 0],high=[bounds[0], bounds[1]],size=2))
            size = np.random.uniform(size_range[0], size_range[1])

            x, y = Particles.get_bin_position(position, n_bins, ((0.0, bounds[0]), (0.0, bounds[1])))
            # TODO update local and exchange values
            local = Particles.get_local_field_values(fields, column=x, row=y)
            exchanges = {f: 0.0 for f in mol_ids}  # TODO exchange rates

            particles[id] = {
                # 'id': str(uuid.uuid4()),
                'position': position,
                'size': size,
                'local': local,
                'exchange': exchanges
            }
            # particles.append(particle)

        return particles

    def update(self, state, interval):
        particles = state['particles']
        fields = state['fields']  # Retrieve the fields

        new_particles = {'_remove': [], '_add': {}}
        new_fields = {
            mol_id: np.zeros_like(field)
            for mol_id, field in fields.items()}

        for particle_id, particle in particles.items():
            updated_particle = {'exchange': {}}

            # Apply diffusion and advection
            dx, dy = np.random.normal(0, self.config['diffusion_rate'], 2) + self.config['advection_rate']

            new_x_position = particle['position'][0] + dx
            new_y_position = particle['position'][1] + dy

            # Check and remove particles if they hit specified boundaries
            if self.check_boundary_hit(new_x_position, new_y_position):
                new_particles['_remove'].append(particle_id)
                continue  # Remove particle if it hits a boundary

            new_position = (new_x_position, new_y_position)
            updated_particle['position'] = (dx, dy) # new_position

            # Retrieve local field concentration for each particle
            x, y = self.get_bin_position(new_position, self.config['n_bins'], self.env_size)

            # Update local environment values for each particle
            updated_particle['local'] = self.get_local_field_values(fields, column=x, row=y)

            # Apply exchanges and reset
            exchange = particle['exchange']
            for mol_id, exchange_rate in exchange.items():
                new_fields[mol_id][x, y] += exchange_rate
                updated_particle['exchange'][mol_id] = 0.0

            new_particles[particle_id] = updated_particle

        # Probabilistically add new particles at user-defined boundaries
        for boundary in self.config['boundary_to_add']:
            if np.random.rand() < self.config['add_probability']:
                # TODO -- reuse function for initializing particles
                position = self.get_boundary_position(boundary)
                x, y = self.get_bin_position(position, self.config['n_bins'], self.env_size)
                local_field_concentrations = self.get_local_field_values(fields, column=x, row=y)
                id = str(uuid.uuid4())
                new_particle = {
                    'id': id,
                    'position': position,
                    'size': np.random.uniform(10, 100),  # Random size for new particles
                    'local': local_field_concentrations,
                    'exchange': {f: 0.0 for f in fields.keys()}  # TODO -- add exchange
                }
                new_particles['_add'][id] = new_particle

        return {
            'particles': new_particles,
            'fields': new_fields
        }

    @staticmethod
    def get_bin_position(position, n_bins, env_size):
        x, y = position
        x_bins, y_bins = n_bins #self.config['n_bins']
        x_min, x_max = env_size[0]
        y_min, y_max = env_size[1]

        # Convert the particle's (x, y) position to the corresponding bin in the 2D grid
        x_bin = int((x - x_min) / (x_max - x_min) * x_bins)
        y_bin = int((y - y_min) / (y_max - y_min) * y_bins)

        # Correct any potential out-of-bound indices
        x_bin = min(max(x_bin, 0), x_bins - 1)
        y_bin = min(max(y_bin, 0), y_bins - 1)

        return x_bin, y_bin

    @staticmethod
    def get_local_field_values(fields, column, row):
        """
        Retrieve local field values for a particle based on its position.

        Parameters:
        - fields: dict of 2D numpy arrays representing fields, keyed by molecule ID.
        - position: Tuple (x, y) representing the particle's position.

        Returns:
        - local_values: dict of field concentrations at the particle's location, keyed by molecule ID.
        """
        local_values = {}
        for mol_id, field in fields.items():
            local_values[mol_id] = field[column, row]

        return local_values

    def check_boundary_hit(self, new_x_position, new_y_position):
        # Check if the particle hits any of the boundaries to be removed
        if 'left' in self.config['boundary_to_remove'] and new_x_position < self.env_size[0][0]:
            return True
        if 'right' in self.config['boundary_to_remove'] and new_x_position > self.env_size[0][1]:
            return True
        if 'top' in self.config['boundary_to_remove'] and new_y_position > self.env_size[1][1]:
            return True
        if 'bottom' in self.config['boundary_to_remove'] and new_y_position < self.env_size[1][0]:
            return True
        return False

    def get_boundary_position(self, boundary):
        if boundary == 'left':
            return self.env_size[0][0], np.random.uniform(*self.env_size[1])
        elif boundary == 'right':
            return self.env_size[0][1], np.random.uniform(*self.env_size[1])
        elif boundary == 'top':
            return np.random.uniform(*self.env_size[0]), self.env_size[1][1]
        elif boundary == 'bottom':
            return np.random.uniform(*self.env_size[0]), self.env_size[1][0]


class MinimalParticle(Process):
    config_schema = {
        'kinetics': {
            '_type': 'map',
            '_value': {
                'vmax': default('float', 0.1),
                'km': default('float', 1.0)},
            '_default': {
                'biomass': {
                    'vmax': -0.1,
                    'km': 0.1},
                'detritus': {
                    'vmax': 0.1,
                    'km': 0.1}}}}


    # config_schema = {
    #     'field_interactions': {
    #         '_type': 'map',
    #         '_value': {
    #             'vmax': default('float', 0.1),
    #             'Km': default('float', 1.0),
    #             'interaction_type': default('enum[uptake,secretion]', 'uptake')},
    #         '_default': {
    #             'biomass': {
    #                 'vmax': 0.1,
    #                 'Km': 1.0,
    #                 'interaction_type': 'uptake'},
    #             'detritus': {
    #                 'vmax': -0.1,
    #                 'Km': 1.0,
    #                 'interaction_type': 'secretion'}}}}


    def inputs(self):
        return {
            'substrates': 'map[float]'
            # 'substrates': 'map[positive_float]'
        }


    def outputs(self):
        return {
            'substrates': 'map[float]'
        }


    def update(self, state, interval):
        substrates = state['substrates']
        exchanges = {}

        for substrate, kinetics in self.config['kinetics'].items():
            if substrate in substrates:
                exchanges[substrate] = (kinetics['vmax'] * substrates[substrate]) / (kinetics['km'] + substrates[substrate])

        return {
            'substrates': exchanges}


    def large_update(self, state, interval):
        substrates_input = state['substrates']
        exchanges = {}

        # Helper functions for interaction types
        def michaelis_menten(uptake_value, vmax, Km):
            """Michaelis-Menten rate law for uptake."""
            return (vmax * uptake_value) / (Km + uptake_value) if Km + uptake_value > 0 else 0

        def calculate_uptake(field_value, vmax, Km):
            """Calculate the net uptake value."""
            uptake_rate = michaelis_menten(field_value, vmax, Km)
            absorbed_value = min(uptake_rate, field_value)  # Limit to available substrate
            return -absorbed_value  # Negative for uptake

        def calculate_secretion(vmax):
            """Calculate the net secretion value."""
            return vmax  # Secretion value is directly proportional to vmax

        # Process each field interaction
        for field, interaction_params in self.config['field_interactions'].items():
            local_field_value = substrates_input.get(field, 0)
            vmax = interaction_params['vmax']
            Km = interaction_params.get('Km', 1)  # Default Km to 1 if not specified
            interaction_type = interaction_params.get('interaction_type', 'uptake')  # Default to 'uptake'

            if interaction_type == 'uptake' and local_field_value > 0:
                exchanges[field] = calculate_uptake(local_field_value, vmax, Km)
            elif interaction_type == 'secretion':
                exchanges[field] = calculate_secretion(vmax)
            else:
                exchanges[field] = 0  # No interaction by default

        # Return updated substrates
        return {
            'substrates': exchanges
        }


# Helper functions to get specs and states
def get_particles_spec(
        n_bins=(20, 20),
        bounds=(10.0, 10.0),
        diffusion_rate=1e-1,
        advection_rate=(0, 0),
        add_probability=0.0,
        boundary_to_add=['top'],
):
    config = locals()
    # Remove any key-value pair where the value is None
    config = {key: value for key, value in config.items() if value is not None}

    return {
        '_type': 'process',
        'address': 'local:Particles',
        'config': config,
        'inputs': {
            'particles': ['particles'],
            'fields': ['fields']},
        'outputs': {
            'particles': ['particles'],
            'fields': ['fields']}}


def get_particles_state(
        n_bins=(20, 20),
        bounds=(10.0, 10.0),
        n_particles=15,
        diffusion_rate=0.1,
        advection_rate=(0, -0.1),
        boundary_to_add=None,
        add_probability=0.4,
        field_interactions=None,
        initial_min_max=None,
        core=None,
):
    if boundary_to_add is None:
        boundary_to_add = ['top']

    if field_interactions is None:
        field_interactions = {
            'biomass': {'vmax': 0.1, 'Km': 1.0, 'interaction_type': 'uptake'},
            'detritus': {'vmax': -0.1, 'Km': 1.0, 'interaction_type': 'secretion'},
        }

    if initial_min_max is None:
        initial_min_max = {
            'biomass': (0.1, 0.2),
            'detritus': (0, 0),
        }

    # initialize fields
    fields = {}
    for field, minmax in initial_min_max.items():
        fields[field] = np.random.uniform(low=minmax[0], high=minmax[1], size=n_bins)

    # initialize particles
    particles = Particles.initialize_particles(
        n_particles=n_particles,
        bounds=bounds,
        fields=fields)

    return {
        'fields': fields,
        'particles': particles,
        'particles_process': get_particles_spec(
            n_bins=n_bins,
            bounds=bounds,
            diffusion_rate=diffusion_rate,
            advection_rate=advection_rate,
            add_probability=add_probability,
            boundary_to_add=boundary_to_add,
        )
    }
