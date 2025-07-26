# @marimo
# # Particle-COMETS: From dynamic FBA to spatial simulations with particles
# Eran Agmon, University of Connecticut
# 
# This notebook is dedicated to developing a particle-COMETS simulation step by step using Vivarium.
# Each section of this notebook incrementally adds layers to our model, illustrating the different computational methods and how they integrate.
# @marimo
import numpy as np
from attr import dataclass
# spatio-flux's customized Vivarium
from spatio_flux import SpatioFluxVivarium, render_path
# @marimo
# ## Make a Vivarium
# view available types and processes
# @marimo
# make a fresh vivarium
vi = SpatioFluxVivarium()
# @marimo
# view the available types
vi.get_types()
# @marimo
# view the available processes
vi.get_processes()
# @marimo
# ## dFBA
# 
# Dynamic Flux Balance Analysis (dFBA) extends traditional Flux Balance Analysis (FBA) to model the dynamic behavior of metabolic networks over time, allowing for the simulation of growth and substrate consumption in a changing environment.
# @marimo
# inspect the DynamicFBA process config
vi.process_config('DynamicFBA')
# @marimo
# dfba_config  = vi.process_config('DynamicFBA', dataclass=True)  # TODO get dataclass to configure
dfba_config = {
    "model_file": "textbook",
    "kinetic_params": {
        "glucose": (0.5, 1),
        "acetate": (0.5, 2)},
    "substrate_update_reactions": {
        "glucose": "EX_glc__D_e",
        "acetate": "EX_ac_e"},
    "biomass_identifier": "biomass",
    "bounds": {
        "EX_o2_e": {"lower": -2, "upper": None},
        "ATPM": {"lower": 1, "upper": 1}}}
# @marimo
# make a fresh vivarium
v1 = SpatioFluxVivarium()

# add a dFBA process
v1.add_process(
    name="dFBA",
    process_id="DynamicFBA",
    config=dfba_config)
v1.diagram(dpi='70')
# @marimo
mol_ids = ["glucose", "acetate", "biomass"]

# add the molecular fields
for mol_id in mol_ids:
    v1.add_object(
        name=mol_id,
        path=['fields'],
        value=np.random.rand())

v1.connect_process(
    name="dFBA",
    inputs={"substrates": {
                mol_id: ['fields', mol_id]
                for mol_id in mol_ids}},
    outputs={"substrates": {
                mol_id: ['fields', mol_id]
                for mol_id in mol_ids}})

# add an emitter to save results
v1.add_emitter()
v1.diagram(dpi='70', show_values=True)
# @marimo
v1.set_value(path=['fields', 'glucose'], value=10)
v1.set_value(path=['fields', 'biomass'], value=0.1)
field = v1.get_value(['fields'])
print(field)
# @marimo
# save a file with the exact simulation state
v1.save(filename='dFBA_t0')
# @marimo
v1
# @marimo
vx = SpatioFluxVivarium(document='out/dFBA_t0.json')
vx.run(interval=10)
vx.diagram(dpi='70', show_values=True)
# @marimo
# run the simulation
v1.run(interval=60)
# @marimo
# view the timeseries
v1.get_timeseries(as_dataframe=True)
# @marimo
# plot the timeseries
v1.plot_timeseries(
    subplot_size=(8, 3),
    combined_vars=[
        [  # combine the variables into a single subplot
            '/fields/glucose',
            '/fields/acetate',
            '/fields/biomass'
        ]]
)
# @marimo
# ## Spatial dFBA
# @marimo
mol_ids = ["glucose", "acetate", "biomass"]
rows = 3
columns = 2

# make a fresh vivarium
v2 = SpatioFluxVivarium()
for mol_id in mol_ids:
    v2.add_object(
        name=mol_id,
        path=['fields'],
        value=np.random.rand(rows, columns))

# add a dynamic FBA process at every location
for i in range(rows):
    for j in range(columns):
        dfba_name = f"dFBA[{i},{j}]"
        v2.add_process(
            name=dfba_name,
            process_id="DynamicFBA",
            config=dfba_config)
        v2.connect_process(
            name=dfba_name,
            inputs={"substrates": {
                        mol_id: ['fields', mol_id, i, j]
                        for mol_id in mol_ids}},
            outputs={"substrates": {
                        mol_id: ['fields', mol_id, i, j]
                        for mol_id in mol_ids}})

# add an emitter to save results
v2.add_emitter()
v2.diagram(dpi='70')
# @marimo
# change some initial values
v2.merge_value(path=['fields', 'glucose', 0, 0], value=10.0)
v2.merge_value(path=['fields', 'biomass', 0, 0], value=0.1)
field = v2.get_value(['fields'])
print(field)
# @marimo
# run a simulation
v2.run(60)
# @marimo
# get a list of all the paths so they can be plotted together in a single graph
all_paths = [
    [render_path(['fields', mol_id, i, j]) for mol_id in mol_ids]
    for i in range(rows)
    for j in range(columns)]

# plot the timeseries
v2.plot_timeseries(
    subplot_size=(8, 3),
    combined_vars=all_paths)
# @marimo
v2.plot_snapshots()
# @marimo
# ## Diffusion/Advection
# 
# This approach models the physical processes of diffusion and advection in two dimensions, providing a way to simulate how substances spread and are transported across a spatial domain, essential for understanding patterns of concentration over time and space.
# @marimo
bounds = (10.0, 10.0)
n_bins = (10, 10)
mol_ids = [
    'glucose',
    'acetate',
    'biomass'
]
diffusion_rate = 1e-1
diffusion_dt = 1e-1
advection_coeffs = {'biomass': (0, -0.1)}

# make a fresh Vivarium
v3 = SpatioFluxVivarium()

# add fields for all the molecules
for mol_id in mol_ids:
    v3.add_object(
        name=mol_id,
        path=['fields'],
        value=np.random.rand(n_bins[0], n_bins[1]))

# add a spatial diffusion advection process
v3.add_process(
    name='diffusion_advection',
    process_id='DiffusionAdvection',
    config={
       'n_bins': n_bins,
       'bounds': bounds,
       'default_diffusion_rate': diffusion_rate,
       'default_diffusion_dt': diffusion_dt,
       'advection_coeffs': advection_coeffs},
    inputs={'fields': ['fields']},
    outputs={'fields': ['fields']})

# add an emitter to save results
v3.add_emitter()
v3.diagram(dpi='70')
# @marimo
v3.run(60)
# @marimo
v3.show_video()
# @marimo
# ## COMETS
# 
# COMETS (Computation Of Microbial Ecosystems in Time and Space) combines dynamic FBA with spatially resolved physical processes (like diffusion and advection) to simulate the growth, metabolism, and interaction of microbial communities within a structured two-dimensional environment, capturing both biological and physical complexities.
# @marimo
bounds = (20.0, 10.0)  # Bounds of the environment
n_bins = (20, 10)
mol_ids = ['glucose', 'acetate', 'biomass']
diffusion_rate = 1e-1
diffusion_dt = 1e-1
advection_coeffs = {'biomass': (0, 0.1)}

# make a fresh vivarium
v4 = SpatioFluxVivarium()

# initialize the molecular fields
max_glc = 10
glc_field = np.random.rand(n_bins[0], n_bins[1]) * max_glc
acetate_field = np.zeros((n_bins[0], n_bins[1]))
biomass_field = np.zeros((n_bins[0], n_bins[1]))
biomass_field[0:int(1*n_bins[0]/5), int(2*n_bins[1]/5):int(3*n_bins[1]/5)] = 0.1  # place some biomass

v4.add_object(name='glucose', path=['fields'], value=glc_field)
v4.add_object(name='biomass', path=['fields'], value=biomass_field)
v4.add_object(name='acetate', path=['fields'], value=acetate_field)

# add a diffusion/advection process
v4.add_process(
    name='diffusion_advection',
    process_id='DiffusionAdvection',
    config={
       'n_bins': n_bins,
       'bounds': bounds,
       'default_diffusion_rate': diffusion_rate,
       'default_diffusion_dt': diffusion_dt,
       'advection_coeffs': advection_coeffs    },
    inputs={'fields': ['fields']},
    outputs={'fields': ['fields']})

# add a dynamic FBA process at every location
for i in range(n_bins[0]):
    for j in range(n_bins[1]):
        dfba_name = f"dFBA[{i},{j}]"
        v4.add_process(
            name=dfba_name,
            process_id="DynamicFBA",
            config=dfba_config        )
        v4.connect_process(
            name=dfba_name,
            inputs={"substrates": {
                        mol_id: ['fields', mol_id, i, j]
                        for mol_id in mol_ids}            },
            outputs={"substrates": {
                        mol_id: ['fields', mol_id, i, j]
                        for mol_id in mol_ids}})

# add an emitter to save results
v4.add_emitter()
v4.diagram(dpi='70',
    remove_nodes=[f"/dFBA[{i},{j}]" for i in range(n_bins[0]-1) for j in range(n_bins[1])]
           )
# @marimo
v4.run(60)
# @marimo
v4.show_video()
# @marimo
v4.plot_timeseries(
    subplot_size=(8, 3),
    query=[
        '/fields/glucose/0/0',
        '/fields/acetate/0/0',
        '/fields/biomass/0/0'],
    combined_vars=[[
        '/fields/glucose/0/0',
        '/fields/acetate/0/0',
        '/fields/biomass/0/0']
    ])
# @marimo
# ## Particles
# @marimo
# import numpy as np
# from spatio_flux import SpatioFluxVivarium
# @marimo
v5 = SpatioFluxVivarium()
v5.process_config('Particles')
# @marimo
bounds = (10.0, 20.0)  # Bounds of the environment
n_bins = (20, 40)  # Number of bins in the x and y directions

# particle movement
v5.add_process(
    name='particle_movement',
    process_id='Particles',
    config={
        'n_bins': n_bins,
        'bounds': bounds,
        'diffusion_rate': 0.1,
        'advection_rate': (0, -0.1),
        'add_probability': 0.3,
        'boundary_to_add': ['top']},
    inputs={'fields': ['fields'],
            'particles': ['particles']},
    outputs={'fields': ['fields'],
             'particles': ['particles']})

v5.initialize_process(
    path='particle_movement',
    config={'n_particles': 2})

v5.add_emitter()
v5.diagram(dpi='70')
# @marimo
v5.save('particle_movement')
# @marimo
v5.run(100)
v5_results = v5.get_results()
# @marimo
v5.plot_particles_snapshots(skip_frames=3)
# @marimo
v5.diagram(dpi='70')
# @marimo
v5.save(filename='v5_post_run.json', outdir='out')
# @marimo
# ## Minimal particle with diffusing fields
# @marimo
import numpy as np
from spatio_flux import SpatioFluxVivarium
from spatio_flux.processes.particles import get_minimal_particle_composition
# @marimo
bounds = (10.0, 20.0)  # Bounds of the environment
n_bins = (10, 20)  # Number of bins in the x and y directions
mol_ids = ['glucose', 'acetate', 'biomass']
diffusion_rate = 1e-1
diffusion_dt = 1e-1
advection_coeffs = {'biomass': (0, 0.1)}

v6 = SpatioFluxVivarium()

# make two fields
v6.add_object(name='glucose',path=['fields'], value=np.ones((n_bins[0], n_bins[1])))
v6.add_object(name='acetate', path=['fields'], value=np.zeros((n_bins[0], n_bins[1])))
# diffusion advection process
v6.add_process(
    name='diffusion_advection',
    process_id='DiffusionAdvection',
    config={
       'n_bins': n_bins,
       'bounds': bounds,
       'default_diffusion_rate': diffusion_rate,
       'default_diffusion_dt': diffusion_dt,
       'advection_coeffs': advection_coeffs},
    inputs={'fields': ['fields']},
    outputs={'fields': ['fields']})
# particle movement process
v6.add_process(
    name='particle_movement',
    process_id='Particles',
    config={
        'n_bins': n_bins,
        'bounds': bounds,
        'diffusion_rate': 0.1,
        'advection_rate': (0, -0.1),
        'add_probability': 0.3,
        'boundary_to_add': ['top']},
    inputs={'fields': ['fields'],
            'particles': ['particles']},
    outputs={'fields': ['fields'],
             'particles': ['particles']})

# add a process into each particle
minimal_particle_config = {
    'reactions': {
        'grow': {
            'glucose': {
                'vmax': 0.0001,
                'kcat': 0.01,
                'role': 'reactant'},
            'acetate': {
                'vmax': 0.00001,
                'kcat': 0.001,
                'role': 'product'}}}}
particle_schema = get_minimal_particle_composition(v6.core, minimal_particle_config)
v6.merge_schema(path=['particles'], schema=particle_schema['particles'])

# add particles to the initial state
v6.initialize_process(
    path='particle_movement',
    config={'n_particles': 1})

v6.diagram(dpi='70')
# @marimo
v6.run(200)
# @marimo
v6.plot_particles_snapshots(skip_frames=4)
# @marimo
# ## Particle-COMETS with dFBA particles
# @marimo
vtest = SpatioFluxVivarium()

default_dfba = vtest.dataclass_config('DynamicFBA')
# @marimo
vtest.get_processes()
# @marimo






from spatio_flux.processes.particles import get_dfba_particle_composition

# TODO -- method to get config from Vivarium.
dfba_config = {
    "model_file": "textbook",
    "kinetic_params": {
        "glucose": (0.5, 1),
        "acetate": (0.5, 2)},
    "substrate_update_reactions": {
        "glucose": "EX_glc__D_e",
        "acetate": "EX_ac_e"},
    "biomass_identifier": "biomass",
    "bounds": {
        "EX_o2_e": {"lower": -2, "upper": None},
        "ATPM": {"lower": 1, "upper": 1}
    }}
# @marimo
bounds = (10.0, 20.0)  # Bounds of the environment
n_bins = (2, 4)  # Number of bins in the x and y directions
mol_ids = ['glucose', 'acetate', 'biomass']
diffusion_rate = 1e-1
diffusion_dt = 1e-1
advection_coeffs = {'biomass': (0, 0.1)}

# make a fresh vivarium
v7 = SpatioFluxVivarium()
# make the fields
biomass_field = np.zeros((n_bins[0], n_bins[1]))
biomass_field[0:int(1*n_bins[0]/5), int(2*n_bins[1]/5):int(3*n_bins[1]/5)] = 0.1  # place some biomass
v7.add_object(name='glucose', path=['fields'], value=np.ones((n_bins[0], n_bins[1])))
v7.add_object(name='acetate', path=['fields'], value=np.zeros((n_bins[0], n_bins[1])))
v7.add_object(name='biomass', path=['fields'], value=biomass_field)
# diffusion advection process
v7.add_process(
    name='diffusion',
    process_id='DiffusionAdvection',
    config={
       'n_bins': n_bins,
       'bounds': bounds,
       'default_diffusion_rate': diffusion_rate,
       'default_diffusion_dt': diffusion_dt,
       'advection_coeffs': advection_coeffs},
    inputs={'fields': ['fields']},
    outputs={'fields': ['fields']})
# particle movement process
v7.add_process(
    name='particle_movement',
    process_id='Particles',
    config={
        'n_bins': n_bins,
        'bounds': bounds,
        'diffusion_rate': 0.1,
        'advection_rate': (0, -0.1),
        'add_probability': 0.3,
        'boundary_to_add': ['top']},
    inputs={'fields': ['fields'],
            'particles': ['particles']},
    outputs={'fields': ['fields'],
             'particles': ['particles']})
# add dynamic FBA process at every location
for i in range(n_bins[0]):
    for j in range(n_bins[1]):
        dfba_name = f"dFBA[{i},{j}]"
        v7.add_process(
            name=dfba_name,
            process_id="DynamicFBA",
            config=dfba_config)
        v7.connect_process(
            name=dfba_name,
            inputs={
                "substrates": {
                    mol_id: ['fields', mol_id, i, j]
                    for mol_id in mol_ids}},
            outputs={
                "substrates": {
                    mol_id: ['fields', mol_id, i, j]
                    for mol_id in mol_ids}})
# add a process into each particle
minimal_particle_config = {
    'reactions': {
        'grow': {
            'glucose': {
                'vmax': 0.01,
                'kcat': 0.01,
                'role': 'reactant'},
            'acetate': {
                'vmax': 0.001,
                'kcat': 0.001,
                'role': 'product'}}}}
# set the dfba particle process into the particle schema
particle_schema = get_dfba_particle_composition()
v7.merge_schema(path=['particles'], schema=particle_schema['particles'])
# add particles to the initial state
v7.initialize_process(
    path='particle_movement',
    config={'n_particles': 1})
# diagram
v7.diagram(dpi='70')
# @marimo
v7.composite.composition['particles']
# @marimo
v7.composite.state.keys()
# @marimo
# v7.run(100)
# @marimo
# v7.plot_particles_snapshots(skip_frames=4)
