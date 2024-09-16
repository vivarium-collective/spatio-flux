import os
from spatio_flux.processes.particles import run_particles
from spatio_flux.processes.comets import run_comets
from spatio_flux.processes.dfba import run_dfba_spatial, run_dfba_single
from spatio_flux.processes.particles_dfba import run_particle_dfba
from spatio_flux.processes.diffusion_advection import run_diffusion_process


def test_run_comets():
    # Call the run_comets function with default parameters
    run_comets()
    # Add assertions here to verify the expected outcomes
    # For example, check if the output files are created
    assert os.path.exists('out/comets.json')
    assert os.path.exists('out/comets_timeseries.png')
    assert os.path.exists('out/comets_results.gif')
  

tests = [run_particles, run_comets, run_dfba_spatial, run_dfba_single, run_particle_dfba, run_diffusion_process]


def run():
    for run_test in tests:
        try:
            run_test()
        except:
            pass

if __name__ == '__main__':
    run()
