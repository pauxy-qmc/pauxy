import pytest
from mpi4py import MPI
import os
from pauxy.qmc.calc import setup_calculation
from pauxy.qmc.afqmc import AFQMC
from pauxy.systems.generic import Generic
from pauxy.systems.ueg import UEG
from pauxy.trial_wavefunction.hartree_fock import HartreeFock

def test_ueg():
    options = {
            'verbosity': 0,
            'get_sha1': False,
            'qmc': {
                'timestep': 0.01,
                'num_steps': 10,
                'num_blocks': 10,
                'rng_seed': 8,
            },
            'model': {
                'name': "UEG",
                'rs': 2.44,
                'ecut': 4,
                'nup': 7,
                'ndown': 7,
            },
            'trial': {
                'name': 'hartree_fock'
            }
        }
    comm = MPI.COMM_WORLD
    # FDM: Fix cython issue.
    afqmc = AFQMC(comm=comm, options=options)
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    ehy = afqmc.psi.walkers[0].hybrid_energy
    assert ehy.real == pytest.approx(1.1153859035083666)
    assert ehy.imag == pytest.approx(0.17265962035671692)

def test_constructor():
    options = {
            'verbosity': 0,
            'get_sha1': False,
            'qmc': {
                'timestep': 0.01,
                'num_steps': 10,
                'num_blocks': 10,
                'rng_seed': 8,
            },
        }
    model = {
        'name': "UEG",
        'rs': 2.44,
        'ecut': 4,
        'nup': 7,
        'ndown': 7,
        }
    system = UEG(model)
    trial = HartreeFock(system, True, {})
    comm = MPI.COMM_WORLD
    afqmc = AFQMC(comm=comm, options=options, system=system, trial=trial)
    assert afqmc.trial.energy.real == pytest.approx(1.7796083856572522)

def teardown_module(self):
    cwd = os.getcwd()
    files = ['estimates.0.h5']
    for f in files:
        try:
            os.remove(cwd+'/'+f)
        except OSError:
            pass
