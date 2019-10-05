import numpy
from mpi4py import MPI
import os
import pytest
from pauxy.analysis.extraction import extract_mixed_estimates
from pauxy.qmc.calc import setup_calculation
from pauxy.qmc.afqmc import AFQMC
from pauxy.systems.generic import Generic
from pauxy.systems.ueg import UEG
from pauxy.utils.testing import generate_hamiltonian
from pauxy.trial_wavefunction.hartree_fock import HartreeFock

def test_constructor():
    options = {
            'verbosity': 0,
            'get_sha1': False,
            'qmc': {
                'timestep': 0.01,
                'num_steps': 10,
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
    afqmc.finalise(verbose=0)
    assert afqmc.trial.energy.real == pytest.approx(1.7796083856572522)


def test_ueg():
    options = {
            'verbosity': 0,
            'get_sha1': False,
            'qmc': {
                'timestep': 0.01,
                'num_steps': 10,
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
    afqmc.estimators.estimators['mixed'].update(afqmc.system, afqmc.qmc,
                                                afqmc.trial, afqmc.psi, 0)
    enum = afqmc.estimators.estimators['mixed'].names
    numer = afqmc.estimators.estimators['mixed'].estimates[enum.enumer]
    assert numer == pytest.approx(210.6272953292560)
    denom = afqmc.estimators.estimators['mixed'].estimates[enum.edenom]
    assert denom == pytest.approx(120)
    weight = afqmc.estimators.estimators['mixed'].estimates[enum.weight]
    assert weight == pytest.approx(118.243099094879)
    ehy = afqmc.psi.walkers[0].hybrid_energy
    assert ehy.real == pytest.approx(1.1153859035083666)
    assert ehy.imag == pytest.approx(0.17265962035671692)

def test_hubbard():
    options = {
            'verbosity': 0,
            'get_sha1': False,
            'qmc': {
                'timestep': 0.01,
                'print_freq': 10,
                'num_steps': 100,
                'rng_seed': 8,
            },
            'model': {
                'name': "Hubbard",
                'nx': 4,
                'ny': 4,
                'nup': 7,
                "U": 4,
                'ndown': 7,
            },
            'trial': {
                'name': 'UHF'
            },
            'propagator': {
                'hubbard_stratonovich': 'discrete'
            }
        }
    comm = MPI.COMM_WORLD
    afqmc = AFQMC(comm=comm, options=options)
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.estimators['mixed'].update(afqmc.system, afqmc.qmc,
                                                afqmc.trial, afqmc.psi, 0)
    enum = afqmc.estimators.estimators['mixed'].names
    numer = afqmc.estimators.estimators['mixed'].estimates[enum.enumer]
    denom = afqmc.estimators.estimators['mixed'].estimates[enum.edenom]
    weight = afqmc.estimators.estimators['mixed'].estimates[enum.weight]
    assert numer.real == pytest.approx(-152.68468568462666)
    data = extract_mixed_estimates('estimates.0.h5')
    assert numpy.mean(data.ETotal.values.real) == pytest.approx(-14.974806533852874)

def test_hubbard_complex():
    options = {
            'verbosity': 0,
            'get_sha1': False,
            'qmc': {
                'timestep': 0.01,
                'print_freq': 10,
                'num_steps': 100,
                'rng_seed': 8,
            },
            'model': {
                'name': "Hubbard",
                'nx': 4,
                'ny': 4,
                'nup': 7,
                "U": 4,
                'ndown': 7,
            },
            'trial': {
                'name': 'UHF'
            },
            'propagator': {
                'hubbard_stratonovich': 'continuous'
            }
        }
    comm = MPI.COMM_WORLD
    afqmc = AFQMC(comm=comm, options=options)
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.estimators['mixed'].update(afqmc.system, afqmc.qmc,
                                                afqmc.trial, afqmc.psi, 0)
    enum = afqmc.estimators.estimators['mixed'].names
    numer = afqmc.estimators.estimators['mixed'].estimates[enum.enumer]
    denom = afqmc.estimators.estimators['mixed'].estimates[enum.edenom]
    weight = afqmc.estimators.estimators['mixed'].estimates[enum.weight]
    assert numer == pytest.approx(-152.91937839611)
    data = extract_mixed_estimates('estimates.0.h5')
    assert numpy.mean(data.ETotal.values.real) == pytest.approx(-15.14323385684513)

def test_generic():
    nmo = 11
    nelec = (3,3)
    options = {
            'verbosity': 0,
            'get_sha1': False,
            'qmc': {
                'timestep': 0.005,
                'print_freq': 10,
                'num_steps': 100,
                'rng_seed': 8,
            },
            'trial': {
                'name': 'MultiSlater'
            }
        }
    numpy.random.seed(7)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys_opts = {'sparse': False}
    sys = Generic(nelec=nelec, h1e=h1e, chol=chol, ecore=enuc, inputs=sys_opts)
    comm = MPI.COMM_WORLD
    afqmc = AFQMC(comm=comm, system=sys, options=options)
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.estimators['mixed'].update(afqmc.system, afqmc.qmc,
                                                afqmc.trial, afqmc.psi, 0)
    enum = afqmc.estimators.estimators['mixed'].names
    numer = afqmc.estimators.estimators['mixed'].estimates[enum.enumer]
    denom = afqmc.estimators.estimators['mixed'].estimates[enum.edenom]
    weight = afqmc.estimators.estimators['mixed'].estimates[enum.weight]
    assert numer.real == pytest.approx(3.8763193646854273)
    data = extract_mixed_estimates('estimates.0.h5')
    assert numpy.mean(data.ETotal.values.real) == pytest.approx(1.5485077038208)

def teardown_module(self):
    cwd = os.getcwd()
    files = ['estimates.0.h5']
    for f in files:
        try:
            os.remove(cwd+'/'+f)
        except OSError:
            pass
