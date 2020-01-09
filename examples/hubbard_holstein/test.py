import numpy
from mpi4py import MPI
import os
import pytest
from pauxy.analysis.extraction import (
        extract_mixed_estimates,
        extract_rdm
        )
from pauxy.qmc.calc import setup_calculation
from pauxy.qmc.afqmc import AFQMC
from pauxy.systems.generic import Generic
from pauxy.systems.ueg import UEG
from pauxy.utils.testing import generate_hamiltonian
from pauxy.trial_wavefunction.hartree_fock import HartreeFock

import h5py

def get_options(nsteps, nblocks):
    options = {
            'verbosity': 0,
            'get_sha1': False,
            "qmc": {
                "timestep": 0.005,
                "num_steps": nsteps,
                "blocks": nblocks,
                "rng_seed": 8,
                "nwalkers": 144,
                "pop_control_freq": 1
            },
            "model": {
                "name": "HubbardHolstein",
                "nx": 10,
                "ny": 1,
                "nup": 5,
                "ndown": 5,
                "t": 1.0,
                "U": 4.0,
                "w0": 0.8,
                "m": 1./0.2,
                "lambda": 0.8
            },
            "trial": {
                "name": "free_electron"
            },
            "estimates": {
                "mixed": {
                    "energy_eval_freq": 1,
                    "evaluate_holstein":True
                }
            },
            "propagator": {
                "hubbard_stratonovich": "discrete",
                "free_projecgtion": False,
                "update_trial": True
            },
            "walkers":{
                "population_control": "comb"
            }
        }
    return options


comm = MPI.COMM_WORLD
# Iteration      WeightFactor            Weight            ENumer            EDenom            ETotal            E1Body            E2Body           EHybrid           Overlap              Time
nblocks = 10
options = get_options(nsteps = 10, nblocks = nblocks) 
afqmc = AFQMC(comm=comm, options=options)
afqmc.run(comm=comm, verbose=1)

f = h5py.File("estimates.0.h5", "r")
Xdata = f['basic/X']
numbers = list(Xdata.keys())
nsamples = len(numbers)
Xavg = numpy.zeros(afqmc.system.nbasis)
for i in numbers:
    X = numpy.array(Xdata[i])
    Xavg += X
Xavg /= nsamples
print(Xavg)

rhodata = f['basic/rho']
numbers = list(rhodata.keys())
nsamples = len(numbers)
rhoavg = numpy.zeros((2,afqmc.system.nbasis))
for i in numbers:
    rho = numpy.array(rhodata[i])
    rhoavg += rho
rhoavg /= nsamples
print(rhoavg)



afqmc.finalise(verbose=0)
afqmc.estimators.estimators['mixed'].update(afqmc.system, afqmc.qmc,
                                            afqmc.trial, afqmc.psi, 0)