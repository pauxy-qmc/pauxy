import numpy
from mpi4py import MPI
import os
import cmath
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
from pauxy.trial_wavefunction.harmonic_oscillator import HarmonicOscillator

import h5py

def get_options(nsteps, nblocks):
    options = {
            'verbosity': 1,
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
                "m": 1./0.8,
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
                "update_trial": False
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

if comm.rank == 0:
    f = h5py.File("estimates.0.h5", "r")
    Xdata = f['basic/X']
    numbers = list(Xdata.keys())
    nsamples = len(numbers)
    Xavg = numpy.zeros(afqmc.system.nbasis)
    for i in numbers:
        X = numpy.array(Xdata[i])
        Xavg += X
    Xavg /= nsamples
    print(afqmc.propagators.boson_trial.xavg)
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

    f.close()

    nX = numpy.array([numpy.diag(Xavg), numpy.diag(Xavg)])
    V = - afqmc.system.g * cmath.sqrt(afqmc.system.m * afqmc.system.w0 * 2.0) * nX
    afqmc.trial.update_wfn(afqmc.system, V, verbose=0) # trial update
else:
    afqmc.trial = None
    rhoavg = None

afqmc.trial = comm.bcast(afqmc.trial, root=0)

rhoavg = comm.bcast(rhoavg, root=0)

for w in afqmc.psi.walkers:
    w.inverse_overlap(afqmc.trial)
    otold = w.ot
    otnew= w.calc_otrial(afqmc.trial)
    oratio_extra = (otnew / otold).real
    phase = cmath.phase(oratio_extra)
    if abs(phase) < 0.5*cmath.pi:
        w.weight = w.weight * oratio_extra
        w.ot *= oratio_extra 
    else:
        w.weight = 0.0

for w in afqmc.psi.walkers:
    phiold = afqmc.propagators.boson_trial.value(w.X) # phi with the previous trial
    shift = numpy.sqrt(afqmc.system.w0*2.0 * afqmc.system.m) * afqmc.system.g * (rhoavg[0]+ rhoavg[1]) / (afqmc.system.m * afqmc.system.w0**2)
    afqmc.propagators.boson_trial = HarmonicOscillator(m = afqmc.system.m, w = afqmc.system.w0, order = 0, shift=shift) # trial updaate
    phinew = afqmc.propagators.boson_trial.value(w.X) # phi with a new trial
    oratio_extra = phinew / phiold
    w.weight *= oratio_extra


afqmc.propagators.update_trial = False

afqmc.run(comm=comm, verbose=1)

afqmc.finalise(verbose=0)
