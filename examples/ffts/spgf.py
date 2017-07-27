#!/usr/bin/env python

# Calculate the UHF energy per site for various 4x4 lattice models by extracting
# the information from the trial wavefunction.

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import afqmcpy
import numpy
import copy
import pandas as pd
from cmath import exp
import time
import scipy.fftpack

# Fake some inputs
table = {
    "model": {
        "name": "Hubbard",
        "t": 1.0,
        "U": 4,
        "nx": 5,
        "ny": 1,
        "nup": 3,
        "ndown": 3,
    },
    "qmc_options": {
        "method": "CPMC",
        "dt": 0.05,
        "nsteps": 2,
        "nmeasure": 10,
        "nwalkers": 100,
        "npop_control": 10000,
        "temperature": 0.0,
        "hubbard_stratonovich": "discrete",
        "importance_sampling": True,
        "rng_seed": 7,
        "ueff": 4,
        "kinetic_kspace": True,
        "trial_wavefunction": "free_electron",
    }
}

state = afqmcpy.state.State(table['model'], table['qmc_options'])
print (state.trial.psi[0][:,0])
trial = afqmcpy.utils.fft_wavefunction(state.trial.psi[0], state.system.nx, state.system.ny,
                         state.system.nup, state.trial.psi[0].shape)
G2 = numpy.identity(state.system.nbasis)-afqmcpy.estimators.gab(state.trial.psi[0], state.trial.psi[0])
G = 0.5*(numpy.fft.fft(G2) + numpy.fft.fft(G2).conj().T)
for i in range(0,len(G)):
    for j in range(0,len(G)):
        if abs(G[i,j]) > 1e-10:
            print (i, j, G[i, j].real, G2[i,j])
