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
        "nx": 4,
        "ny": 4,
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
trial = copy.deepcopy(state.trial.psi)
start = time.time()
trial[0] = state.propagators.bt2.dot(trial[0])
trial[1] = state.propagators.bt2.dot(trial[1])
# print (trial[1])
print (time.time() - start)
start = time.time()
afqmcpy.propagation.kinetic_kspace(state, trial)
print (time.time() - start)
