#!/usr/bin/env python

# Calculate the UHF energy per site for various 4x4 lattice models by extracting
# the information from the trial wavefunction.

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import afqmcpy
import numpy
import pandas as pd
from cmath import exp

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
        "trial_wavefunction": "free_electron",
    }
}

def spgf(kp, eks, kc, tau, nup, rij, nsites):
    # Assume nup = ndown
    unocc = len(kp) - nup
    spgf = sum(exp(1j*(kc*k).dot(rij)-tau*ek) for (k, ek) in zip(kp[nup:], eks[nup:]))

    return spgf.real / nsites

state = afqmcpy.state.State(table['model'], table['qmc_options'])
rij = numpy.array([0])
taus = numpy.linspace(0,10,100)
g = [spgf(state.system.kpoints, state.system.eks, state.system.kc, t,
        state.system.nup, rij, state.system.nx*state.system.ny) for t in taus]

print (pd.DataFrame({'tau': taus, 'g00': g}, columns=['tau',
'g00']).to_string(index=False))
