#!/usr/bin/env python

# Calculate the UHF energy per site for various 4x4 lattice models by extracting
# the information from the trial wavefunction.

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import afqmcpy

table = {
    "model": {
        "name": "Hubbard",
        "t": 1.0,
        "U": 4,
        "nx": 4,
        "ny": 4,
        "ktwist": [0, 0],
        "nup": 7,
        "ndown": 7,
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
        "rng_seed": 7
    },
    "trial_wavefunction": {
        "name": "multi_determinant",
        "type": "GHF",
        "orbital_file": "GHF_4x4_7u7d_u4.0_ndet001_orb.dat",
        "coefficients_file": "GHF_4x4_7u7d_u4.0_ndet001_coeff.dat",
        "ndets": 1
    }
}
state = afqmcpy.state.State(table['model'], table['qmc_options'],
                            table['trial_wavefunction'])
print (state.trial.emin)
