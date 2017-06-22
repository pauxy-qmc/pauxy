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
        "rng_seed": 7,
        "ueff": 4,
        "trial_wavefunction": "UHF",
    }
}

print ("{:7} {:7} {:>4}  {:<13}".format("lattice", "filling", "U", "Energy"))
for n in range(2, 9):
    table['model']['nup'] = n
    table['model']['ndown'] = n
    for u in [2, 4, 8]:
        table['qmc_options']['ueff'] = u
        state = afqmcpy.state.State(table['model'], table['qmc_options'])
        emin = state.trial.emin/state.system.nbasis
        lattice = "(%s,%s)"%(state.system.nx, state.system.ny)
        filling = "(%s,%s)"%(n,n)
        print ("{:>7s} {:>7s} {:>4d} {: .10}".format(lattice, filling, u, emin))
