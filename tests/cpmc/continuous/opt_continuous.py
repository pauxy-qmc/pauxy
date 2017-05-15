#!/usr/bin/env python

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import afqmcpy

model = {
    'name': 'Hubbard',
    't': 1.0,
    'U': 4,
    'nx': 4,
    'ny': 1,
    'nup': 1,
    'ndown': 1,
}
qmc_options = {
    'method': 'CPMC',
    'dt': 0.01,
    'nsteps': 100,
    'nmeasure': 1,
    'npop_control': 100,
    'nwalkers': 100,
    'rng_seed': 7,
    'temperature': 0.0,
    'hubbard_stratonovich': 'dumb_continuous',
    'importance_sampling': True,
}
# Set up the calculation state, i.e., model + method + common options
state = afqmcpy.state.State(model, qmc_options)
# Print out calculation information for posterity
state.write_json()
# Run QMC calculation printing to stdout
psi0 = [afqmcpy.walker.Walker(1, state.system, state.psi_trial) for w in range(0, state.nwalkers)]
psi = afqmcpy.qmc.do_qmc(state, psi0, interactive=True)
# for w in psi:
    # print (w.ot)
# print ("# End Time: %s"%time.asctime())
# state.nsteps = 1
# afqmcpy.qmc.do_qmc(state, psi, interactive=True)
print ("# End Time: %s"%time.asctime())
# Compare to S. Zhang et.al (PRB 55, 7464 (1997))'s value of -6.6632 +/- 0.056
# and exact value of -6.672
print ("# End Time: %s"%time.asctime())
