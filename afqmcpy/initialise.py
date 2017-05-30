import sys
import json
import afqmcpy.state
import afqmcpy.qmc
import afqmcpy.walker
import time

def initialise(input_file):

    with open(input_file) as inp:
        options = json.load(inp)

    print('# Initialising AFQMCPY simulation from %s'%input_file)
    state = afqmcpy.state.State(options['model'], options['qmc_options'])
    state.write_json()
    psi0 = [afqmcpy.walker.Walker(1, state.system, state.trial.psi, w) for w in
            range(state.nwalkers)]
    afqmcpy.qmc.do_qmc(state, psi0)

def finalise():

    print ("# End Time: %s"%time.asctime())
