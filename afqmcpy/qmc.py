import numpy as np
import scipy.linalg
import time
import afqmcpy.walker as walker
import afqmcpy.estimators as estimators
import afqmcpy.pop_control as pop_control

def do_qmc(state, interactive=False):
    '''
'''

    est = []
    psi = [walker.Walker(1, state.system, state.psi_trial) for w in range(0, state.nwalkers)]
    E_T = estimators.local_energy(state.system, psi[0].G)[0]
    estimators.header()
    elocal = 0
    total_weight = 0
    init_time = time.time()
    for step in range(0, state.nsteps):
        for w in psi:
            state.propagators.propagate_walker(w, state)
            w.weight = w.weight * np.exp(state.dt*(E_T-state.cfac))
            elocal += w.weight * estimators.local_energy(state.system, w.G)[0]
            total_weight += w.weight
            if step%state.nmeasure == 0:
                w.reortho()
        if step%state.nmeasure == 0:
            end_time = time.time()
            if interactive:
                est.append(elocal/(state.nmeasure*total_weight))
            else:
                print (("%9d %10.8e %10.8e %10.8e %.3f")%(step, total_weight/state.nmeasure,
                        elocal/state.nmeasure,np.exp(state.dt*(E_T-state.cfac)),
                        end_time-init_time))
            pop_control.comb(psi, state.nwalkers)
            E_T = elocal / total_weight
            init_time = end_time
            elocal = 0.0
            total_weight = 0.0

    if interactive:
        return est

