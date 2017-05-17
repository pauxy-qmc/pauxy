import numpy as np
import scipy.linalg
import afqmcpy.walker as walker
import afqmcpy.estimators as estimators
import afqmcpy.pop_control as pop_control

def do_qmc(state, psi, interactive=False):
    '''
'''

    est = []
    E_T = estimators.local_energy(state.system, psi[0].G)[0]
    estimates = estimators.Estimators()
    estimates.print_header()
    for step in range(0, state.nsteps):
        for w in psi:
            # Hack
            if w.weight > 1e-8:
                state.propagators.propagate_walker(w, state)
            # Constant factors
            w.weight = w.weight * np.exp(state.dt*E_T)
            estimates.update(w, state, step)
            if step%state.nmeasure == 0:
                if state.importance_sampling:
                    w.reortho()
                else:
                    w.reortho_free()
        if step%state.npop_control == 0:
            pop_control.comb(psi, state.nwalkers)
        if step%state.nmeasure == 0:
            # if interactive:
                # # est.append(elocal/(state.nmeasure*total_weight))
            # else:
            E_T = estimates.energy_denom / estimates.denom
            estimates.print_step(state)

    if interactive:
        return psi
