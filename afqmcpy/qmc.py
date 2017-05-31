import numpy as np
import scipy.linalg
import afqmcpy.walker as walker
import afqmcpy.estimators as estimators
import afqmcpy.pop_control as pop_control

def do_qmc(state, psi, comm, interactive=False):
    '''
'''

    est = []
    (E_T, pe, ke) = estimators.local_energy(state.system, psi[0].G)
    estimates = estimators.Estimators()
    estimates.print_header(state.root)
    for w in psi:
        estimates.update(w, state)
    estimates.print_step(state, comm, 0)

    for step in range(0, state.nsteps):
        for w in psi:
            # Hack
            if abs(w.weight) > 1e-8:
                state.propagators.propagate_walker(w, state)
            # Constant factors
            w.weight = w.weight * np.exp(state.dt*E_T)
            estimates.update(w, state)
            if step%state.nmeasure == 0 and step != 0:
                if state.importance_sampling:
                    w.reortho()
                else:
                    w.reortho_free()
        if step%state.npop_control == 0:
            pop_control.comb(psi, state.nwalkers)
        if step%state.nmeasure == 0:
            E_T = (estimates.energy_denom/estimates.denom).real
            estimates.print_step(state, comm, step)
        if step < state.nequilibrate:
            state.mean_local_energy = E_T

    return (state, psi)
