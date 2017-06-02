import numpy as np
import scipy.linalg
import copy
import afqmcpy.walker as walker
import afqmcpy.estimators as estimators
import afqmcpy.pop_control as pop_control
import afqmcpy.propagation

def do_qmc(state, psi, comm, interactive=False):
    '''
'''

    est = []
    (E_T, pe, ke) = estimators.local_energy(state.system, psi[0].G)
    # initialise back propagated wavefunctions
    psit = copy.deepcopy(psi)
    # psibp only stores the auxiliary fields in the interval of tbp.
    psi_bp = copy.deepcopy(psi)
    estimates = estimators.Estimators()
    estimates.print_header(state.root)
    for w in psi:
        estimates.update(w, state)
    estimates.print_step(state, comm, 0)

    for step in range(1, state.nsteps):
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
        if step%state.nback_prop == 0:
            afqmcpy.propagation.back_propagate(state, psi, psit, psi_bp, estimates)
        if step%state.nmeasure == 0:
            E_T = (estimates.energy_denom/estimates.denom).real
            estimates.print_step(state, comm, step)
        if step < state.nequilibrate:
            state.mean_local_energy = E_T

    return (state, psi)
