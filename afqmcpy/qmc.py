import numpy as np
import scipy.linalg
import copy
import random
import afqmcpy.walker as walker
import afqmcpy.estimators as estimators
import afqmcpy.pop_control as pop_control
import afqmcpy.propagation

def do_qmc(state, psi, comm, interactive=False):
    '''
'''

    est = []
    (E_T, ke, pe) = estimators.local_energy(state.system, psi[0].G)
    # initialise back propagated wavefunctions
    psit = copy.deepcopy(psi)
    # psibp only stores the auxiliary fields in the interval of tbp.
    psi_bp = copy.deepcopy(psi)
    estimates = estimators.Estimators(state)
    estimates.print_header(state.root, estimates.header)
    if state.back_propagation and state.root:
        estimates.print_header(state.root, estimates.back_propagated_header,
                               print_function=estimates.funit.write, eol='\n')
    for w in psi:
        estimates.update(w, state)
    # We can't have possibly performed back propagation yet so don't print out
    # zero which would mess up the averages.
    estimates.print_step(state, comm, 0, print_bp=False)

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
        if state.back_propagation and step%state.nback_prop == 0:
            psi_left = afqmcpy.propagation.back_propagate(state, psi, psit)
            estimates.update_back_propagated_observables(state.system, psi, psit, psi_bp)
            if state.itcf and step%state.nprop_tot == state.nback_prop:
                # save these for calculating the itcf
                psi_right = copy.deepcopy(psit)
                psi_left = copy.deepcopy(psi_bp)
                psi_nm = copy.deepcopy(psi)
            psit = copy.deepcopy(psi)
        if state.itcf and step%state.nprop_tot == 0:
            estimates.calculate_itcf(state, psi, psi_nm, psi_left, psi_right)
            if state.root:
                estimates.print_itcf(state.dt, estimates.itcf_unit)
        if step%state.nmeasure == 0:
            E_T = (estimates.estimates[estimates.names.enumer]/estimates.estimates[estimates.names.edenom]).real
            estimates.print_step(state, comm, step)
        if step < state.nequilibrate:
            state.mean_local_energy = E_T

    return (state, psi)
