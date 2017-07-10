import numpy
from math import exp
import scipy.linalg
import copy
import random
import afqmcpy.walker as walker
import afqmcpy.estimators as estimators
import afqmcpy.pop_control as pop_control
import afqmcpy.propagation

def do_qmc(state, psi, comm):
    """Perform CPMC simulation on state object.

    Parameters
    ----------
    state : :class:`afqmcpy.state.State` object
        Model and qmc parameters.
    psi : list of :class:`afqmcpy.walker.Walker` objects
        Initial wavefunction / distribution of walkers.
    comm : MPI communicator
    """

    # Energy of the initial distribution.
    (E_T, ke, pe) = estimators.local_energy(state.system, psi[0].G)
    # initialise back propagated wavefunctions
    psi_n = copy.deepcopy(psi)
    # initialise wavefunction for ITCF
    psi_right = copy.deepcopy(psi)
    # psibp only stores the auxiliary fields in the interval of tbp.
    psi_bp = copy.deepcopy(psi)
    # Set up estimators.
    estimates = estimators.Estimators(state)
    estimates.print_header(state.root, estimates.header)
    if state.back_propagation and state.root:
        estimates.print_header(state.root, estimates.back_propagated_header,
                               print_function=estimates.funit.write, eol='\n')
    # Calculate estimates for initial distribution of walkers.
    for w in psi:
        estimates.update(w, state)
    # We can't have possibly performed back propagation yet so don't print out
    # zero which would mess up the averages.
    estimates.print_step(state, comm, 0, print_bp=False, print_itcf=False)

    for step in range(1, state.nsteps):
        for w in psi:
            # Want to possibly allow for walkers with negative / complex weights
            # when not using a constraint. I'm not so sure about the criteria
            # for complex weighted walkers.
            if abs(w.weight) > 1e-8:
                state.propagators.propagate_walker(w, state)
            # Constant factors
            w.weight = w.weight * exp(state.dt*E_T)
            # Add current (propagated) walkers contribution to estimates.
            estimates.update(w, state)
            if step%state.nmeasure == 0 and step != 0:
                if state.importance_sampling:
                    w.reortho()
                else:
                    w.reortho_free()
        if step%state.npop_control == 0:
            pop_control.comb(psi, state.nwalkers)
        if state.back_propagation and step%state.nback_prop == 0:
            # Headache re one-indexing the steps and using modular arithmetic for
            # indexing the zero-indexed auxiliary field arrays.
            bp_step = (step-1)%state.nprop_tot
            psi_left = afqmcpy.propagation.back_propagate(state, psi, bp_step)
            estimates.update_back_propagated_observables(state.system, psi,
                                                         psi_n, psi_left)
            # set (n+m)th (i.e. the current step's) wfn to be nth wfn for
            # next back propagation step.
            psi_n = copy.deepcopy(psi)
        if state.itcf and step%state.nprop_tot == 0:
            if state.itcf_stable:
                estimates.calculate_itcf(state, psi, psi_right, psi_left)
            else:
                estimates.calculate_itcf_unstable(state, psi, psi_right, psi_left)
            # New nth right-hand wfn for next estimate of ITCF.
            psi_right = copy.deepcopy(psi)
        if step%state.nmeasure == 0:
            # Todo: proj energy function
            E_T = (estimates.estimates[estimates.names.enumer]/estimates.estimates[estimates.names.edenom]).real
            estimates.print_step(state, comm, step)
        if step < state.nequilibrate:
            # Update local energy bound.
            state.mean_local_energy = E_T

    return (state, psi)
