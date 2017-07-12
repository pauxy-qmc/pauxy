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
    if state.back_propagation:
        # Easier to just keep a histroy of all walkers for population control
        # purposes if a bit memory inefficient.
        psi_hist = numpy.empty(shape=(state.nwalkers, state.nprop_tot+1),
                               dtype=object)
        psi_hist[:,0] = copy.deepcopy(psi)
    else:
        psi_hist = None

    (E_T, ke, pe) = estimators.local_energy(state.system, psi[0].G)
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
            if step%state.nmeasure == 0:
                if state.importance_sampling:
                    w.reortho()
                else:
                    w.reortho_free()
        bp_step = (step-1)%state.nprop_tot
        if state.back_propagation:
            psi_hist[:,bp_step+1] = copy.deepcopy(psi)
            if step%state.nback_prop == 0:
                # start and end points for selecting field configurations.
                s = bp_step - state.nback_prop + 1
                e = bp_step + 1
                # the first entry in psi_hist (with index 0) contains the
                # wavefunction at the step where we start to accumulate the
                # auxiliary field path.  Since the propagated wavefunction,
                # i.e., the (n+1) st wfn, contains the fields which propagate
                # Psi_n to Psi_{n+1} we want to select the next entry in the
                # array, i.e., s+1. Slicing excludes the endpoint which we need
                # so also add one to e.
                psi_left = afqmcpy.propagation.back_propagate(state,
                                                              psi_hist[:,s+1:e+1])
                estimates.update_back_propagated_observables(state.system,
                                                             psi_hist[:,e],
                                                             psi_hist[:,s],
                                                             psi_left)
                if not state.itcf:
                    # New nth right-hand wfn for next estimate of ITCF.
                    psi_hist[:,0] = copy.deepcopy(psi)
        if state.itcf and step%state.nprop_tot == 0:
            if state.itcf_stable:
                estimates.calculate_itcf(state, psi_hist, psi_left)
            else:
                estimates.calculate_itcf_unstable(state, psi_hist, psi_left)
            # New nth right-hand wfn for next estimate of ITCF.
            psi_hist[:,0] = copy.deepcopy(psi)
        if step%state.nmeasure == 0:
            # Todo: proj energy function
            E_T = (estimates.estimates[estimates.names.enumer]/estimates.estimates[estimates.names.edenom]).real
            estimates.print_step(state, comm, step)
        if step < state.nequilibrate:
            # Update local energy bound.
            state.mean_local_energy = E_T
        if step%state.npop_control == 0:
            psi_hist = pop_control.comb(psi, state.nwalkers, psi_hist)

    return (state, psi)
