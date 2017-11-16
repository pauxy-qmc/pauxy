import numpy
from math import exp
import scipy.linalg
import copy
import random
import afqmcpy.walker as walker
import afqmcpy.estimators
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
    if state.estimators.back_propagation:
        # Easier to just keep a histroy of all walkers for population control
        # purposes if a bit memory inefficient.
        # TODO: just store historic fields rather than all the walkers.
        state.estimators.psi_hist[:,0] = copy.deepcopy(psi)
    else:
        state.estimators.psi_hist = None

    (E_T, ke, pe) = psi[0].local_energy(state.system)
    state.qmc.mean_local_energy = E_T.real
    # Calculate estimates for initial distribution of walkers.
    for w in psi:
        state.estimators.update(w, state)
    # Print out zeroth step for convenience.
    state.estimators.print_step(state, comm, 0, print_bp=False, print_itcf=False)

    for step in range(1, state.qmc.nsteps+1):
        for w in psi:
            # Want to possibly allow for walkers with negative / complex weights
            # when not using a constraint. I'm not so sure about the criteria
            # for complex weighted walkers.
            if abs(w.weight) > 1e-8:
                state.propagators.propagate_walker(w, state)
            # Constant factors
            w.weight = w.weight * exp(state.qmc.dt*E_T.real)
            # Add current (propagated) walkers contribution to estimates.
            state.estimators.update(w, state)
            if step%state.qmc.nstblz == 0:
                detR = w.reortho(state.system.nup)
                if not state.qmc.importance_sampling:
                    w.weight = detR * w.weight
        bp_step = (step-1)%state.estimators.nprop_tot
        if state.estimators.back_propagation:
            state.estimators.psi_hist[:,bp_step+1] = copy.deepcopy(psi)
            if step%state.estimators.back_prop.nmax == 0:
                # start and end points for selecting field configurations.
                s = bp_step - state.estimators.back_prop.nmax + 1
                e = bp_step + 1
                # the first entry in psi_hist (with index 0) contains the
                # wavefunction at the step where we start to accumulate the
                # auxiliary field path.  Since the propagated wavefunction,
                # i.e., the (n+1) st wfn, contains the fields which propagate
                # Psi_n to Psi_{n+1} we want to select the next entry in the
                # array, i.e., s+1. Slicing excludes the endpoint which we need
                # so also add one to e.
                psi_left = state.propagators.back_propagate(state.system,
                        state.estimators.psi_hist[:,s+1:e+1], state.trial)
                state.estimators.back_prop.update(state.system,
                                              state.estimators.psi_hist[:,e],
                                              state.estimators.psi_hist[:,s],
                                              psi_left)
                if not state.estimators.calc_itcf:
                    # New nth right-hand wfn for next estimate of ITCF.
                    state.estimators.psi_hist[:,0] = copy.deepcopy(psi)
        if state.estimators.calc_itcf and step%state.estimators.nprop_tot == 0:
            if state.estimators.itcf.stable:
                state.estimators.itcf.calculate_spgf(state,
                                                     state.estimators.psi_hist,
                                                     psi_left)
            else:
                state.estimators.itcf.calculate_spgf_unstable(state,
                                                              state.estimators.psi_hist,
                                                              psi_left)
            # New nth right-hand wfn for next estimate of ITCF.
            state.estimators.psi_hist[:,0] = copy.deepcopy(psi)
        if step%state.qmc.nmeasure == 0:
            # Todo: proj energy function
            E_T = afqmcpy.estimators.eproj(state.estimators.estimates,
                                           state.estimators.names)
            state.estimators.print_step(state, comm, step)
        if step < state.qmc.nequilibrate:
            # Update local energy bound.
            state.mean_local_energy = E_T
        if step%state.qmc.npop_control == 0:
            state.estimators.psi_hist = pop_control.comb(psi, state.qmc.nwalkers,
                                                         state.estimators.psi_hist)

    return (state, psi)
