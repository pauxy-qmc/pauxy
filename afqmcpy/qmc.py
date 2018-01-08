import numpy
from math import exp
import scipy.linalg
import copy
import random
import afqmcpy.walker as walker
import afqmcpy.estimators
import afqmcpy.pop_control as pop_control
import afqmcpy.propagation

def do_qmc(state, psi, estimators, comm):
    """Perform CPMC simulation on state object.

    Parameters
    ----------
    state : :class:`afqmcpy.state.State` object
        Model and qmc parameters.
    psi : list of :class:`afqmcpy.walker.Walker` objects
        Initial wavefunction / distribution of walkers.
    comm : MPI communicator
    """
    if estimators.back_propagation:
        # Easier to just keep a histroy of all walkers for population control
        # purposes if a bit memory inefficient.
        # TODO: just store historic fields rather than all the walkers.
        estimators.psi_hist[:,0] = copy.deepcopy(psi)
    else:
        estimators.psi_hist = None

    (E_T, ke, pe) = psi.walkers[0].local_energy(state.system)
    state.qmc.mean_local_energy = E_T.real
    # Calculate estimates for initial distribution of walkers.
    estimators.estimators['mixed'].update(state.system, state.qmc,
                                          state.trial, psi, 0)
    # Print out zeroth step for convenience.
    estimators.estimators['mixed'].print_step(comm, state.nprocs, 0, 1)

    for step in range(1, state.qmc.nsteps+1):
        for w in psi.walkers:
            # Want to possibly allow for walkers with negative / complex weights
            # when not using a constraint. I'm not so sure about the criteria
            # for complex weighted walkers.
            if abs(w.weight) > 1e-8:
                state.propagators.propagate_walker(w, state)
            # Constant factors
            w.weight = w.weight * exp(state.qmc.dt*E_T.real)
            # Add current (propagated) walkers contribution to estimates.
        if step%state.qmc.nstblz == 0:
            psi.orthogonalise(state.qmc.importance_sampling)
        # calculate estimators
        estimators.update(state.system, state.qmc, state.trial, psi, step)
        if step%state.qmc.nmeasure == 0:
            # Todo: proj energy function
            E_T = afqmcpy.estimators.eproj(estimators.estimators['mixed'].estimates,
                                           estimators.estimators['mixed'].names)
            estimators.print_step(comm, state.nprocs, step, state.qmc.nmeasure)
        if step < state.qmc.nequilibrate:
            # Update local energy bound.
            state.mean_local_energy = E_T
        if step%state.qmc.npop_control == 0:
            estimators.psi_hist = pop_control.comb(psi, state.qmc.nwalkers,
                                                         estimators.psi_hist)

    return (state, psi)
