"""Routines and classes for estimation of observables."""
import numpy
import time
from enum import Enum
from mpi4py import MPI
import scipy.linalg
import afqmcpy.utils


class Estimators():
    """Container for qmc estimates of observables.

    Attributes
    ----------
    energy_num : float
        Numerator of local energy estimator for the whole collection of walkers.
    total_weight : float
        Total weight of all the walkers in the simulation
    denom : float
        Denominator for energy estimates, usually the same as total_weight (see
        Estimators.update)
    init_time : float
        CPU time zero for estimating time taken to complete one step of the
        algorithm (not currently a per core quantity).
    """

    def __init__(self, state):
        self.header = ['iteration', 'Weight', 'E_num', 'E_denom', 'E', 'time']
        if state.back_propagation:
            if state.root:
                self.funit = open('back_propagated_estimates_%s.out'%state.uuid[:8], 'a')
                state.write_json(print_function=self.funit.write, eol='\n', verbose=False)
            self.back_propagated_header = ['iteration', 'E', 'T', 'V']
            # don't communicate the estimators header
            self.nestimators = len(self.header+self.back_propagated_header) - 2
            self.names = EstimatorEnum(self.nestimators)
        else:
            self.nestimators = len(self.header)
            self.names = EstimatorEnum(self.nestimators+3)
        self.estimates = numpy.zeros(self.nestimators)
        self.zero()


    def zero(self):
        self.estimates[:] = 0
        self.estimates[self.names.time] = time.time()

    def print_header(self, root, header, print_function=print, eol=''):
        '''Print out header for estimators'''
        if root:
            print_function(afqmcpy.utils.format_fixed_width_strings(header)+eol)

    def print_step(self, state, comm, step):
        """Print QMC estimates

        Note that the back-propagated estimates correspond to step-dt_bp.

        """
        es = self.estimates
        ns = self.names
        es[ns.eproj] = (state.nmeasure*es[ns.enumer]/(state.nprocs*es[ns.edenom])).real
        es[ns.weight:ns.enumer] = es[ns.weight:ns.enumer].real
        es[ns.time] = (time.time()-es[ns.time])/state.nprocs
        global_estimates = numpy.zeros(len(self.estimates))
        comm.Reduce(es, global_estimates, op=MPI.SUM)
        global_estimates[:ns.time] = global_estimates[:ns.time] / state.nmeasure
        if state.root:
            print(afqmcpy.utils.format_fixed_width_floats([step]+
                                                          list(global_estimates[:ns.evar])))
            if state.back_propagation:
                ff = afqmcpy.utils.format_fixed_width_floats([step]+
                                                             list(global_estimates[ns.evar:]))
                self.funit.write(ff+'\n')
        self.zero()

    def update(self, w, state):
        """Update estimates for walker w.

        Parameters
        ----------
        w : :class:`afqmcpy.walker.Walkder`
            current walker
        state : :class:`afqmcpy.state.State`
            system parameters as well as current 'state' of the simulation.
        """
        if state.importance_sampling:
            # When using importance sampling we only need to know the current
            # walkers weight as well as the local energy, the walker's overlap
            # with the trial wavefunction is not needed.
            if state.cplx:
                self.estimates[self.names.enumer] += w.weight * w.E_L.real
            else:
                self.estimates[self.names.enumer] += w.weight * local_energy(state.system, w.G)[0]
            self.estimates[self.names.weight] += w.weight
            self.estimates[self.names.edenom] += w.weight
        else:
            self.estimates[self.names.enumer] += w.weight * local_energy(state.system, w.G)[0] * w.ot
            self.estimates[self.names.weight] += w.weight
            self.estimates[self.names.edenom] += w.weight * w.ot

    def update_back_propagated_observables(self, system, psi, psit, psib):
        """"Update estimates using back propagated wavefunctions.

        Parameters
        ----------
        state : :class:`afqmcpy.state.State`
            state object
        psi : list of :class:`afqmcpy.walker.Walker` objects
            current distribution of walkers, i.e., at the current iteration in the
            simulation corresponding to :math:`\tau'=\tau+\tau_{bp}`.
        psit : list of :class:`afqmcpy.walker.Walker` objects
            previous distribution of walkers, i.e., at the current iteration in the
            simulation corresponding to :math:`\tau`.
        psib : list of :class:`afqmcpy.walker.Walker` objects
            backpropagated walkers at time :math:`\tau_{bp}`.
        """

        self.estimates[self.names.evar:] = back_propagated_energy(system, psi, psit, psib)


class EstimatorEnum:
    """Enum structure for help with indexing estimators array.

    python's support for enums doesn't help as it indexes from 1.
    """

    def __init__(self, nestimators):
        (self.weight, self.enumer, self.edenom, self.eproj,
         self.time, self.evar, self.kin, self.pot) = range(nestimators)


def local_energy(system, G):
    '''Calculate local energy of walker for the Hubbard model.

Parameters
----------
system : :class:`Hubbard`
    System information for the Hubbard model.
G : :class:`numpy.ndarray`
    Greens function for given walker phi, i.e.,
    :math:`G=\langle \phi_T| c_i^{\dagger}c_j | \phi\rangle`.

Returns
-------
E_L(phi) : float
    Local energy of given walker phi.
'''

    ke = numpy.sum(system.T * (G[0] + G[1]))
    pe = sum(system.U*G[0][i][i]*G[1][i][i] for i in range(0, system.nbasis))

    return (ke + pe, ke, pe)


def back_propagated_energy(system, psi, psit, psib):
    """Calculate back-propagated "local" energy for given walker/determinant.

    Parameters
    ----------
    psi : list of :class:`afqmcpy.walker.Walker` objects
        current distribution of walkers, i.e., at the current iteration in the
        simulation corresponding to :math:`\tau'=\tau+\tau_{bp}`.
    psit : list of :class:`afqmcpy.walker.Walker` objects
        previous distribution of walkers, i.e., at the current iteration in the
        simulation corresponding to :math:`\tau`.
    psib : list of :class:`afqmcpy.walker.Walker` objects
        backpropagated walkers at time :math:`\tau_{bp}`.
    """
    denominator = sum(w.weight for w in psi)
    estimates = numpy.zeros(3)
    GTB = [0, 0]
    for (w, wt, wb) in zip(psi, psit, psib):
        GTB[0] = gab(wb.phi[0], wt.phi[0])
        GTB[1] = gab(wb.phi[1], wt.phi[1])
        estimates = estimates + w.weight*numpy.array(list(local_energy(system, GTB)))
        # print (w.weight, wt.weight, wb.weight, local_energy(system, GTB))
    return estimates / denominator


def gab(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,
    .. math::
        \langle phi_A|c_i^{\dagger}c_j|phi_B\rangle = [B(A^{*T}B)^{-1}A^{*T}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    inv_O = scipy.linalg.inv((A.conj().T).dot(B))
    GAB = B.dot(inv_O.dot(A.conj().T)).T
    return GAB
