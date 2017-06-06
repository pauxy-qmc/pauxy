import numpy
import time
from enum import Enum
from mpi4py import MPI
import scipy.linalg
import afqmcpy.utils


class Estimators():

    def __init__(self, state):
        self.header = ['iteration', 'Weight', 'E_num', 'E_denom', 'E', 'time']
        if state.back_propagation:
            self.funit = open('back_propagated_estimates_%s.out'%state.uuid[:8], 'a')
            self.back_propagated_header = ['tau_bp', 'E_var', 'T', 'V']
            state.write_json(print_function=self.funit.write, eol='\n')
            self.nestimators = len(self.header+self.back_propagated_header)
        else:
            self.nestimators = len(self.header)
        self.names = EstimatorEnum()
        self.estimates = numpy.zeros(self.nestimators)


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
        es[ns.time] = time.time()-es[ns.time]
        global_estimates = numpy.zeros(len(self.estimates))
        comm.Reduce(es, global_estimates, op=MPI.SUM)
        global_estimates = global_estimates / state.nmeasure
        if state.root:
            global_estimates[ns.iteration] = step
            print(afqmcpy.utils.format_fixed_width_floats(global_estimates[:ns.evar-1]))
        if state.back_propagation:
            global_estimates[ns.tau_bp] = state.dt*(step-state.nback_prop)
            self.funit.write(afqmcpy.utils.format_fixed_width_floats(global_estimates[ns.tau_bp:])+'\n')
        self.zero()

    def update(self, w, state):
        if state.importance_sampling:
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
    iteration = 0
    weight = 1
    enumer = 2
    edenom = 3
    eproj = 4
    time = 5
    tau_bp = 6
    evar = 7
    kin = 8
    pot = 9


def local_energy(system, G):
    '''Calculate local energy of walker for the Hubbard model.

Parameters
----------
system : :class:`Hubbard`
    System information for the Hubbard model.
G : :class:`numpy.ndarray`
    Greens function for given walker phi, i.e.,
    :math:`G=\langle \phi_T| c_j^{\dagger}c_i | \phi\rangle`.

Returns
-------
E_L(phi) : float
    Local energy of given walker phi.
'''

    ke = numpy.sum(system.T * (G[0] + G[1]))
    pe = sum(system.U*G[0][i][i]*G[1][i][i] for i in range(0, system.nbasis))

    return (ke + pe, pe, ke)


def back_propagated_energy(system, psi, psit, psib):
    """
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
        GTB[0] = gab(wt.phi[0], wb.phi[0])
        GTB[1] = gab(wt.phi[1], wb.phi[1])
        estimates = estimates + w.weight*numpy.array(list(local_energy(system, GTB)))
        # print (w.weight, local_energy(system, GTB)[0])
    return estimates / denominator


def gab(a, b):
    inv_o = scipy.linalg.inv((a.conj().T).dot(b))
    gab = a.dot(inv_o.dot(b.conj().T)).T
    return gab
