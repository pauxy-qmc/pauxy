import numpy
from mpi4py import MPI
import time
import scipy.linalg
import afqmcpy.utils


class Estimators():

    def __init__(self):
        self.energy_denom = 0.0
        self.total_weight = 0.0
        self.denom = 0.0
        self.init_time = time.time()

    def print_header(self, root):
        '''Print out header for estimators'''
        headers = ['iteration', 'Weight', 'E_num', 'E_denom', 'E', 'time']
        if root:
            print (' '.join('{:>17}'.format(h) for h in headers))


    def print_step(self, state, comm, step):
        local_estimates = numpy.array([step*state.nmeasure/state.nprocs,
                                       self.total_weight.real,
                                       self.energy_denom.real,
                                       self.denom.real,
                                       (state.nmeasure*self.energy_denom/(state.nprocs*self.denom)).real,
                                       time.time()-self.init_time])
        global_estimates = numpy.zeros(len(local_estimates))
        comm.Reduce(local_estimates, global_estimates, op=MPI.SUM)
        if state.root:
            print (' '.join('{: .10e}'.format(v/(state.nmeasure)) for v in global_estimates))
        self.__init__()

    def update(self, w, state):
        if state.importance_sampling:
            if state.cplx:
                self.energy_denom += w.weight * w.E_L.real
            else:
                self.energy_denom += w.weight * local_energy(state.system, w.G)[0]
            self.total_weight += w.weight
            self.denom += w.weight
        else:
            self.energy_denom += w.weight * local_energy(state.system, w.G)[0] * w.ot
            self.total_weight += w.weight
            self.denom += w.weight * w.ot

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

def update_back_propagated_observables(self, state, a, b):

    (self.evar, self.ke, self.pe) = sum(back_propagated_energy(w, a, b) for (w, a, b) in
                                        zip(psi.weights, psit.phi, psib.phi))

def gab(a, b):
    inv_o = scipy.linalg.inv((a.conj().T).dot(b))
    gab = a.dot(inv_o.dot(b.conj().T)).T
    return gab
