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
        self.step = 0
        self.init_time = time.time()

    def print_header(self):
        '''Print out header for estimators'''
        headers = ['iteration', 'Weight', 'E_num', 'E_denom', 'E', 'time']
        print (' '.join('{:>17}'.format(h) for h in headers))


    def print_step(self, state, comm):
        local_estimates = numpy.array([self.step, self.total_weight.real, self.energy_denom.real,
                                       self.denom.real,
                                       (self.energy_denom/self.denom).real,
                                       time.time()-self.init_time])
        global_estimates = numpy.zeros(len(local_estimates))
        comm.Reduce(local_estimates, global_estimates, op=MPI.SUM)
        # print (local_estimates)
        # print (global_estimates)
        if state.root:
            print (' '.join('{: .10e}'.format(v) for v in global_estimates))
        self.__init__()

    def update(self, w, state, step):
        self.step = step
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

def gab(a, b):
    inv_o = scipy.linalg.inv((a.conj().T).dot(b))
    gab = a.dot(inv_o.dot(b.conj().T)).T
    return gab
