import numpy as np
import time


class Estimators():

    def __init__(self):
        self.energy_denom = 0.0
        self.total_weight = 0.0
        self.denom = 0.0
        self.step = 0
        self.init_time = time.time()

    def print_header(self):
        '''Print out header for estimators'''
        print ("%9s %14s %15s %14s %5s"%('iteration', 'Weight', 'E_num',
               'E_denom', 'time'))


    def print_step(self, state):
        print (("%9d %10.8e %10.8e %10.8e %.3f")%(self.step, self.total_weight/state.nmeasure,
                self.energy_denom/state.nmeasure, self.denom/state.nmeasure,
                time.time()-self.init_time))
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

    ke = np.sum(system.T * (G[0] + G[1]))
    pe = sum(system.U*G[0][i][i]*G[1][i][i] for i in range(0, system.nbasis))

    return (ke + pe, pe, ke)
