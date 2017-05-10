import numpy as np


def header():
    '''Print out header for estimators'''

    print ("%9s %14s %15s %14s %5s"%('iteration', 'Weight', 'E_num',
           'exp(delta)', 'time'))


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

    return ke + pe
