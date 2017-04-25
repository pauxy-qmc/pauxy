'''Hubbard model specific classes and methods'''

import numpy as np
import scipy.linalg

class Hubbard:

    def __init__(self, inputs):
        self.nup = inputs['nup']
        self.ndown = inputs['ndown']
        self.ne = self.nup + self.ndown
        self.t = inputs['t']
        self.U = inputs['U']
        self.nx = inputs['nx']
        self.ny = inputs['ny']
        if self.ny > 1:
            self.nbasis = self.nx*self.ny
        else:
            self.nbasis = self.nx
        self.T = kinetic(self.t, self.nbasis, self.nx, self.ny)

class Projectors:

    def __init__(self, hubb, dt):
        self.bt2 = scipy.linalg.expm(-0.5*dt*hubb.T)


def kinetic(t, nbasis, nx, ny):
    '''Kinetic part of the Hamiltonian

'''

    T = np.zeros((nbasis, nbasis))

    for i in range(0, nbasis):
        for j in range(0, nbasis):
            xy1 = decode_basis(nx, ny, i)
            xy2 = decode_basis(nx, ny, j)
            # Only consider square/cubic grids for simplicity.
            dij = sum(abs(xy1-xy2))
            if (dij == 1 or dij == nx-1):
                T[i, j] = -t
    return T

def decode_basis(nx, ny, i):
    return np.array([i%nx, i/nx])
