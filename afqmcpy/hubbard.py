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
        for j in range(i+1, nbasis):
            xy1 = decode_basis(nx, ny, i)
            xy2 = decode_basis(nx, ny, j)
            dij = abs(xy1-xy2)
            if sum(dij) == 1:
                T[i, j] = -t
            # Take care of periodic boundary conditions
            if ((dij==[nx-1,0]).all() or (dij==[0,ny-1]).all()):
                T[i, j] += -t

    return T + T.T

def decode_basis(nx, ny, i):
    return np.array([i%nx, i/nx])
