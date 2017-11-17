'''Hubbard model specific classes and methods'''

import numpy
import cmath
import math
import scipy.linalg
import afqmcpy.kpoints

class Hubbard:
    """Hubbard model system class.

    Only consider 1 and 2 case with nearest neighbour hopping.

    Parameters
    ----------
    inputs : dict
        dictionary of system input options.

    Attributes
    ----------
    nup : int
        Number of up electrons.
    ndown : int
        Number of down electrons.
    ne : int
        Number of electrons.
    t : float
        Hopping parameter.
    U : float
        Hubbard U interaction strength.
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.
    nbasis : int
        Number of single-particle basis functions.
    T : numpy.array
        Hopping matrix
    gamma : numpy.array
        Super matrix (not currently implemented).
    """

    def __init__(self, inputs, dt):
        self.nup = inputs['nup']
        self.ndown = inputs['ndown']
        self.ne = self.nup + self.ndown
        self.t = inputs['t']
        self.U = inputs['U']
        self.nx = inputs['nx']
        self.ny = inputs['ny']
        self.ktwist = numpy.array(inputs.get('ktwist'))
        self.nbasis = self.nx * self.ny
        (self.kpoints, self.kc, self.eks) = afqmcpy.kpoints.kpoints(self.t,
                                                                    self.nx,
                                                                    self.ny)
        self.T = kinetic(self.t, self.nbasis, self.nx,
                         self.ny, self.ktwist)
        self.Text = scipy.linalg.block_diag(self.T, self.T)
        self.super = _super_matrix(self.U, self.nbasis)
        self.P = transform_matrix(self.nbasis, self.kpoints,
                                  self.kc, self.nx, self.ny)
        self.gamma = numpy.arccosh(numpy.exp(0.5*dt*self.U))
        self.auxf = numpy.array([[numpy.exp(self.gamma), numpy.exp(-self.gamma)],
                                [numpy.exp(-self.gamma), numpy.exp(self.gamma)]])
        self.auxf = self.auxf * numpy.exp(-0.5*dt*self.U)

def transform_matrix(nbasis, kpoints, kc, nx, ny):
    U = numpy.zeros(shape=(nbasis, nbasis), dtype=complex)
    for (i, k_i) in enumerate(kpoints):
        for j in range(0, nbasis):
            r_j = decode_basis(nx, ny, j)
            U[i,j] = numpy.exp(1j*numpy.dot(kc*k_i,r_j))

    return U


def kinetic(t, nbasis, nx, ny, ks):
    """Kinetic part of the Hamiltonian in our one-electron basis.

    Parameters
    ----------
    t : float
        Hopping parameter
    nbasis : int
        Number of one-electron basis functions.
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.

    Returns
    -------
    T : numpy.array
        Hopping Hamiltonian matrix.
    """

    if ks.all() is None:
        T = numpy.zeros((nbasis, nbasis), dtype=float)
    else:
        T = numpy.zeros((nbasis, nbasis), dtype=complex)

    for i in range(0, nbasis):
        for j in range(i+1, nbasis):
            xy1 = decode_basis(nx, ny, i)
            xy2 = decode_basis(nx, ny, j)
            dij = abs(xy1-xy2)
            if sum(dij) == 1:
                T[i, j] = -t
            # Take care of periodic boundary conditions
            # there should be a less stupid way of doing this.
            if ny == 1 and dij == [nx-1]:
                if ks.all() is not None:
                    phase = cmath.exp(1j*numpy.dot(cmath.pi*ks,[1]))
                else:
                    phase = 1.0
                T[i,j] += -t * phase
            elif (dij==[nx-1, 0]).all():
                if ks.all() is not None:
                    phase = cmath.exp(1j*numpy.dot(cmath.pi*ks,[1,0]))
                else:
                    phase = 1.0
                T[i, j] += -t * phase 
            elif (dij==[0, ny-1]).all():
                if ks.all() is not None:
                    phase = cmath.exp(1j*numpy.dot(cmath.pi*ks,[0,1]))
                else:
                    phase = 1.0
                T[i, j] += -t * phase 

    # This only works because the diagonal of T is zero.
    return T + T.conj().T

def decode_basis(nx, ny, i):
    """Return cartesian lattice coordinates from basis index.

    Parameters
    ----------
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.
    i : int
        Basis index (same for up and down spins).
    """
    if ny == 1:
        return numpy.array([i%nx])
    else:
        return numpy.array([i//nx, i%nx])

def _super_matrix(U, nbasis):
    '''Construct super-matrix from v_{ijkl}'''

