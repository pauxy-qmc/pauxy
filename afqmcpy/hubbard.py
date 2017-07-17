'''Hubbard model specific classes and methods'''

import numpy
import cmath
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
        (self.kpoints, self.kc, self.eks) = afqmcpy.kpoints.kpoints(self.t, self.nx, self.ny)
        print (self.kpoints)
        self.T = kinetic(self.t, self.nbasis, self.nx, self.ny)
        self.gamma = _super_matrix(self.U, self.nbasis)
        # Transformation matrix.
        self.P = transform_matrix(self.nbasis, self.kpoints,
                                                self.kc, self.nx, self.ny)

def transform_matrix(nbasis, kpoints, kc, nx, ny):
    U = numpy.zeros(shape=(nbasis, nbasis), dtype=complex)
    for (i, k_i) in enumerate(kpoints):
        for j in range(0, nbasis):
            r_j = decode_basis(nx, ny, j)
            U[i,j] = numpy.exp(1j*numpy.dot(kc*k_i,r_j))

    return U


def kinetic(t, nbasis, nx, ny):
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

    T = numpy.zeros((nbasis, nbasis))

    for i in range(0, nbasis):
        for j in range(i+1, nbasis):
            xy1 = decode_basis(nx, ny, i)
            xy2 = decode_basis(nx, ny, j)
            dij = abs(xy1-xy2)
            if sum(dij) == 1:
                T[i, j] = -t
            # Take care of periodic boundary conditions
            if ny == 1 and dij == [nx-1]:
                T[i,j] += -t
            elif ((dij==[nx-1, 0]).all() or (dij==[0,ny-1]).all()):
                T[i, j] += -t

    return T + T.T

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

