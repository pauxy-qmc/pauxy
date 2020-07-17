'''Hubbard model specific classes and methods'''

import cmath
import math
from math import cos, pi, sqrt
import numpy
import numpy
import scipy.linalg
from pauxy.utils.io import fcidump_header


class HubbardHolstein(object):
    """HubbardHolstein model system class.

    1 and 2 case with nearest neighbour hopping.

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
    w0 : float
        Frequency w0
    g : float
        electron-phonon coupling
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.
    nbasis : int
        Number of single-particle basis functions.
    T : numpy.array
        Hopping matrix
    """

    def __init__(self, inputs, verbose=False):
        if verbose:
            print("# Parsing input options.")
        self.nup = inputs.get('nup')
        self.ndown = inputs.get('ndown')
        self.ne = self.nup + self.ndown
        self.t = inputs.get('t', 1.0)
        self.lmbda = inputs.get('lambda', 1.0)
        self.w0 = inputs.get('w0', 1.0)
        self.m = inputs.get('m', 1.0/self.w0) # mass

        self.U = inputs['U']
        self.nx = inputs['nx']
        self.ny = inputs['ny']
        self.ktwist = numpy.array(inputs.get('ktwist'))
        self.symmetric = inputs.get('symmetric', False)

        self.ypbc = inputs.get('ypbc', True)
        self.xpbc = inputs.get('xpbc', True)

        if self.symmetric:
            # An unusual convention for the sign of the chemical potential is
            # used in Phys. Rev. B 99, 045108 (2018)
            # Symmetric uses the symmetric form of the hubbard model and will
            # also change the sign of the chemical potential in the density
            # matrix.
            self._alt_convention = True
        else:
            self._alt_convention = False
        self.nbasis = self.nx * self.ny

        # This depends on the dimension of the system hard-coded to do 1D
        d = 2
        if (self.nx == 1 or self.ny == 1): # 1d
            d = 1

        self.g = inputs.get('g', None)
        
        if (self.g == None):
            # This is assuming self.m = 1 / self.w0
            # to include mass see 10.1103/PhysRevLett.97.056402
            self.g = sqrt(float(d) * 2.0 * self.lmbda * self.t * self.w0)

        self.lang_firsov = inputs.get('lang_firsov', False)

        if verbose:
            print("# d = {}".format(d))
            print("# nx, ny = {},{}".format(self.nx, self.ny))
            print("# nbasis = {}".format(self.nbasis))
            print("# t, U = {}, {}".format(self.t, self.U))
            print("# m, w0, g, lambda = {}, {}, {}, {}".format(self.m, self.w0, self.g, self.lmbda))
            print("# lang_firsov = {}".format(self.lang_firsov))

        self.gamma = 0.0

        if (self.lang_firsov):
            self.gamma = self.g * numpy.sqrt(2.0 * self.m / self.w0)
            Ueff = self.U + self.gamma**2 * self.w0 - 2.0 * self.g * self.gamma * numpy.sqrt(2.0 * self.m * self.w0)
            if verbose:
                print("# Ueff = {}".format(Ueff))

        self.nactive = self.nbasis
        self.nfv = 0
        self.ncore = 0
        (self.kpoints, self.kc, self.eks) = kpoints(self.t, self.nx, self.ny)
        self.pinning = inputs.get('pinning_fields', False)
        self._opt = True
        if verbose:
            print("# Setting up one-body operator.")
        if self.pinning:
            if verbose:
                print("# Using pinning field.")
            self.T = kinetic_pinning_alt(self.t, self.nbasis, self.nx, self.ny)
        else:
            self.T = kinetic(self.t, self.nbasis, self.nx,
                             self.ny, self.ktwist, xpbc=self.xpbc, ypbc=self.ypbc)
        self.H1 = self.T
        self.Text = scipy.linalg.block_diag(self.T[0], self.T[1])
        self.P = transform_matrix(self.nbasis, self.kpoints,
                                  self.kc, self.nx, self.ny)
        self.mu = inputs.get('mu', None)
        # For interface consistency.
        self.ecore = 0.0
        # Number of field configurations per walker.
        self.nfields = self.nbasis
        self.name = "HubbardHolstein"
        if verbose:
            print("# Finished setting up Hubbard-Holstein system object.")
        # "Volume" to define density.
        self.vol = self.nx * self.ny
        self.construct_h1e_mod()

    def fcidump(self, to_string=False):
        """Dump 1- and 2-electron integrals to file.

        Parameters
        ----------
        to_string : bool
            Return fcidump as string. Default print to stdout.
        """
        header = fcidump_header(self.ne, self.nbasis, self.nup-self.ndown)
        for i in range(1, self.nbasis+1):
            if self.T.dtype == complex:
                fmt = "({: 10.8e}, {: 10.8e}) {:>3d} {:>3d} {:>3d} {:>3d}\n"
                line = fmt.format(self.U.real, self.U.imag, i, i, i, i)
            else:
                fmt = "{: 10.8e} {:>3d} {:>3d} {:>3d} {:>3d}\n"
                line = fmt.format(self.U, i, i, i, i)
            header += line
        for i in range(0, self.nbasis):
            for j in range(i+1, self.nbasis):
                integral = self.T[0][i,j]
                if (abs(integral) > 1e-8):
                    if self.T.dtype == complex:
                        fmt = (
                            "({: 10.8e}, {: 10.8e}) {:>3d} {:>3d} {:>3d} {:>3d}\n"
                        )
                        line = fmt.format(integral.real, integral.imag,
                                          i+1, j+1, 0, 0)
                    else:
                        fmt = "{: 10.8e} {:>3d} {:>3d} {:>3d} {:>3d}\n"
                        line = fmt.format(integral, i+1, j+1, 0, 0)
                    header += line
        if self.T.dtype == complex:
            fmt = "({: 10.8e}, {: 10.8e}) {:>3d} {:>3d} {:>3d} {:>3d}\n"
            header += fmt.format(0, 0, 0, 0, 0, 0)
        else:
            fmt = "{: 10.8e} {:>3d} {:>3d} {:>3d} {:>3d}\n"
            header += fmt.format(0, 0, 0, 0, 0)
        if to_string:
            print(header)
        else:
            return header
    
    def hijkl(self,i,j,k,l):
        if (i == k and j == l and i==j):
            return self.U
        else:
            return 0.0

    def construct_h1e_mod(self):
        # Subtract one-body bit following reordering of 2-body operators.
        # Eqn (17) of [Motta17]_
        if not self.symmetric:
            v0 = 0.5 * self.U * numpy.eye(self.nbasis)
            self.h1e_mod = numpy.array([self.H1[0]-v0, self.H1[1]-v0])
        else:
            self.h1e_mod = self.H1



def transform_matrix(nbasis, kpoints, kc, nx, ny):
    U = numpy.zeros(shape=(nbasis, nbasis), dtype=complex)
    for (i, k_i) in enumerate(kpoints):
        for j in range(0, nbasis):
            r_j = decode_basis(nx, ny, j)
            U[i,j] = numpy.exp(1j*numpy.dot(kc*k_i,r_j))

    return U


def kinetic(t, nbasis, nx, ny, ks, xpbc = True, ypbc = True):
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
        xy1 = decode_basis(nx, ny, i)
        for j in range(i+1, nbasis):
            xy2 = decode_basis(nx, ny, j)
            dij = abs(xy1-xy2)
            if sum(dij) == 1:
                T[i, j] = -t
            # Take care of periodic boundary conditions
            # there should be a less stupid way of doing this.
            if ny == 1 and dij == [nx-1] and xpbc:
                if ks.all() is not None:
                    phase = cmath.exp(1j*numpy.dot(cmath.pi*ks,[1]))
                else:
                    phase = 1.0
                T[i,j] += -t * phase
            elif (dij==[nx-1, 0]).all() and xpbc:
                if ks.all() is not None:
                    phase = cmath.exp(1j*numpy.dot(cmath.pi*ks,[1,0]))
                else:
                    phase = 1.0
                T[i, j] += -t * phase
            elif (dij==[0, ny-1]).all() and ypbc:
                if ks.all() is not None:
                    phase = cmath.exp(1j*numpy.dot(cmath.pi*ks,[0,1]))
                else:
                    phase = 1.0
                T[i, j] += -t * phase

    # This only works because the diagonal of T is zero.
    return numpy.array([T+T.conj().T, T+T.conj().T])

def kinetic_lang_firsov(t, gamma, P, nx, ny, ks):
    """Kinetic part of the Hamiltonian in our one-electron basis.

    Parameters
    ----------
    t : float
        Hopping parameter
    P : numpy.array
        momentum vector
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.
    ks : numpy.array
         k-twist
    
    Returns
    -------
    T : numpy.array
        Hopping Hamiltonian matrix.
    """

    nbasis = P.shape[0]

    T = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)

    for i in range(0, nbasis):
        xy1 = decode_basis(nx, ny, i)
        for j in range(i+1, nbasis):
            xy2 = decode_basis(nx, ny, j)
            dij = abs(xy1-xy2)

            exppij = numpy.exp(1j * gamma * (P[i]-P[j]))

            if sum(dij) == 1:
                T[i, j] = -t * exppij
            # Take care of periodic boundary conditions
            # there should be a less stupid way of doing this.
            if ny == 1 and dij == [nx-1]:
                if ks.all() is not None:
                    phase = cmath.exp(1j*numpy.dot(cmath.pi*ks,[1]))
                else:
                    phase = 1.0
                T[i,j] += -t * exppij * phase
            elif (dij==[nx-1, 0]).all():
                if ks.all() is not None:
                    phase = cmath.exp(1j*numpy.dot(cmath.pi*ks,[1,0]))
                else:
                    phase = 1.0
                T[i, j] += -t * exppij * phase
            elif (dij==[0, ny-1]).all():
                if ks.all() is not None:
                    phase = cmath.exp(1j*numpy.dot(cmath.pi*ks,[0,1]))
                else:
                    phase = 1.0
                T[i, j] += -t * exppij * phase

    # This only works because the diagonal of T is zero.
    return numpy.array([T+T.conj().T, T+T.conj().T])

def kinetic_pinning(t, nbasis, nx, ny):
    r"""Kinetic part of the Hamiltonian in our one-electron basis.

    Adds pinning fields as outlined in [Qin16]_. This forces periodic boundary
    conditions along x and open boundary conditions along y. Pinning fields are
    applied in the y direction as:

        .. math::
            \nu_{i\uparrow} = -\nu_{i\downarrow} = (-1)^{i_x+i_y}\nu_0,

    for :math:`i_y=1,L_y` and :math:`\nu_0=t/4`.

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

    Tup = numpy.zeros((nbasis, nbasis))
    Tdown = numpy.zeros((nbasis, nbasis))
    nu0 = 0.25*t

    for i in range(0, nbasis):
        # pinning field along y.
        xy1 = decode_basis(nx, ny, i)
        if (xy1[1] == 0 or xy1[1] == ny-1):
            Tup[i, i] += (-1.0)**(xy1[0]+xy1[1]) * nu0
            Tdown[i, i] += (-1.0)**(xy1[0]+xy1[1]+1) * nu0
        for j in range(i+1, nbasis):
            xy2 = decode_basis(nx, ny, j)
            dij = abs(xy1-xy2)
            if sum(dij) == 1:
                Tup[i, j] = Tdown[i,j] = -t
            # periodic bcs in x.
            if (dij==[nx-1, 0]).all():
                Tup[i, j] += -t
                Tdown[i, j] += -t

    return numpy.array([Tup+numpy.triu(Tup,1).T, Tdown+numpy.triu(Tdown,1).T])

def kinetic_pinning_alt(t, nbasis, nx, ny):
    r"""Kinetic part of the Hamiltonian in our one-electron basis.

    Adds pinning fields as outlined in [Qin16]_. This forces periodic boundary
    conditions along x and open boundary conditions along y. Pinning fields are
    applied in the y direction as:

        .. math::
            \nu_{i\uparrow} = -\nu_{i\downarrow} = (-1)^{i_x+i_y}\nu_0,

    for :math:`i_y=1,L_y` and :math:`\nu_0=t/4`.

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

    Tup = numpy.zeros((nbasis, nbasis))
    Tdown = numpy.zeros((nbasis, nbasis))
    h = 0.1*t

    for i in range(0, nbasis):
        # pinning field along y direction when i_x = 0.
        xy1 = decode_basis(nx, ny, i)
        if xy1[0] == 0:
            Tup[i,i] += (-1.0)**(xy1[1]) * h
            Tdown[i,i] += (-1.0)**(xy1[1]+1) * h
        for j in range(i+1, nbasis):
            xy2 = decode_basis(nx, ny, j)
            dij = abs(xy1-xy2)
            if sum(dij) == 1:
                Tup[i,j] = Tdown[i,j] = -t
            # periodic bcs in y.
            if (dij==[0, ny-1]).all():
                Tup[i,j] += -t
                Tdown[i,j] += -t

    return numpy.array([Tup+numpy.triu(Tup,1).T, Tdown+numpy.triu(Tdown,1).T])

def decode_basis(nx, ny, i):
    """Return cartesian lattice coordinates from basis index.

    Consider a 3x3 lattice then we index lattice sites like::

        (0,2) (1,2) (2,2)       6 7 8
        (0,1) (1,1) (2,1)  ->   3 4 5
        (0,0) (1,0) (2,0)       0 1 2

    i.e., i = i_x + n_x * i_y, and i_x = i%n_x, i_y = i//nx.

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
        return numpy.array([i%nx, i//nx])

def encode_basis(i, j, nx):
    """Encode 2d index to one dimensional index.

    See decode basis for layout.

    Parameters
    ----------
    i : int
        x coordinate.
    j : int
        y coordinate
    nx : int
        Number of x lattice sites.

    Returns
    -------
    ix : int
        basis index.
    """
    return i + j*nx

def _super_matrix(U, nbasis):
    '''Construct super-matrix from v_{ijkl}'''

def kpoints(t, nx, ny):
    """ Construct kpoints for system.

    Parameters
    ----------
    t : float
        Hopping amplitude.
    nx : int
        Number of x lattice sites.
    nx : int
        Number of y lattice sites.

    Returns
    -------
    kp : numpy array
        System kpoints Note these are not sorted according to eigenvalue energy
        but rather so as to conform with numpys default kpoint indexing for FFTs.
    kfac : float
        Kpoint scaling factor (2pi/L).
    eigs : numpy array
        Single particle eigenvalues associated with kp.
    """
    kp = []
    eigs = []
    if ny == 1:
        kfac = numpy.array([2.0*pi/nx])
        for n in range(0, nx):
            kp.append(numpy.array([n]))
            eigs.append(ek(t, n, kfac, ny))
    else:
        kfac = numpy.array([2.0*pi/nx, 2.0*pi/ny])
        for n in range(0, nx):
            for m in range(0, ny):
                k = numpy.array([n, m])
                kp.append(k)
                eigs.append(ek(t, k, kfac, ny))

    eigs = numpy.array(eigs)
    kp = numpy.array(kp)
    return (kp, kfac, eigs)


def ek(t, k, kc, ny):
    """ Calculate single-particle energies.

    Parameters
    ----------
    t : float
        Hopping amplitude.
    k : numpy array
        Kpoint.
    kc : float
        Scaling factor.
    ny : int
        Number of y lattice points.
    """
    if ny == 1:
        e = -2.0*t*cos(kc*k)
    else:
        e = -2.0*t*(cos(kc[0]*k[0])+cos(kc[1]*k[1]))

    return e

def get_strip(cfunc, cfunc_err, ix, nx, ny, stag=False):
    iy = [i for i in range(ny)]
    idx = [encode_basis(ix,i,nx) for i in iy]
    if stag:
        c = [((-1)**(ix+i))*cfunc[ib] for (i, ib) in zip(iy,idx)]
    else:
        c = [cfunc[ib] for ib in idx]
    cerr = [cfunc_err[ib] for ib in idx]
    return c, cerr


def unit_test():
    import itertools
    from pauxy.systems.hubbard import Hubbard
    from pauxy.estimators.ci import simple_fci_bose_fermi, simple_fci, simple_lang_firsov, simple_lang_firsov_unitary
    import scipy
    import numpy
    import scipy.sparse.linalg
    import pandas as pd

    # lmbdas = [0.5, 0.3, 0.8, 1.0]
    # w0s = [0.1, 0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0, 4.0]
    # lmbdas = [0.8,1.0]
    lmbdas = [0.1]
    # w0s = [0.1, 0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0, 4.0]
    # w0s = [100.0]
    w0s = [0.1]

    df = pd.DataFrame()

    for il, lmbda in enumerate(lmbdas):
        for iw0, w0 in enumerate(w0s):
            print ("w0 = {}".format(w0))
            options = {
            "name": "HubbardHolstein",
            "nup": 1,
            "ndown": 1,
            "nx": 2,
            "ny": 1,
            "U": 0.0,
            "t": 1.0,
            "w0": w0,
            "lambda": lmbda,
            "lang_firsov":False,
            "xpbc" :True,
            "ypbc" :True
            }

            system = HubbardHolstein (options, verbose=True)
            system0 = Hubbard (options, verbose=True)

            # Ueff = system0.U + system.gamma**2 * system.w0 - 2.0 * system.g * system.gamma * numpy.sqrt(2.0 * system.m * system.w0)
            # d = system.gamma**2 * system.w0 / 2.0 - system.g * system.gamma * numpy.sqrt(2.0 * system.m * system.w0)
            # system0.H1[0] = numpy.zeros_like(system0.T[0])
            # system0.H1[1] = numpy.zeros_like(system0.T[1])
            # system0.H1[0] = numpy.array(system0.T[0] + numpy.diag(numpy.eye(system.nbasis)*d))
            # system0.H1[1] = numpy.array(system0.T[1] + numpy.diag(numpy.eye(system.nbasis)*d))
            # system0.U = Ueff

            (eig, evec), H = simple_fci(system0, hamil=True)

            # print("eig = {}".format(eig[0]))
            # exit()
            # print("H w/o boson = {}".format(H))
            # nbosons = [5,7,10]
            nbosons = [2,3,10,20,30]
            # nbosons = [20]
            eigs = []
            eigs += [eig[0]]
            for nboson in nbosons:
                # print("# nboson = {}".format(nboson))
                (eig, evec), H = simple_fci_bose_fermi(system, nboson_max=nboson, hamil=True)
                # (eig, evec), H = simple_lang_firsov(system, nboson_max=nboson, hamil=True)
                # (eig, evec), H = simple_lang_firsov_unitary(system, nboson_max=nboson, hamil=True)
                # print(H)
                # print("eig = {}".format(eig[0]))
                eigs += [eig[0]]
            nbosons = [0] + nbosons
            
            w = [w0 for i in range(len(nbosons))]
            l = [lmbda for i in range(len(nbosons))]
            # if (iw0 == 0):
                # df = pd.DataFrame({"lambda":l, "w0":w, "nbosons":nbosons, 
                #                  "E":eigs})
            # else:
            df0 = pd.DataFrame({"lambda":lmbda, "w0":w0, "nbosons":nbosons, 
                             "E":eigs})
            df = df.append(df0)

            print(df.to_string(index=False))
            # alpha = numpy.sqrt(system.w0 * 2.0) * system.g 
            # shift = alpha * alpha / (2.0 * system.w0**2) * system.nbasis
            # print("shift = {}".format(shift))

            # print(eigs)

if __name__=="__main__":
    unit_test()
