import numpy
import scipy.linalg


def greens_function_unstable(A):
    r"""Construct Green's function from density matrix.

    .. math::
        G_{ij} = \langle c_{i} c_j^{\dagger} \rangle \\
               = \left[\frac{1}{1+A}\right]_{ij}

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Density matrix (product of B matrices).

    Returns
    -------
    G : :class:`numpy.ndarray`
        Thermal Green's function.
    """
    I = numpy.identity(A.shape[-1])
    return numpy.array([scipy.linalg.inv(I+A[0]), scipy.linalg.inv(I+A[1])])

def greens_function(A):
    """Construct Greens function from density matrix.

    .. math::
        G_{ij} = \langle c_{i} c_j^{\dagger} \rangle \\
               = \left[\frac{1}{1+A}\right]_{ij}

    Uses stable algorithm from White et al. (1988)

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Density matrix (product of B matrices).

    Returns
    -------
    G : :class:`numpy.ndarray`
        Thermal Green's function.
    """
    G = numpy.zeros(A.shape, dtype=A.dtype)
    (U1,S1,V1) = scipy.linalg.svd(A)
    T = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
    (U2,S2,V2) = scipy.linalg.svd(T)
    U3 = numpy.dot(U1, U2)
    D3 = numpy.diag(1.0/S2)
    V3 = numpy.dot(V2, V1)
    G = (V3.conj().T).dot(D3).dot(U3.conj().T)
    return G

def inverse_greens_function(A):
    """Inverse greens function from A"""

    Ginv = numpy.zeros(A.shape, dtype=A.dtype)
    (U1,S1,V1) = scipy.linalg.svd(A)
    T = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
    (U2,S2,V2) = scipy.linalg.svd(T)
    U3 = numpy.dot(U1, U2)
    D3 = numpy.diag(S2)
    V3 = numpy.dot(V2, V1)
    Ginv = (V3.conj().T).dot(D3).dot(U3.conj().T)
    return Ginv

def inverse_greens_function_qr(A):
    """Inverse greens function from A"""

    Ginv = numpy.zeros(A.shape, dtype=A.dtype)
    
    (U1, V1) = scipy.linalg.qr(A, pivoting = False)
    V1inv = scipy.linalg.solve_triangular(V1, numpy.identity(V1.shape[0]))
    T = numpy.dot(U1.conj().T, V1inv) + numpy.identity(V1.shape[0])
    (U2, V2) = scipy.linalg.qr(T, pivoting = False)
    U3 = numpy.dot(U1, U2)
    V3 = numpy.dot(V2, V1)
    Ginv = U3.dot(V3)
    # (U1,S1,V1) = scipy.linalg.svd(A)
    # T = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
    # (U2,S2,V2) = scipy.linalg.svd(T)
    # U3 = numpy.dot(U1, U2)
    # D3 = numpy.diag(S2)
    # V3 = numpy.dot(V2, V1)
    # Ginv = (V3.conj().T).dot(D3).dot(U3.conj().T)
    return Ginv


def one_rdm(A):
    """Compute one-particle reduced density matrix

    .. math::
        rho_{ij} = \langle c_{i}^{\dagger} c_{j} \rangle \\
                 = 1 - G_{ji}
    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Density matrix (product of B matrices).

    Returns
    -------
    P : :class:`numpy.ndarray`
        Thermal 1RDM.
    """
    I = numpy.identity(A.shape[-1])
    G = numpy.array([greens_function(A[0]),greens_function(A[1])])
    return numpy.array([I-G[0].T, I-G[1].T])

def one_rdm_from_G(G):
    """Compute one-particle reduced density matrix from Green's function.

    .. math::
        rho_{ij} = \langle c_{i}^{\dagger} c_{j} \rangle \\
                 = 1 - G_{ji}
    Parameters
    ----------
    G : :class:`numpy.ndarray`
        Thermal Green's function.

    Returns
    -------
    P : :class:`numpy.ndarray`
        Thermal 1RDM.
    """
    I = numpy.identity(G.shape[-1])
    return numpy.array([I-G[0].T, I-G[1].T],dtype = numpy.complex128)

def particle_number(dmat):
    """Compute average particle number.

    Parameters
    ----------
    dmat : :class:`numpy.ndarray`
        Thermal 1RDM.

    Returns
    -------
    nav : float
        Average particle number.
    """
    nav = dmat[0].trace() + dmat[1].trace()
    return nav
