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
    # for spin in [0,1]:
    (U1,S1,V1) = scipy.linalg.svd(A)
    T = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
    (U2,S2,V2) = scipy.linalg.svd(T)
    U3 = numpy.dot(U1, U2)
    D3 = numpy.diag(1.0/S2)
    V3 = numpy.dot(V2, V1)
    G = (V3.conj().T).dot(D3).dot(U3.conj().T)
    return G

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
    G = greens_function(A)
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
    return numpy.array([I-G[0].T, I-G[1].T])

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
