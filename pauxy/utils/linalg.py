import numpy
import scipy.linalg

def sherman_morrison(Ainv, u, vt):
    r"""Sherman-Morrison update of a matrix inverse:

    .. math::
        (A + u \otimes v)^{-1} = A^{-1} - \frac{A^{-1}u v^{T} A^{-1}}
                                               {1+v^{T}A^{-1} u}

    Parameters
    ----------
    Ainv : numpy.ndarray
        Matrix inverse of A to be updated.
    u : numpy.array
        column vector
    vt : numpy.array
        transpose of row vector

    Returns
    -------
    Ainv : numpy.ndarray
        Updated matrix inverse.
    """

    return (
        Ainv - (Ainv.dot(numpy.outer(u,vt)).dot(Ainv))/(1.0+vt.dot(Ainv).dot(u))
    )


def diagonalise_sorted(H):
    """Diagonalise Hermitian matrix H and return sorted eigenvalues and vectors.

    Eigenvalues are sorted as e_1 < e_2 < .... < e_N, where H is an NxN
    Hermitian matrix.

    Parameters
    ----------
    H : :class:`numpy.ndarray`
        Hamiltonian matrix to be diagonalised.

    Returns
    -------
    eigs : :class:`numpy.array`
        Sorted eigenvalues
    eigv :  :class:`numpy.array`
        Sorted eigenvectors (same sorting as eigenvalues).
    """

    (eigs, eigv) = scipy.linalg.eigh(H)
    idx = eigs.argsort()
    eigs = eigs[idx]
    eigv = eigv[:, idx]

    return (eigs, eigv)


def regularise_matrix_inverse(A, cutoff=1e-10):
    """Perform inverse of singular matrix.

    First compute SVD of input matrix then add a tuneable cutoff which washes
    out elements whose singular values are close to zero.

    Parameters
    ----------
    A : class:`numpy.array`
        Input matrix.
    cutoff : float
        Cutoff parameter.

    Returns
    -------
    B : class:`numpy.array`
        Regularised matrix inverse (pseudo-inverse).
    """
    (U, D, V) = scipy.linalg.svd(A)
    D = D / (cutoff**2.0 + D**2.0)
    return (V.conj().T).dot(numpy.diag(D)).dot(U.conj().T)


def reortho(A):
    (A, R) = scipy.linalg.qr(A, mode='economic')
    signs = numpy.diag(numpy.sign(numpy.diag(R)))
    A = A.dot(signs)
    detR = scipy.linalg.det(signs.dot(R))
    return detR


def reortho(A):
    """Reorthogonalise a MxN matrix A.

    Performs a QR decomposition of A. Note that for consistency elsewhere we
    want to preserve detR > 0 which is not guaranteed. We thus factor the signs
    of the diagonal of R into Q.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        MxN matrix.

    Returns
    -------
    Q : :class:`numpy.ndarray`
        Orthogonal matrix. A = QR.
    detR : float
        Determinant of upper triangular matrix (R) from QR decomposition.
    """
    (Q, R) = scipy.linalg.qr(A, mode='economic')
    signs = numpy.diag(numpy.sign(numpy.diag(R)))
    Q = Q.dot(signs)
    detR = scipy.linalg.det(signs.dot(R))
    return (Q, detR)


def modified_cholesky(M, kappa, verbose=False):
    """Modified cholesky decomposition of matrix.

    See, e.g. [Motta17]_

    Parameters
    ----------
    M : :class:`numpy.ndarray`
        Positive semi-definite, symmetric matrix.
    kappa : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors.
    """
    # matrix of residuals.
    delta = numpy.copy(M)
    # index of largest diagonal element of residual matrix.
    nu = numpy.argmax(delta.diagonal())
    delta_max = delta[nu,nu]
    if verbose:
        print ("# iteration %d: delta_max = %f"%(0, delta_max))
    # Store for current approximation to input matrix.
    Mapprox = numpy.zeros(M.shape)
    chol_vecs = []
    nchol = 0
    while abs(delta_max) > kappa:
        # Update cholesky vector
        L = (numpy.copy(delta[:,nu])/(delta_max)**0.5)
        chol_vecs.append(L)
        Mapprox += numpy.einsum('i,j->ij', L, L)
        delta = M - Mapprox
        nu = numpy.argmax(delta.diagonal())
        delta_max = delta[nu,nu]
        nchol += 1
        if verbose:
            print ("# iteration %d: delta_max = %f"%(nchol, delta_max))

    return numpy.array(chol_vecs)

def exponentiate_matrix(M, order=6):
    """Taylor series approximation for matrix exponential"""
    T = numpy.copy(M)
    EXPM = numpy.identity(M.shape[0], dtype=M.dtype)
    for n in range(1, order+1):
        EXPM += T
        T = M.dot(T) / (n+1)
    return EXPM
