'''Various useful routines maybe not appropriate elsewhere'''

import numpy
import scipy.linalg

def sherman_morrison(Ainv, u, vt):
    '''Sherman-Morrison update of a matrix inverse

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
    Update inverse: :math:`(A + u x vT)^{-1}`
'''

    return (
        Ainv - (Ainv.dot(numpy.outer(u,vt)).dot(Ainv))/(1.0+vt.dot(Ainv).dot(u))
    )


def diagonalise_sorted(H):
    '''Diagonalise Hermitian matrix H and return sorted eigenvalues and vectors.

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
'''

    (eigs, eigv) = scipy.linalg.eigh(H)
    idx = eigs.argsort()
    eigs = eigs[idx]
    eigv = eigv[:,idx]

    return (eigs, eigv)

def format_fixed_width_strings(strings):
    return ' '.join('{:>17}'.format(s) for s in strings)

def format_fixed_width_floats(floats):
    return ' '.join('{: .10e}'.format(f) for f in floats)
