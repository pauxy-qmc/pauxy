'''Discrete and continuous Hubbard-Stratonovich transformations.'''

import numpy
import scipy.linalg
import random

def construct_generic_one_body(gamma, method='eigen'):
    '''Construct one-body operators from super matrix Gamma

Parameters
----------
gamma : numpy.ndarray
    Supermatrix whose elements contain two-electron matrix elements:
    :math:`\Gamma_{ab} = \Gamma_{(ik)(lj)} = v_{ijkl}`. Assumes the compound
    index a and b are appropriately defined and that Gamma is Hermitian.
method : string
    How to factorise two-body interaction. Default is via eigen decomposition.

Returns
-------
U : numpy.ndarray
    Matrix containing eigenvectors of gamma (times :math:`\sqrt{-\lambda}`.
'''

    if method == 'eigen':
        (eigs, U) = scipy.linalg.eigh(gamma)
        # Multiply the columns of U by sqrt(-lambda_i), i.e,
        # U_ij = U_ij\sqrt{-\lambda_j}
        U = numpy.einsum('ij,j->ij', eigv, numpy.sqrt(-eigs))

        return U

