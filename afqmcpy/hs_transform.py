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

def generic_continuous(, nmax_exp=4):

    # Generate ~M^2 normally distributed auxiliary fields.
    sigma = numpy.random.normal(0.0, 1.0, len(U))
    # Construct HS potential, V_HS = sigma dot U
    V_HS = numpy.einsum('ij,j->i', sigma, U)
    # Reshape so we can apply to MxN Slater determinant.
    V_HS = numpy.reshape(V_HS, (M,M))
    for n in range(1, nmax_exp+1):
        phi += numpy.factorial(n) * np.dot(V_HS, phi)
