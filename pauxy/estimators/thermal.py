import numpy
import scipy.linalg


def greens_function(A):
    I = numpy.identity(A.shape[-1])
    return numpy.array([scipy.linalg.inv(I+A[0]), scipy.linalg.inv(I+A[1])])

def one_rdm(A):
    I = numpy.identity(A.shape[-1])
    G = greens_function(A)
    return numpy.array([I-G[0].T, I-G[1].T])

def particle_number(dmat):
    nav = dmat[0].trace() + dmat[1].trace()
    return nav
