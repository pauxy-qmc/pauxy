import numpy
import scipy.linalg


def greens_function(A):
    I = numpy.identity(A[0].shape[0])
    return numpy.array([scipy.linalg.inv(I+A[0]), scipy.linalg.inv(I+A[1])])

def particle_number(G):
    nav = G[0].trace() + G[1].trace()
    return nav
