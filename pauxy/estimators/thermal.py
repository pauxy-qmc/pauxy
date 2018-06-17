import numpy
import scipy.linalg


def greens_function(A):
    I = numpy.identity(A.shape[-1])
    return numpy.array([scipy.linalg.inv(I+A[0]), scipy.linalg.inv(I+A[1])])

def greens_function_stable(A):
    G = numpy.zeros(A.shape, dtype=A.dtype)
    for spin in [0,1]:
        (U1,S1,V1) = scipy.linalg.svd(A[spin])
        T = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
        (U2,S2,V2) = scipy.linalg.svd(T)
        U3 = numpy.dot(U1, U2)
        D3 = numpy.diag(1.0/S2)
        V3 = numpy.dot(V2, V1)
        G[spin] = (V3.conj().T).dot(D3).dot(U3.conj().T)
    return G

def one_rdm(A):
    I = numpy.identity(A.shape[-1])
    G = greens_function_stable(A)
    return numpy.array([I-G[0].T, I-G[1].T])

def one_rdm_from_G(G):
    I = numpy.identity(G.shape[-1])
    return numpy.array([I-G[0].T, I-G[1].T])

def particle_number(dmat):
    nav = dmat[0].trace() + dmat[1].trace()
    return nav
