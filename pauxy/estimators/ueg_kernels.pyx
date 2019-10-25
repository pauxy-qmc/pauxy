import numpy
cimport numpy
import math
import itertools
from pauxy.estimators.utils import convolve

DTYPE_CX = numpy.complex128
DTYPE = numpy.float64


def vq(numpy.ndarray q):
    assert(q.shape[0] == 3)
    cdef double q2 = numpy.dot(q, q)
    if (q2 < 1e-10):
        return 0.0
    else:
        return 4*math.pi / q2

def mod_one_body(numpy.ndarray T, numpy.ndarray basis, double vol, double kfac):
    """ Add a diagonal term of two-body Hamiltonian to the one-body term
    Parameters
    ----------
    T : float
        one-body Hamiltonian (i.e. kinetic energy)
    Returns
    -------
    h1e_mod: float
        modified one-body Hamiltonian
    """

    cdef size_t nbsf = basis.shape[0]
    cdef numpy.ndarray h1e_mod = T.copy()
    cdef double fac = 1.0 / (2.0 * vol)

    for (i, ki) in enumerate(basis):
        for (j, kj) in enumerate(basis):
            if i != j:
                q = kfac * (ki - kj)
                h1e_mod[i,i] = h1e_mod[i,i] - fac * vq(q)

    return h1e_mod

def coulomb_greens_function_per_qvec(numpy.ndarray kpq_i, numpy.ndarray kpq, numpy.ndarray pmq_i, numpy.ndarray pmq, double complex[:,:] G):

    cdef int nq = kpq_i.shape[0]

    cdef int idxkpq, idxpmq, i, j, iq

    cdef double complex Gkpq = 0.0
    cdef double complex Gpmq = 0.0

    for (idxkpq,i) in zip(kpq,kpq_i):
        Gkpq += G[i,idxkpq]
    for (idxpmq,i) in zip(pmq,pmq_i):
        Gpmq += G[i,idxpmq]

    return Gkpq, Gpmq

def exchange_greens_function_per_qvec(long[:] kpq_i, long[:] kpq, long[:] pmq_i, long[:] pmq, double complex[:,:] G):

    cdef int nkpq = kpq_i.shape[0]
    cdef int npmq = pmq_i.shape[0]

    cdef double complex Gprod = 0.0

    cdef int idxkpq, idxpmq, i, j

    for inkpq in range(nkpq):
        idxkpq = kpq[inkpq]
        i = kpq_i[inkpq]
        for jnpmq in range(npmq):
            idxpmq = pmq[jnpmq]
            j = pmq_i[jnpmq]
            Gprod += G[j,idxkpq]*G[i,idxpmq]

    return Gprod

def exchange_greens_function_fft (long nocc, long nbsf,
    long[:] mesh, long[:] qmesh, long[:] gmap, long[:] qmap,
    double complex[:,:] CTdagger, double complex[:,:] Ghalf):

    assert (mesh.shape[0] == 3)
    assert (qmesh.shape[0] == 3)
    assert (Ghalf.shape[0] == nocc)
    assert (Ghalf.shape[1] == nbsf)
    assert (CTdagger.shape[0] == nocc)
    assert (CTdagger.shape[1] == nbsf)

    cdef long ngrid = numpy.prod(mesh)
    cdef long nqgrid = numpy.prod(qmesh)

    cdef long nq = qmap.shape[0]
    cdef numpy.ndarray Gprod = numpy.zeros(nq, dtype=DTYPE_CX)

    cdef numpy.ndarray Gh_i = numpy.zeros(nbsf, dtype=DTYPE_CX)
    cdef numpy.ndarray CTdagger_j = numpy.zeros(nbsf, dtype=DTYPE_CX)
    cdef numpy.ndarray Gh_i_cube = numpy.zeros(ngrid, dtype=DTYPE_CX)
    cdef numpy.ndarray CTdagger_j_cube = numpy.zeros(ngrid, dtype=DTYPE_CX)

    cdef numpy.ndarray Gh_j = numpy.zeros(nbsf, dtype=DTYPE_CX)
    cdef numpy.ndarray CTdagger_i = numpy.zeros(nbsf, dtype=DTYPE_CX)
    cdef numpy.ndarray Gh_j_cube = numpy.zeros(ngrid, dtype=DTYPE_CX)
    cdef numpy.ndarray CTdagger_i_cube = numpy.zeros(ngrid, dtype=DTYPE_CX)

    cdef numpy.ndarray lQ_ji = numpy.zeros(nqgrid, dtype=DTYPE_CX)
    cdef numpy.ndarray lQ_ij = numpy.zeros(nqgrid, dtype=DTYPE_CX)

    for i in range(nocc):
        for j in range(nocc):
            Gh_i = numpy.flip(numpy.asarray(Ghalf[i,:]))
            CTdagger_j = numpy.asarray(CTdagger[j,:])

            Gh_i_cube = numpy.zeros(ngrid, dtype=DTYPE_CX)
            CTdagger_j_cube = numpy.zeros(ngrid, dtype=DTYPE_CX)

            Gh_i_cube[gmap] = Gh_i
            CTdagger_j_cube[gmap] = CTdagger_j

            lQ_ji = numpy.flip(convolve(CTdagger_j_cube, Gh_i_cube, mesh))[qmap]

            Gh_j = numpy.asarray(Ghalf[j,:])
            CTdagger_i = numpy.flip(numpy.asarray(CTdagger[i,:]))

            Gh_j_cube = numpy.zeros(ngrid, dtype=DTYPE_CX)
            CTdagger_i_cube = numpy.zeros(ngrid, dtype=DTYPE_CX)

            Gh_j_cube[gmap] = Gh_j
            CTdagger_i_cube[gmap] = CTdagger_i

            lQ_ij =  numpy.flip(convolve(Gh_j_cube, CTdagger_i_cube, mesh))[qmap]

            Gprod += lQ_ji*lQ_ij

    return Gprod
