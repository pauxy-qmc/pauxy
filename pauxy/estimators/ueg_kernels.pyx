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