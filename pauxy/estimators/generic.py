import numpy

def local_energy_generic(system, G, Ghalf=None):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the full form for the two-electron integrals.

    For testing purposes only.

    Parameters
    ----------
    system : :class:`hubbard`
        System information for the hubbard model.
    G : :class:`numpy.ndarray`
        Walker's "green's function"

    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    e1 = (numpy.einsum('ij,ij->', system.H1[0], G[0]) +
          numpy.einsum('ij,ij->', system.H1[1], G[1]))
    euu = 0.5*(numpy.einsum('prqs,pr,qs->', system.h2e, G[0], G[0]) -
               numpy.einsum('prqs,ps,qr->', system.h2e, G[0], G[0]))
    edd = 0.5*(numpy.einsum('prqs,pr,qs->', system.h2e, G[1], G[1]) -
               numpy.einsum('prqs,ps,qr->', system.h2e, G[1], G[1]))
    eud = 0.5*numpy.einsum('prqs,pr,qs->', system.h2e, G[0], G[1])
    edu = 0.5*numpy.einsum('prqs,pr,qs->', system.h2e, G[1], G[0])
    e2 = euu + edd + eud + edu
    return (e1+e2+system.ecore, e1+system.ecore, e2)

def local_energy_generic_opt(system, G, Ghalf=None):
    # Element wise multiplication.
    e1b = numpy.sum(system.H1[0]*G[0]) + numpy.sum(system.H1[1]*G[1])
    Gup = Ghalf[0].ravel()
    Gdn = Ghalf[1].ravel()
    euu = 0.5 * Gup.dot(system.vaklb[0].dot(Gup))
    edd = 0.5 * Gdn.dot(system.vaklb[1].dot(Gdn))
    eud = 0.5 * numpy.dot(Gup.T*system.rchol_vecs[0],
                          Gdn.T*system.rchol_vecs[1])
    edu = 0.5 * numpy.dot(Gdn.T*system.rchol_vecs[1],
                          Gup.T*system.rchol_vecs[0])
    e2b = euu + edd + eud + edu
    return (e1b + e2b + system.ecore, e1b + system.ecore, e2b)

def local_energy_generic_cholesky(system, G, Ghalf=None):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals.

    Parameters
    ----------
    system : :class:`hubbard`
        System information for the hubbard model.
    G : :class:`numpy.ndarray`
        Walker's "green's function"

    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    # Element wise multiplication.
    e1b = numpy.sum(system.H1[0]*G[0]) + numpy.sum(system.H1[1]*G[1])
    cv = system.chol_vecs
    ecoul_uu = numpy.dot(numpy.sum(cv*G[0], axis=(1,2)),
                         numpy.sum(cv*G[0], axis=(1,2)))
    ecoul_dd = numpy.dot(numpy.sum(cv*G[1], axis=(1,2)),
                         numpy.sum(cv*G[1], axis=(1,2)))
    ecoul_ud = numpy.dot(numpy.sum(cv*G[0], axis=(1,2)),
                         numpy.sum(cv*G[1], axis=(1,2)))
    ecoul_du = numpy.dot(numpy.sum(cv*G[1], axis=(1,2)),
                         numpy.sum(cv*G[0], axis=(1,2)))
    exx_uu = 0
    for c in cv:
        # t1 = numpy.einsum('lpr,ps->lrs',cv,G[0])
        t1 = numpy.dot(c.T, G[0])
        exx_uu += numpy.dot(t1,t1).trace()
        # t2 = numpy.einsum('lqs,qr->lsr',cv,G[0])
        # exx = numpy.einsum('lrs,lsr')
    exx_dd = 0
    for c in cv:
        # t1 = numpy.einsum('lpr,ps->lrs',cv,G[0])
        t1 = numpy.dot(c.T, G[1])
        exx_dd += numpy.dot(t1,t1).trace()
        # t2 = numpy.einsum('lqs,qr->lsr',cv,G[0])
        # exx = numpy.einsum('lrs,lsr')
    # t1 = numpy.einsum('lpr,ps->lrs', cv, G[1])
    # exx_dd = numpy.einsum('lrs,lsr->', t1, t1)
    euu = 0.5*(ecoul_uu-exx_uu)
    edd = 0.5*(ecoul_dd-exx_dd)
    eud = 0.5 * ecoul_ud
    edu = 0.5 * ecoul_du
    e2b = euu + edd + eud + edu
    return (e1b+e2b+system.ecore, e1b+system.ecore, e2b)

def core_contribution(system, Gcore):
    hc_a = (numpy.einsum('pqrs,pq->rs', system.h2e, Gcore[0]) -
            0.5*numpy.einsum('prsq,pq->rs', system.h2e, Gcore[0]))
    hc_b = (numpy.einsum('pqrs,pq->rs', system.h2e, Gcore[1]) -
            0.5*numpy.einsum('prsq,pq->rs', system.h2e, Gcore[1]))
    return (hc_a, hc_b)

def core_contribution_cholesky(system, G):
    cv = system.chol_vecs
    hca_j = numpy.einsum('l,lij->ij', numpy.sum(cv*G[0], axis=(1,2)), cv)
    ta_k = numpy.einsum('lpr,pq->lrq', cv, G[0])
    hca_k = 0.5*numpy.einsum('lrq,lsq->rs', ta_k, cv)
    hca = hca_j - hca_k
    hcb_j = numpy.einsum('l,lij->ij', numpy.sum(cv*G[1], axis=(1,2)), cv)
    tb_k = numpy.einsum('lpr,pq->lrq', cv, G[1])
    hcb_k = 0.5*numpy.einsum('lrq,lsq->rs', tb_k, cv)
    hcb = hcb_j - hcb_k
    return (hca, hcb)
