import numpy
import sys

def local_energy_generic(h1e, eri, G, ecore=0.0, Ghalf=None):
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
    e1 = (numpy.einsum('ij,ij->', h1e[0], G[0]) +
          numpy.einsum('ij,ij->', h1e[1], G[1]))
    euu = 0.5*(numpy.einsum('prqs,pr,qs->', eri, G[0], G[0]) -
               numpy.einsum('prqs,ps,qr->', eri, G[0], G[0]))
    edd = 0.5*(numpy.einsum('prqs,pr,qs->', eri, G[1], G[1]) -
               numpy.einsum('prqs,ps,qr->', eri, G[1], G[1]))
    eud = 0.5*numpy.einsum('prqs,pr,qs->', eri, G[0], G[1])
    edu = 0.5*numpy.einsum('prqs,pr,qs->', eri, G[1], G[0])
    e2 = euu + edd + eud + edu
    return (e1+e2+ecore, e1+ecore, e2)

def local_energy_generic_opt(system, G, Ghalf=None):
    # Element wise multiplication.
    e1b = numpy.sum(system.H1[0]*G[0]) + numpy.sum(system.H1[1]*G[1])
    Gup = Ghalf[0].ravel()
    Gdn = Ghalf[1].ravel()
    euu = 0.5 * Gup.dot(system.vakbl[0].dot(Gup))
    edd = 0.5 * Gdn.dot(system.vakbl[1].dot(Gdn))
    # TODO: Fix this. Very dumb.
    eos =  numpy.dot((system.rchol_vecs[0].T).dot(Gup),
                     (system.rchol_vecs[1].T).dot(Gdn))

    e2b = euu + edd + eos #eud + edu
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
    ecoul_uu = 0
    ecoul_dd = 0
    ecoul_ud = 0
    ecoul_du = 0
    exx_uu = 0
    exx_dd = 0
    # Below to compute exx_uu/dd we do
    # t1 = numpy.einsum('nik,il->nkl', cv, G[0])
    # t2 = numpy.einsum('nlj,jk->nlk', cv.conj(), G[0])
    # exx_uu = numpy.einsum('nkl,nlk->', t1, t2)
    exx_uu = 0
    for c in cv:
        ecoul_uu += numpy.sum(c*G[0]) * numpy.sum(c.conj().T*G[0])
        ecoul_dd += numpy.sum(c*G[1]) * numpy.sum(c.conj().T*G[1])
        ecoul_ud += numpy.sum(c*G[0]) * numpy.sum(c.conj().T*G[1])
        ecoul_du += numpy.sum(c*G[1]) * numpy.sum(c.conj().T*G[0])
        t1 = numpy.dot(c.T, G[0])
        # print(t1.sum())
        t2 = numpy.dot(c.conj(), G[0])
        # print(t2.sum())
        exx_uu += numpy.einsum('ij,ji->',t1,t2)
        # print("sum:", exx_uu)
        t1 = numpy.dot(c.T, G[1])
        t2 = numpy.dot(c.conj(), G[1])
        exx_dd += numpy.einsum('ij,ji->',t1,t2)
    euu = 0.5 * (ecoul_uu-exx_uu)
    edd = 0.5 * (ecoul_dd-exx_dd)
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

def fock_generic(system, P):
    if system.sparse:
        mf_shift = 1j*P[0].ravel()*system.hs_pot
        mf_shift += 1j*P[1].ravel()*system.hs_pot
        VMF = 1j*system.hs_pot.dot(mf_shift).reshape(system.nbasis,system.nbasis)
    else:
        mf_shift = 1j*numpy.einsum('lpq,spq->l', system.hs_pot, P)
        VMF = 1j*numpy.einsum('lpq,l->pq', system.hs_pot, mf_shift)
    return system.h1e_mod - VMF
