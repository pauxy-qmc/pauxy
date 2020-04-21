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

def local_energy_generic_cholesky_opt(system, G, Ghalf=None, rchol=None):
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
    if rchol is None:
        rchol = system.rchol_vecs
    nalpha, nbeta= system.nup, system.ndown
    nbasis = system.nbasis
    Ga, Gb = Ghalf[0], Ghalf[1]
    Xa = rchol[0].T.dot(Ga.ravel())
    Xb = rchol[1].T.dot(Gb.ravel())
    ecoul = numpy.dot(Xa,Xa)
    ecoul += numpy.dot(Xb,Xb)
    ecoul += 2*numpy.dot(Xa,Xb)
    if system.sparse:
        rchol_a, rchol_b = [rchol[0].toarray(), rchol[1].toarray()]
    else:
        rchol_a, rchol_b = rchol[0], rchol[1]
    # T_{abn} = \sum_k Theta_{ak} LL_{ak,n}
    # LL_{ak,n} = \sum_i L_{ik,n} A^*_{ia}
    Ta = numpy.tensordot(Ga, rchol_a.reshape((nalpha,nbasis,-1)), axes=((1),(1)))
    exxa = numpy.tensordot(Ta, Ta, axes=((0,1,2),(1,0,2)))
    Tb = numpy.tensordot(Gb, rchol_b.reshape((nbeta,nbasis,-1)), axes=((1),(1)))
    exxb = numpy.tensordot(Tb, Tb, axes=((0,1,2),(1,0,2)))
    exx = exxa + exxb
    e2b = 0.5 * (ecoul - exx)
    return (e1b + e2b + system.ecore, e1b + system.ecore, e2b)

def local_energy_generic_cholesky_opt_stochastic(system, G, nsamples, Ghalf=None, rchol=None):
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
    
    if rchol is None:
        rchol = system.rchol_vecs

    nalpha, nbeta= system.nup, system.ndown
    nbasis = system.nbasis
    
    Ga, Gb = Ghalf[0], Ghalf[1]
    Xa = rchol[0].T.dot(Ga.ravel())
    Xb = rchol[1].T.dot(Gb.ravel())
    ecoul = numpy.dot(Xa,Xa)
    ecoul += numpy.dot(Xb,Xb)
    ecoul += 2*numpy.dot(Xa,Xb)

    # O(ON s)
    # Xa = numpy.tensordot(ra, Ga, axes=((0,1),(0,1)))
    # Xb = numpy.tensordot(rb, Gb, axes=((0,1),(0,1)))
    # ecoul = numpy.dot(Xa,Xa)
    # ecoul += numpy.dot(Xb,Xb)
    # ecoul += 2*numpy.dot(Xa,Xb)

    naux = rchol[0].shape[-1]
    theta = numpy.zeros((naux,nsamples), dtype=numpy.int64)
    for i in range(nsamples):
        theta[:,i] = (2*numpy.random.randint(0,2,size=(naux))-1)

    if system.sparse:
        rchol_a, rchol_b = [rchol[0].toarray(), rchol[1].toarray()]
    else:
        rchol_a, rchol_b = rchol[0], rchol[1]
    
    rchol_a = rchol_a.reshape((nalpha,nbasis, naux))
    rchol_b = rchol_b.reshape((nbeta,nbasis, naux))

    # ra = numpy.tensordot(rchol_a, theta, axes=((2),(0))) * numpy.sqrt(1.0/nsamples)
    # rb = numpy.tensordot(rchol_b, theta, axes=((2),(0))) * numpy.sqrt(1.0/nsamples)
    ra = numpy.einsum("ipX,Xs->ips",rchol_a, theta, optimize=True) * numpy.sqrt(1.0/nsamples)
    rb = numpy.einsum("ipX,Xs->ips",rchol_b, theta, optimize=True) * numpy.sqrt(1.0/nsamples)
    Gra = numpy.einsum("kq,lqx->lkx", Ga, ra, optimize=True)    
    Grb = numpy.einsum("kq,lqx->lkx", Gb, rb, optimize=True)    

    # Gra = numpy.tensordot(Ga, ra, axes=((1),(1))).transpose(1,0,2)
    # Grb = numpy.tensordot(Gb, rb, axes=((1),(1))).transpose(1,0,2)
    exxa = numpy.tensordot(Gra, Gra, axes=((0,1,2),(1,0,2)))
    exxb = numpy.tensordot(Grb, Grb, axes=((0,1,2),(1,0,2)))

    exx = exxa + exxb
    e2b = 0.5 * (ecoul - exx)

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
    nalpha, nbeta = system.nup, system.ndown
    nbasis = system.nbasis
    cv = system.chol_vecs.reshape((-1,nbasis*nbasis))
    Ga, Gb = G[0], G[1]
    Xa = cv.dot(Ga.ravel())
    Xb = cv.dot(Gb.ravel())
    ecoul = numpy.dot(Xa,Xa)
    ecoul += numpy.dot(Xb,Xb)
    ecoul += 2*numpy.dot(Xa,Xb)
    # T[n,l,k] = \sum_i L[n,i,k] G[i,l]
    # exx  = \sum_{nlk} T[n,l,k] T[n,k,l]
    Ta = numpy.tensordot(system.chol_vecs, Ga, axes=((1),(1)))
    exxa = numpy.tensordot(Ta, Ta, axes=((0,1,2),(0,2,1)))
    Tb = numpy.tensordot(system.chol_vecs, Gb, axes=((1),(1)))
    exxb = numpy.tensordot(Tb, Tb, axes=((0,1,2),(0,2,1)))
    exx = exxa + exxb
    e2b = 0.5 * (ecoul - exx)
    return (e1b+e2b+system.ecore, e1b+system.ecore, e2b)

def core_contribution(system, Gcore):
    hc_a = (numpy.einsum('pqrs,pq->rs', system.h2e, Gcore[0]) -
            0.5*numpy.einsum('prsq,pq->rs', system.h2e, Gcore[0]))
    hc_b = (numpy.einsum('pqrs,pq->rs', system.h2e, Gcore[1]) -
            0.5*numpy.einsum('prsq,pq->rs', system.h2e, Gcore[1]))
    return (hc_a, hc_b)

def core_contribution_cholesky(chol, G):
    nb = G[0].shape[-1]
    cmat = chol.reshape((-1,nb*nb))
    X = numpy.dot(cmat, G[0].ravel())
    Ja = numpy.dot(cmat.T, X).reshape(nb,nb)
    T = numpy.tensordot(chol, G[0], axes=((1),(0)))
    Ka = numpy.tensordot(T, chol, axes=((0,2),(0,2)))
    hca = Ja - 0.5 * Ka
    X = numpy.dot(cmat, G[1].ravel())
    Jb = numpy.dot(cmat.T, X).reshape(nb,nb)
    T = numpy.tensordot(chol, G[1], axes=((1),(0)))
    Kb = numpy.tensordot(T, chol, axes=((0,2),(0,2)))
    hcb = Jb - 0.5 * Kb
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
