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
    e1 = (numpy.einsum('ij,ij->', system.T[0], G[0]) +
          numpy.einsum('ij,ij->', system.T[1], G[1]))
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
    e1b = numpy.sum(system.T[0]*G[0]) + numpy.sum(system.T[1]*G[1])
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
    e1b = numpy.sum(system.T[0]*G[0]) + numpy.sum(system.T[1]*G[1])
    euu = 0.5*(numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                            system.chol_vecs, G[0], G[0]) -
               numpy.einsum('lpr,lqs,ps,qr->', system.chol_vecs,
                            system.chol_vecs, G[0], G[0]))
    edd = 0.5*(numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                            system.chol_vecs, G[1], G[1]) -
               numpy.einsum('lpr,lqs,ps,qr->', system.chol_vecs,
                            system.chol_vecs, G[1], G[1]))
    eud = 0.5*numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                           system.chol_vecs, G[0], G[1])
    edu = 0.5*numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                           system.chol_vecs, G[1], G[0])
    e2b = euu + edd + eud + edu
    return (e1b+e2b+system.ecore, e1b+system.ecore, e2b)

def local_energy_multi_det_full(system, A, B, coeffsA, coeffsB):
    weight = 0
    energies = 0
    for ix, (Aix, cix) in enumerate(zip(A, coeffsA)):
        for iy, (Biy, ciy) in enumerate(zip(B, coeffsB)):
            # construct "local" green's functions for each component of A
            inv_O = scipy.linalg.inv((Aix.conj().T).dot(Biy))
            GAB = (Biy.dot(inv_O)).dot(Aix.conj().T)
            weight = cix*(ciy.conj()) / scipy.linalg.det(inv_O)
            energies += weight * numpy.array(local_energy_generic(system, GAB))
            denom += weight
    return tuple(energies/denom)

def local_energy_multi_det(system, Gi, weights):
    weight = 0
    energies = 0
    for w, G in zip(weights, Gi):
        # construct "local" green's functions for each component of A
        energies += w * numpy.array(local_energy_generic(system, G))
        denom += w
    return tuple(energies/denom)

def core_contribution(system, Gcore):
    hc_a = (numpy.einsum('pqrs,pq->rs', system.h2e, Gcore[0]) -
            0.5*numpy.einsum('prsq,pq->rs', system.h2e, Gcore[0]))
    hc_b = (numpy.einsum('pqrs,pq->rs', system.h2e, Gcore[1]) -
            0.5*numpy.einsum('prsq,pq->rs', system.h2e, Gcore[1]))
    return (hc_a, hc_b)
