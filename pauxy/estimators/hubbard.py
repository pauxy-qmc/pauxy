import cmath
import numpy

def local_energy_hubbard(system, G, Ghalf=None):
    r"""Calculate local energy of walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    G : :class:`numpy.ndarray`
        Walker's "Green's function"

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    ke = numpy.sum(system.T[0] * G[0] + system.T[1] * G[1])
    # Todo: Stupid
    if system.symmetric:
        pe = -0.5*system.U*(G[0].trace() + G[1].trace())
    pe = system.U * numpy.dot(G[0].diagonal(), G[1].diagonal())

    return (ke + pe, ke, pe)

def local_energy_hubbard_holstein(system, G, X, Lap, Ghalf=None):
    r"""Calculate local energy of walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    G : :class:`numpy.ndarray`
        Walker's "Green's function"

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    ke = numpy.sum(system.T[0] * G[0] + system.T[1] * G[1])

    if system.symmetric:
        pe = -0.5*system.U*(G[0].trace() + G[1].trace())

    pe = system.U * numpy.dot(G[0].diagonal(), G[1].diagonal())

    
    pe_ph = 0.5 * system.w0 ** 2 * numpy.sum(X * X)
    ke_ph = -0.5 * numpy.sum(Lap) - 0.5 * system.w0 * system.nbasis
    
    rho = G[0].diagonal() + G[1].diagonal()
    e_eph = - system.g * cmath.sqrt(system.w0 * 2.0) * numpy.dot(rho, X)


    etot = ke + pe + pe_ph + ke_ph + e_eph

    Eph = ke_ph + pe_ph
    Eel = ke + pe
    Eeb = e_eph

    # print("# etot = {}".format(etot))
    # print("# (etot, ke+ke_ph, pe+pe_ph+e_eph) = {}".format((etot, ke+ke_ph, pe+pe_ph+e_eph)))

    # print("# Eel, Eph, Eeb, Etot = {}, {}, {}, {}".format(Eel, Eph, Eeb, Eel+Eph+Eeb))
    # print("# ke_ph, pe_ph, e_ph = {}, {}, {}".format(ke_ph, pe_ph, ke_ph+pe_ph))
    # print("# Eeb = {}".format(Eeb))
    # print("# Eph, Eeb = {}, {}".format(Eph, Eeb))

    # if (numpy.abs(etot + 4.0) > 1e-6):
    #     print ("# (etot, ke+ke_ph, pe+pe_ph+e_eph) = {}".format((etot, ke+ke_ph, pe+pe_ph+e_eph)))
    #     print("# Eel, Eph, Eeb, etot = {}, {}, {}, {}".format(Eel, Eph, Eeb, etot))

    return (etot, ke+pe, ke_ph+pe_ph+e_eph)


def local_energy_hubbard_ghf(system, Gi, weights, denom):
    """Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    Gi : :class:`numpy.ndarray`
        Array of Walker's "Green's function"
    denom : float
        Overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    ke = numpy.einsum('i,ikl,kl->', weights, Gi, system.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(Gi[:,:system.nbasis,:system.nbasis], axis1=1, axis2=2)
    gdd = numpy.diagonal(Gi[:,system.nbasis:,system.nbasis:], axis1=1, axis2=2)
    gud = numpy.diagonal(Gi[:,system.nbasis:,:system.nbasis], axis1=1, axis2=2)
    gdu = numpy.diagonal(Gi[:,:system.nbasis,system.nbasis:], axis1=1, axis2=2)
    gdiag = guu*gdd - gud*gdu
    pe = system.U * numpy.einsum('j,jk->', weights, gdiag) / denom
    return (ke+pe, ke, pe)

def local_energy_hubbard_ghf_full(system, GAB, weights):
    r"""Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    GAB : :class:`numpy.ndarray`
        Matrix of Green's functions for different SDs A and B.
    weights : :class:`numpy.ndarray`
        Components of overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L, T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    denom = numpy.sum(weights)
    ke = numpy.einsum('ij,ijkl,kl->', weights, GAB, system.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(GAB[:,:,:system.nbasis,:system.nbasis], axis1=2,
                         axis2=3)
    gdd = numpy.diagonal(GAB[:,:,system.nbasis:,system.nbasis:], axis1=2,
                         axis2=3)
    gud = numpy.diagonal(GAB[:,:,system.nbasis:,:system.nbasis], axis1=2,
                         axis2=3)
    gdu = numpy.diagonal(GAB[:,:,:system.nbasis,system.nbasis:], axis1=2,
                         axis2=3)
    gdiag = guu*gdd - gud*gdu
    pe = system.U * numpy.einsum('ij,ijk->', weights, gdiag) / denom
    return (ke+pe, ke, pe)


def local_energy_multi_det(system, Gi, weights):
    """Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    Gi : :class:`numpy.ndarray`
        Array of Walker's "Green's function"
    weights : :class:`numpy.ndarray`
        Components of overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    denom = numpy.sum(weights)
    ke = numpy.einsum('i,ikl,kl->', weights, Gi, system.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(Gi[:,:,:system.nup], axis1=1,
                         axis2=2)
    gdd = numpy.diagonal(Gi[:,:,system.nup:], axis1=1,
                         axis2=2)
    pe = system.U * numpy.einsum('j,jk->', weights, guu*gdd) / denom
    return (ke+pe, ke, pe)
