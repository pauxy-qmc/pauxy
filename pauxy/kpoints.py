from math import cos, pi
import numpy


def kpoints(t, nx, ny):
    """ Construct kpoints for system.

    Parameters
    ----------
    t : float
        Hopping amplitude.
    nx : int
        Number of x lattice sites.
    nx : int
        Number of y lattice sites.

    Returns
    -------
    kp : numpy array
        System kpoints Note these are not sorted according to eigenvalue energy
        but rather so as to conform with numpys default kpoint indexing for FFTs.
    kfac : float
        Kpoint scaling factor (2pi/L).
    eigs : numpy array
        Single particle eigenvalues associated with kp.
    """
    kp = []
    eigs = []
    if ny == 1:
        kfac = numpy.array([2.0*pi/nx])
        for n in range(0, nx):
            kp.append(numpy.array([n]))
            eigs.append(ek(t, n, kfac, ny))
    else:
        kfac = numpy.array([2.0*pi/nx, 2.0*pi/ny])
        for n in range(0, nx):
            for m in range(0, ny):
                k = numpy.array([n, m])
                kp.append(k)
                eigs.append(ek(t, k, kfac, ny))

    eigs = numpy.array(eigs)
    kp = numpy.array(kp)
    return (kp, kfac, eigs)


def ek(t, k, kc, ny):
    """ Calculate single-particle energies.

    Parameters
    ----------
    t : float
        Hopping amplitude.
    k : numpy array
        Kpoint.
    kc : float
        Scaling factor.
    ny : int
        Number of y lattice points.
    """
    if ny == 1:
        e = -2.0*t*cos(kc*k)
    else:
        e = -2.0*t*(cos(kc[0]*k[0])+cos(kc[1]*k[1]))

    return e
