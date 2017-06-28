from math import cos, pi
import numpy

def kpoints(t, nx, ny):
    kp = []
    eigs = []
    if ny == 1:
        kfac = numpy.array([2.0*pi/nx])
        for n in range(-int(nx/2), int(nx/2)+1):
            kp.append(n)
            eigs.append(ek(t, n, kfac, ny))
    else:
        kfac = numpy.array([2.0*pi/nx, 2.0*pi/ny])
        for n in range(-int(nx/2), int(nx/2)+1):
            for m in range(-int(ny/2), int(ny/2)):
                k = numpy.array([n, m])
                kp.append(k)
                eigs.append(ek(t, k, kfac, ny))

    eigs = numpy.array(eigs)
    kp = numpy.array(kp)
    idx = eigs.argsort()
    eigs = eigs[idx]
    kp = kp[idx]
    return (kp, kfac, eigs)

def ek(t, k, kc, ny):
    if ny == 1:
        e = -2.0*t*cos(kc*k)
    else:
        e = -2.0*t*(cos(kc[0]*k[0])+cos(kc[1]*k[1]))

    return e
