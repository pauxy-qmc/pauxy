from math import cos, pi
import numpy

def kpoints(nx, ny):
    kp = []
    if ny == 1:
        kfac = numpy.array([2.0*pi/nx])
        for n in range(-nx//2, nx//2):
            kp.append(nx)
    else:
        kfac = numpy.array([2.0*pi/nx, 2.0*pi/ny])
        for n in range(-nx//2, nx//2):
            for m in range(-ny//2, ny//2):
                kp.append(numpy.array([nx, ny]))

    return (kp, kfac)

def single_particle_eigs(t, kpoints, kc, ny):
    ek = []
    if ny == 1:
        for k in kpoints:
            print (k, kc)
            e = -2.0*t*cos(kc*k)
            ek.append(e)
    else:
        for k in kpoints:
            e = -2.0*t*(cos(kc[0]*k[0])+cos(kc[1]*k[1]))
            ek.append(e)

    return ek
