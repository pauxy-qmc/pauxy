import time
import numpy
try:
    from mpi4py import MPI
    mpi_sum = MPI.SUM
except ImportError:
    mpi_sum = None
import scipy.linalg

def local_energy_ueg(system, G):
    
    ke = numpy.sum(system.T[0] * G[0] + system.T[1] * G[1]) # kinetic energy
    
    Gkpq =  [numpy.zeros(len(system.qvecs), dtype=numpy.complex128) for i in range(2)]
    Gpmq =  [numpy.zeros(len(system.qvecs), dtype=numpy.complex128) for i in range(2)]
    Gprod = [numpy.zeros(len(system.qvecs), dtype=numpy.complex128) for i in range(2)]

    ne = [system.nup, system.ndown]

    G[0] = G[0].T
    G[1] = G[1].T

#   Todo: make it work for different spin
    kf = system.basis[0:ne[0]]

    ikpq = []
    for (iq, q) in enumerate(system.qvecs):
        idxkpq =[]
        for i, k in enumerate(kf):
            kpq = k + q
            idx = system.lookup_basis(kpq)
            if idx is not None:
                idxkpq += [(i,idx)]
        ikpq += [idxkpq]

    # print(ikpq)
    # exit()

    ipmq = []
    for (iq, q) in enumerate(system.qvecs):
        idxpmq =[]
        for i, p in enumerate(kf):
            pmq = p - q
            idx = system.lookup_basis(pmq)
            if idx is not None:
                idxpmq += [(i,idx)]
        ipmq += [idxpmq]

    for s in [0, 1]:
        for (iq, q) in enumerate(system.qvecs):
            for (i, idxkpq) in ikpq[iq]:
                Gkpq[s][iq] += G[s][idxkpq,i]
                for (j,idxpmq) in ipmq[iq]:
                    Gprod[s][iq] += G[s][idxkpq,j]*G[s][idxpmq,i]

        for (iq, q) in enumerate(system.qvecs):
            for (j,idxpmq) in ipmq[iq]:
                Gpmq[s][iq] += G[s][idxpmq,j]

    essa = (1.0/(2.0*system.vol))*system.vqvec.dot(Gkpq[0]*Gpmq[0]-Gprod[0])
    essb = (1.0/(2.0*system.vol))*system.vqvec.dot(Gkpq[1]*Gpmq[1]-Gprod[1])
    eos = 0.5*((1.0/system.vol)*system.vqvec.dot(Gkpq[0]*Gpmq[1]) + (1.0/system.vol)*system.vqvec.dot(Gkpq[1]*Gpmq[0]))

    pe = essa + essb + eos

    G[0] = G[0].T
    G[1] = G[1].T

    return (ke+pe, ke, pe)

def unit_test():
    from pauxy.systems.ueg import UEG
    import numpy as np
    inputs = {'nup':7, 
    'ndown':7,
    'rs':1.0,
    'ecut':1.0}
    system = UEG(inputs, True)
    nbsf = system.nbasis
    Pa = np.zeros([nbsf,nbsf])
    Pb = np.zeros([nbsf,nbsf])
    na = system.nup
    nb = system.ndown
    for i in range(na):
        Pa[i,i] = 1.0
    for i in range(nb):
        Pb[i,i] = 1.0
    P = [Pa, Pb]
    # ((21.149879935489658+0j), 24.743544532817815, (-3.5936645973281554+0j))
    etot, ekin, epot = local_energy_ueg(system, P)

    print (etot, ekin, epot)

if __name__=="__main__":
    unit_test()
