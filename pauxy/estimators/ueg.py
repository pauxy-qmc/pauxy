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

    for s in [0, 1]:
        for (iq, q) in enumerate(system.qvecs):
            for i, k in enumerate(system.basis):
                kpq = k + q
                ikpq = system.lookup_basis(kpq)
                if ikpq is not None:
                    Gkpq[s][iq] += G[s][ikpq,i]
                    # if (abs(Gkpq[i]) > 1e-16):
                        # print ("NZ: ", i, k, q, kpq, ikpq)
                    for (j, p) in enumerate(system.basis):
                        pmq = p - q
                        # if (kpq.dot(kpq) == 0): print ("kpq: ", ikpq, i, k, j, p, q, pmq)
                        ipmq = system.lookup_basis(pmq)
                        if ipmq is not None:
                            # if (numpy.all(k==[0,0,1])):
                                # print (i, k, iq, kpq, kpq.dot(kpq), ikpq,
                                        # system.lookup_basis([0,0,0]),
                                        # system.map_basis_to_index([0,0,0]),
                                        # G[0][ikpq,j], G[0][ipmq,i])
                            Gprod[s][iq] += G[s][ikpq,j]*G[s][ipmq,i]
                pmq = k - q
                ipmq = system.lookup_basis(pmq)
                if ipmq is not None:
                    Gpmq[s][iq] += G[s][ipmq,i]

    essa = (1.0/(2.0*system.vol))*system.vqvec.dot(Gkpq[0]*Gpmq[0]-Gprod[0])
    essb = (1.0/(2.0*system.vol))*system.vqvec.dot(Gkpq[1]*Gpmq[1]-Gprod[1])
    eos = 0.5*((1.0/system.vol)*system.vqvec.dot(Gkpq[0]*Gpmq[1]) + (1.0/system.vol)*system.vqvec.dot(Gkpq[1]*Gpmq[0]))

    # print("essa = %10.5f, essb = %10.5f, eos = %10.5f"%(essa, essb, eos))

    pe = essa + essb + eos

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
    
    etot, ekin, epot = local_energy_ueg(system, P)

    print (etot, ekin, epot)

if __name__=="__main__":
    unit_test()
