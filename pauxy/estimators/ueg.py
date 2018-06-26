import time
import numpy
try:
    from mpi4py import MPI
    mpi_sum = MPI.SUM
except ImportError:
    mpi_sum = None
import scipy.linalg

def local_energy_ueg(system, G):
    """Local energy computation for uniform electron gas
    Parameters
    ----------
    system :
        system class
    G : 
        Green's function
    Returns
    -------
    etot : float
        total energy
    ke : float
        kinetic energy
    pe : float
        potential energy
    """
    ke = numpy.einsum('sij,sji->',system.H1,G)

    Gkpq =  numpy.zeros((2,len(system.qvecs)), dtype=numpy.complex128)
    Gpmq =  numpy.zeros((2,len(system.qvecs)), dtype=numpy.complex128)
    Gprod = numpy.zeros((2,len(system.qvecs)), dtype=numpy.complex128)

    ne = [system.nup, system.ndown]

    ikpq = []
    for (iq, q) in enumerate(system.qvecs):
        idxkpq_list =[]
        # for i, k in enumerate(system.basis[0:ne[0]]):
        for i, k in enumerate(system.basis):
            kpq = k + q
            idxkpq = system.lookup_basis(kpq)
            if idxkpq is not None:
                idxkpq_list += [(idxkpq,i)]
        ikpq += [idxkpq_list]

    ipmq = []
    for (iq, q) in enumerate(system.qvecs):
        idxpmq_list =[]
        # for i, p in enumerate(system.basis[0:ne[0]]):
        for i, p in enumerate(system.basis):
            pmq = p - q
            idxpmq = system.lookup_basis(pmq)
            if idxpmq is not None:
                idxpmq_list += [(idxpmq,i)]
        ipmq += [idxpmq_list]

    ess = [0.0, 0.0]
    eos = 0.0

    for s in [0, 1]:
        for (iq, q) in enumerate(system.qvecs):
            for (idxkpq,i) in ikpq[iq]:
                Gkpq[s][iq] += G[s][i,idxkpq]
                for (idxpmq,j) in ipmq[iq]:
                    Gprod[s][iq] += G[s][j,idxkpq]*G[s][i,idxpmq]
        for (iq, q) in enumerate(system.qvecs):
            for (idxpmq,i) in ipmq[iq]:
                Gpmq[s][iq] += G[s][i,idxpmq]
        tmp = numpy.multiply(Gkpq[s],Gpmq[s]) - Gprod[s]
        ess[s] = (1.0/(2.0*system.vol))*system.vqvec.dot(tmp)

    eos = (1.0/(2.0*system.vol))*system.vqvec.dot(numpy.multiply(Gkpq[0],Gpmq[1])) \
        + (1.0/(2.0*system.vol))*system.vqvec.dot(numpy.multiply(Gkpq[1],Gpmq[0]))

    pe = ess[0] + ess[1] + eos

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
