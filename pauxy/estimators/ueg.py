import time
import numpy
try:
    from mpi4py import MPI
    mpi_sum = MPI.SUM
except ImportError:
    mpi_sum = None
import scipy.linalg

from pauxy.estimators.ueg_kernels  import  exchange_greens_function_per_qvec

def exchange_greens_function(nq, kpq_i, kpq, pmq_i, pmq, Gprod, G):
    for iq in range(nq):
        for (idxkpq,i) in zip(kpq[iq],kpq_i[iq]):
            for (idxpmq,j) in zip(pmq[iq],pmq_i[iq]):
                Gprod[iq] += G[j,idxkpq]*G[i,idxpmq]

def coulomb_greens_function(nq, kpq_i, kpq, pmq_i, pmq, Gkpq, Gpmq, G):
    for iq in range(nq):
        for (idxkpq,i) in zip(kpq[iq],kpq_i[iq]):
            Gkpq[iq] += G[i,idxkpq]
        for (idxpmq,i) in zip(pmq[iq],pmq_i[iq]):
            Gpmq[iq] += G[i,idxpmq]

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
    nq = numpy.shape(system.qvecs)[0]

    for s in [0, 1]:
        # exchange_greens_function(nq, system.ikpq_i, system.ikpq_kpq, system.ipmq_i,system.ipmq_pmq, Gprod[s],G[s])
        coulomb_greens_function(nq, system.ikpq_i, system.ikpq_kpq,  system.ipmq_i,system.ipmq_pmq, Gkpq[s],Gpmq[s],G[s])
        for iq in range(nq):
            Gprod[s,iq] = exchange_greens_function_per_qvec(system.ikpq_i[iq], system.ikpq_kpq[iq],
                                                            system.ipmq_i[iq],system.ipmq_pmq[iq],
                                                            G[s])

    tmp = numpy.multiply(Gkpq[0],Gpmq[0]) - Gprod[0]
    essa = (1.0/(2.0*system.vol))*system.vqvec.dot(tmp)
    
    tmp = numpy.multiply(Gkpq[1],Gpmq[1]) - Gprod[1]
    essb = (1.0/(2.0*system.vol))*system.vqvec.dot(tmp)

    eos = (1.0/(2.0*system.vol))*system.vqvec.dot(numpy.multiply(Gkpq[0],Gpmq[1])) \
        + (1.0/(2.0*system.vol))*system.vqvec.dot(numpy.multiply(Gkpq[1],Gpmq[0]))

    pe = essa + essb + eos

    return (ke+pe, ke, pe)

def unit_test():
    from pauxy.systems.ueg import UEG
    import numpy as np
    inputs = {'nup':2, 
    'ndown':2,
    'rs':1.0,
    'ecut':0.5}
    system = UEG(inputs, True)
    nbsf = system.nbasis
    Pa = np.zeros([nbsf,nbsf],dtype = np.complex128)
    Pb = np.zeros([nbsf,nbsf],dtype = np.complex128)
    na = system.nup
    nb = system.ndown
    for i in range(na):
        Pa[i,i] = 1.0
    for i in range(nb):
        Pb[i,i] = 1.0
    P = np.array([Pa, Pb])
    etot, ekin, epot = local_energy_ueg(system, P)
    print (etot, ekin, epot)
# ((13.603557335564197+0j), 15.692780148560848, (-2.0892228129966508+0j))

if __name__=="__main__":
    unit_test()
