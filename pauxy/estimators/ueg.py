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

def local_energy_ueg(system, G, Ghalf=None, two_rdm=None):
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
            Gprod[s,iq] = exchange_greens_function_per_qvec(system.ikpq_i[iq],
                                                            system.ikpq_kpq[iq],
                                                            system.ipmq_i[iq],
                                                            system.ipmq_pmq[iq],
                                                            G[s])

    if two_rdm is None:
        two_rdm = numpy.zeros((2,2,len(system.qvecs)), dtype=numpy.complex128)
    two_rdm[0,0] = numpy.multiply(Gkpq[0],Gpmq[0]) - Gprod[0]
    essa = (1.0/(2.0*system.vol))*system.vqvec.dot(two_rdm[0,0])

    two_rdm[1,1] = numpy.multiply(Gkpq[1],Gpmq[1]) - Gprod[1]
    essb = (1.0/(2.0*system.vol))*system.vqvec.dot(two_rdm[1,1])

    two_rdm[0,1] = numpy.multiply(Gkpq[0],Gpmq[1])
    two_rdm[1,0] = numpy.multiply(Gkpq[1],Gpmq[0])
    eos = (
        (1.0/(2.0*system.vol))*system.vqvec.dot(two_rdm[0,1])
        + (1.0/(2.0*system.vol))*system.vqvec.dot(two_rdm[1,0])
    )

    pe = essa + essb + eos

    return (ke+pe, ke, pe)

def fock_ueg(system, G):
    """Fock matrix computation for uniform electron gas
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
    # ke = numpy.einsum('sij,sji->',system.H1,G)
    T = [system.H1[0], system.H1[1]] # kinetic energy integrals
    nbsf = system.nbasis
    nq = numpy.shape(system.qvecs)[0]

    Fock = [numpy.zeros((nbsf, nbsf), dtype = numpy.complex128), numpy.zeros((nbsf, nbsf), dtype = numpy.complex128)]
    J = [numpy.zeros((nbsf, nbsf), dtype = numpy.complex128), numpy.zeros((nbsf, nbsf), dtype = numpy.complex128)]
    K = [numpy.zeros((nbsf, nbsf), dtype = numpy.complex128), numpy.zeros((nbsf, nbsf), dtype = numpy.complex128)]


    Gkpq =  numpy.zeros((2,len(system.qvecs)), dtype=numpy.complex128)
    Gpmq =  numpy.zeros((2,len(system.qvecs)), dtype=numpy.complex128)

    for s in [0, 1]:
        coulomb_greens_function(nq, system.ikpq_i, system.ikpq_kpq,  system.ipmq_i,system.ipmq_pmq, Gkpq[s],Gpmq[s],G[s])


    for (iq, q) in enumerate(system.qvecs):
        for idxi, i in enumerate(system.basis[0:system.nbasis]):
            for idxj, j in enumerate(system.basis[0:system.nup]):
                jpq = j + q
                idxjpq = system.lookup_basis(jpq)
                if (idxjpq is not None) and (idxjpq == idxi):
                    J[0][idxj,idxi] += (1.0/(2.0*system.vol)) * system.vqvec[iq] * (Gpmq[0][iq] + Gpmq[1][iq])
    
    for (iq, q) in enumerate(system.qvecs):
        for idxi, i in enumerate(system.basis[0:system.nbasis]):
            for idxj, j in enumerate(system.basis[0:system.nup]):
                jpq = j - q
                idxjmq = system.lookup_basis(jpq)
                if (idxjmq is not None) and (idxjmq == idxi):
                    J[0][idxj,idxi] += (1.0/(2.0*system.vol)) * system.vqvec[iq] * (Gpmq[0][iq] + Gpmq[1][iq])

    J[1] = J[0]

    for s in [0, 1]:
        for iq in range(nq):
            for (idxjmq,idxj) in zip(system.ipmq_pmq[iq],system.ipmq_i[iq]):
                for (idxkpq,idxk) in zip(system.ikpq_kpq[iq],system.ikpq_i[iq]):
                    K[s][idxj, idxkpq] += - (1.0/(2.0*system.vol)) * system.vqvec[iq] * G[s][idxjmq, idxk]
        for iq in range(nq):
            for (idxjpq,idxj) in zip(system.ikpq_kpq[iq],system.ikpq_i[iq]):
                for (idxpmq,idxp) in zip(system.ipmq_pmq[iq],system.ipmq_i[iq]):
                    K[s][idxj, idxpmq] += - (1.0/(2.0*system.vol)) * system.vqvec[iq] * G[s][idxjpq, idxp]

    for s in [0, 1]:
        Fock[s] = T[s] + J[s] + 0.5*K[s]

    return Fock

def unit_test():
    from pauxy.systems.ueg import UEG
    import numpy as np
    inputs = {'nup':7,
    'ndown':7,
    'rs':5.0,
    'ecut':2.5}
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
    etot, ekin, epot = local_energy_ueg(system, G=P)
    print("initial = {}".format(etot, ekin, epot))

    from pauxy.utils.linalg import exponentiate_matrix, reortho
    from pauxy.estimators.greens_function import gab

    Ca = numpy.random.rand(nbsf, na)
    Cb = numpy.random.rand(nbsf, nb)
    Ca, detR = reortho(Ca)
    Cb, detR = reortho(Cb)

    dt = 0.5
    for i in range(10000):
        Fock = fock_ueg(system, G=P)
        expF = [exponentiate_matrix(-dt*Fock[0]), exponentiate_matrix(-dt*Fock[1])]
        Ca = expF[0].dot(Ca)
        Cb = expF[1].dot(Cb)
        Ca, detR = reortho(Ca)
        Cb, detR = reortho(Cb)
        P = [gab(Ca, Ca), gab(Cb, Cb)]
        print (local_energy_ueg(system, P))


if __name__=="__main__":
    unit_test()
