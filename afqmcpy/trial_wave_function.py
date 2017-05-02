import scipy.linalg
import numpy

def free_electron(system):

    (eigs, eigv) = scipy.linalg.eigh(system.T)
    idx = eigs.argsort()
    eigs = eigs[idx]
    eigv = eigv[:,idx]
    psi_trial = numpy.array([eigv[:,:system.nup], eigv[:,:system.ndown]])

    return (psi_trial, eigs)
