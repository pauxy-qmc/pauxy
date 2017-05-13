import scipy.linalg
import numpy

def free_electron(system, cplx):

    (eigs, eigv) = scipy.linalg.eigh(system.T)
    idx = eigs.argsort()
    eigs = eigs[idx]
    eigv = eigv[:,idx]
    if cplx:
        trial_type = complex
    else:
        trial_type = float
    psi_trial = numpy.array([eigv[:,:system.nup], eigv[:,:system.ndown]],
                            dtype=trial_type)

    return (psi_trial, eigs)
