import scipy.linalg
import numpy
import math
import cmath
import scipy.optimize
import afqmcpy.hubbard
import copy
import afqmcpy.estimators


def diag_sorted(H):

    (eigs, eigv) = scipy.linalg.eigh(H)
    idx = eigs.argsort()
    eigs = eigs[idx]
    eigv = eigv[:,idx]

    return (eigs, eigv)


def free_electron(system, cplx):

    (eigs, eigv) = diag_sorted(system.T)
    if cplx:
        trial_type = complex
    else:
        trial_type = float
    psi_trial = numpy.array([eigv[:,:system.nup], eigv[:,:system.ndown]],
                            dtype=trial_type)

    return (psi_trial, eigs)


def uhf(system, cplx, ueff, nit_max=1000, alpha=0.5):

    emin = 0
    uold = system.U
    system.U = ueff
    for attempt in range(1, 1000):
        random = numpy.random.random((system.nbasis, system.nbasis))
        random = 0.5*(random + random.T)
        (e_up, ev_up) = diag_sorted(random)
        random = numpy.random.random((system.nbasis, system.nbasis))
        random = 0.5*(random + random.T)
        (e_down, ev_down) = diag_sorted(system.T)
        if cplx:
            trial_type = complex
        else:
            trial_type = float

        psi_trial = numpy.array([ev_up[:,:system.nup], ev_down[:,:system.ndown]],
                                dtype=trial_type)
        niup = numpy.diag(psi_trial[0].dot((psi_trial[0].conj()).T))
        nidown = numpy.diag(psi_trial[1].dot((psi_trial[1].conj()).T))
        HMFU = system.T + numpy.diag(ueff*nidown) - 0.5*ueff*sum(niup*nidown)*numpy.identity(len(system.T))
        HMFD = system.T + numpy.diag(ueff*niup) - 0.5*ueff*sum(niup*nidown)*numpy.identity(len(system.T))
        (e_up, ev_up) = diag_sorted(HMFU)
        (e_down, ev_down) = diag_sorted(HMFD)
        psi_trial = numpy.array([ev_up[:,:system.nup], ev_down[:,:system.ndown]],
                                 dtype=trial_type)
        eold = sum(e_up[:system.nup]+e_down[:system.ndown])
        # print numpy.diag(psi_trial[0].dot((psi_trial[0].conj()).T))
        # print (1-alpha)*numpy.diag(psi_trial[1].dot((psi_trial[1].conj()).T)) + alpha*nidown
        niup = alpha*numpy.diag(psi_trial[0].dot((psi_trial[0].conj()).T)) + (1-alpha)*niup
        nidown = alpha*numpy.diag(psi_trial[1].dot((psi_trial[1].conj()).T)) + (1-alpha)*nidown
        for it in range(0, nit_max):
            HMFU = system.T + numpy.diag(ueff*nidown) - 0.5*ueff*sum(niup*nidown)*numpy.identity(len(system.T))
            HMFD = system.T + numpy.diag(ueff*niup) - 0.5*ueff*sum(niup*nidown)*numpy.identity(len(system.T))
            (e_up, ev_up) = diag_sorted(HMFU)
            (e_down, ev_down) = diag_sorted(HMFD)
            psi_trial = numpy.array([ev_up[:,:system.nup], ev_down[:,:system.ndown]],
                                     dtype=trial_type)
            enew = sum(e_up[:system.nup]+e_down[:system.ndown])
            Gup = psi_trial[0].dot(scipy.linalg.inv(psi_trial[0].conj().T.dot(psi_trial[0]))).dot(psi_trial[0].conj().T)
            Gdown = psi_trial[1].dot(scipy.linalg.inv(psi_trial[1].conj().T.dot(psi_trial[1])).dot(psi_trial[1].conj().T))
            enew = afqmcpy.estimators.local_energy(system, [Gup,Gdown])[0].real
            if abs(enew-eold) < 1e-8:
                # print enew
                if enew < emin:
                    emin = enew
                    psi_accept = copy.deepcopy(psi_trial)
                    # Gup = psi_accept[0].dot(scipy.linalg.inv(psi_accept[0].conj().T.dot(psi_accept[0]))).dot(psi_accept[0].conj().T)
                    # Gdown = psi_accept[1].dot(scipy.linalg.inv(psi_accept[1].conj().T.dot(psi_accept[1])).dot(psi_accept[1].conj().T))
                    print emin, afqmcpy.estimators.local_energy(system, [Gup,Gdown])[0].real, sum(e_up[:system.nup]+e_down[:system.ndown])
                break
            else:
                niup = alpha*numpy.diag(psi_trial[0].dot((psi_trial[0].conj()).T)) + (1-alpha)*niup
                nidown = alpha*numpy.diag(psi_trial[1].dot((psi_trial[1].conj()).T)) + (1-alpha)*nidown
                eold = enew
                # print it, enew

    system.U = uold

    return (psi_accept, e_up)
