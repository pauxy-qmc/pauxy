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


def self_consistant(enew, eold, niup, niup_old, nidown, nidown_old):
    e_cond= enew - eold < 1e-8
    nup_cond = sum(abs(niup-niup_old)) < 1e-8
    ndown_cond = sum(abs(nidown-nidown_old)) < 1e-8
    return e_cond and nup_cond and ndown_cond


def uhf(system, cplx, ueff, ninit=100, nit_max=1000, alpha=0.5):

    emin = 0
    uold = system.U
    system.U = ueff
    for attempt in range(0, ninit):
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
        niup_old = numpy.diag(psi_trial[0].dot((psi_trial[0].conj()).T))
        nidown_old = numpy.diag(psi_trial[1].dot((psi_trial[1].conj()).T))
        eold = sum(e_up[:system.nup]+e_down[:system.ndown])
        for it in range(0, nit_max):
            HMFU = system.T + numpy.diag(ueff*nidown)
            HMFD = system.T + numpy.diag(ueff*niup)
            (e_up, ev_up) = diag_sorted(HMFU)
            (e_down, ev_down) = diag_sorted(HMFD)
            psi_trial = numpy.array([ev_up[:,:system.nup], ev_down[:,:system.ndown]],
                                     dtype=trial_type)
            Gup = afqmcpy.estimators.gab(psi_trial[0], psi_trial[0])
            Gdown = afqmcpy.estimators.gab(psi_trial[1], psi_trial[1])
            enew = afqmcpy.estimators.local_energy(system, [Gup,Gdown])[0].real
            niup = alpha*numpy.diag(psi_trial[0].dot((psi_trial[0].conj()).T)) + (1-alpha)*niup_old
            nidown = alpha*numpy.diag(psi_trial[1].dot((psi_trial[1].conj()).T)) + (1-alpha)*nidown_old
            if self_consistant(enew, eold, niup, niup_old, nidown, nidown_old):
                if enew < emin:
                    emin = enew
                    psi_accept = copy.deepcopy(psi_trial)
                break
            else:
                eold = enew
                niup_old = niup
                nidown_old = nidown

    system.U = uold

    return (psi_accept, e_up)
