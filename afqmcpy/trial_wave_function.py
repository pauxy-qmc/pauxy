import scipy.linalg
import numpy
import math
import cmath
import scipy.optimize
import afqmcpy.hubbard
import copy
import afqmcpy.estimators
import time
import afqmcpy.utils


class Free_Electron:

    def __init__(self, system, cplx):
        init_time = time.time()
        (self.eigs, self.eigv) = afqmcpy.utils.diagonalise_sorted(system.T)
        if cplx:
            self.trial_type = complex
        else:
            self.trial_type = float
        self.psi = numpy.array([self.eigv[:,:system.nup], self.eigv[:,:system.ndown]],
                                dtype=self.trial_type)
        self.initialisation_time = time.time() - init_time


class UHF:

    def __init__(self, system, cplx, ueff, ninit=100, nit_max=1000, alpha=0.5):
        init_time = time.time()
        if cplx:
            self.trial_type = complex
        else:
            self.trial_type = float
        (self.psi, self.eigs) = self.find_uhf_wfn(system, cplx, ueff, ninit, nit_max, alpha)
        self.initialisation_time = time.time() - init_time

    def find_uhf_wfn(self, system, cplx, ueff, ninit=100, nit_max=1000, alpha=0.5):
        emin = 0
        uold = system.U
        system.U = ueff
        for attempt in range(0, ninit):
            random = numpy.random.random((system.nbasis, system.nbasis))
            random = 0.5*(random + random.T)
            (e_up, ev_up) = afqmcpy.utils.diagonalise_sorted(random)
            random = numpy.random.random((system.nbasis, system.nbasis))
            random = 0.5*(random + random.T)
            (e_down, ev_down) = afqmcpy.utils.diagonalise_sorted(random)
            if cplx:
                trial_type = complex
            else:
                trial_type = float

            trial = numpy.array([ev_up[:,:system.nup], ev_down[:,:system.ndown]],
                                    dtype=trial_type)
            niup = numpy.diag(trial[0].dot((trial[0].conj()).T))
            nidown = numpy.diag(trial[1].dot((trial[1].conj()).T))
            niup_old = numpy.diag(trial[0].dot((trial[0].conj()).T))
            nidown_old = numpy.diag(trial[1].dot((trial[1].conj()).T))
            eold = sum(e_up[:system.nup]+e_down[:system.ndown])
            for it in range(0, nit_max):
                HMFU = system.T + numpy.diag(ueff*nidown)
                HMFD = system.T + numpy.diag(ueff*niup)
                (e_up, ev_up) = afqmcpy.utils.diagonalise_sorted(HMFU)
                (e_down, ev_down) = afqmcpy.utils.diagonalise_sorted(HMFD)
                trial = numpy.array([ev_up[:,:system.nup], ev_down[:,:system.ndown]],
                                         dtype=trial_type)
                Gup = afqmcpy.estimators.gab(trial[0], trial[0])
                Gdown = afqmcpy.estimators.gab(trial[1], trial[1])
                enew = afqmcpy.estimators.local_energy(system, [Gup,Gdown])[0].real
                niup = alpha*numpy.diag(trial[0].dot((trial[0].conj()).T)) + (1-alpha)*niup_old
                nidown = alpha*numpy.diag(trial[1].dot((trial[1].conj()).T)) + (1-alpha)*nidown_old
                if self.self_consistant(enew, eold, niup, niup_old, nidown, nidown_old):
                    if enew-emin < -1e-8:
                        emin = enew
                        psi_accept = copy.deepcopy(trial)
                        e_accept = numpy.append(e_up, e_down)
                    break
                else:
                    eold = enew
                    niup_old = niup
                    nidown_old = nidown

        system.U = uold
        return (psi_accept, e_accept)

    def self_consistant(self, enew, eold, niup, niup_old, nidown, nidown_old):
        '''Check if system parameters are converged'''

        e_cond= abs(enew-eold) < 1e-8
        nup_cond = sum(abs(niup-niup_old)) < 1e-8
        ndown_cond = sum(abs(nidown-nidown_old)) < 1e-8

        return e_cond and nup_cond and ndown_cond
