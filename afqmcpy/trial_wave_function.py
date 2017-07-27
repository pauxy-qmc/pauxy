import numpy
import math
import cmath
import time
import copy
import sys
import scipy.optimize
import scipy.linalg
import afqmcpy.hubbard
import afqmcpy.estimators
import afqmcpy.utils


class Free_Electron:

    def __init__(self, system, cplx):
        init_time = time.time()
        (self.eigs, self.eigv) = afqmcpy.utils.diagonalise_sorted(system.T)
        if cplx:
            self.trial_type = complex
        else:
            self.trial_type = float
        # I think this is slightly cleaner than using two separate matrices.
        self.psi = numpy.zeros(shape=(system.nbasis, system.nup+system.ndown),
                               dtype=self.trial_type)
        self.psi[:,:system.nup] = self.eigv[:,:system.nup]
        self.psi[:,system.nup:] = self.eigv[:,:system.ndown]
        self.emin = sum(self.eigs[:system.nup]) + sum(self.eigs[:system.ndown])
        self.initialisation_time = time.time() - init_time


class UHF:

    def __init__(self, system, cplx, ueff, ninit=100, nit_max=5000, alpha=0.5):
        print ("# Constructing trial wavefunction")
        init_time = time.time()
        if cplx:
            self.trial_type = complex
        else:
            self.trial_type = float
        (self.psi, self.eigs, self.emin) = self.find_uhf_wfn(system, cplx, ueff, ninit, nit_max, alpha)
        self.initialisation_time = time.time() - init_time

    def find_uhf_wfn(self, system, cplx, ueff, ninit, nit_max, alpha,
                     deps=1e-8):
        emin = 0
        uold = system.U
        system.U = ueff
        minima= [0]
        nup = system.nup
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

            trial = numpy.zeros(shape=(system.nbasis, system.nup+system.ndown),
                                dtype=trial_type)
            trial[:,:system.nup] = ev_up[:,:system.nup]
            trial[:,system.nup:] = ev_down[:,:system.ndown]
            niup = numpy.diag(trial[:,:nup].dot((trial[:,:nup].conj()).T))
            nidown = numpy.diag(trial[:,nup:].dot((trial[:,nup:].conj()).T))
            niup_old = numpy.diag(trial[:,:nup].dot((trial[:,:nup].conj()).T))
            nidown_old = numpy.diag(trial[:,nup:].dot((trial[:,nup:].conj()).T))
            eold = sum(e_up[:system.nup]) + sum(e_down[:system.ndown])
            for it in range(0, nit_max):
                HMFU = system.T + numpy.diag(ueff*nidown)
                HMFD = system.T + numpy.diag(ueff*niup)
                (e_up, ev_up) = afqmcpy.utils.diagonalise_sorted(HMFU)
                (e_down, ev_down) = afqmcpy.utils.diagonalise_sorted(HMFD)
                trial = numpy.zeros(shape=(system.nbasis,
                                    system.nup+system.ndown), dtype=trial_type)
                trial[:,:system.nup] = ev_up[:,:system.nup]
                trial[:,system.nup:] = ev_down[:,:system.ndown]
                niup = numpy.diag(trial[:,:nup].dot((trial[:,:nup].conj()).T))
                nidown = numpy.diag(trial[:,nup:].dot((trial[:,nup:].conj()).T))
                Gup = afqmcpy.estimators.gab(trial[:,:nup], trial[:,:nup]).T
                Gdown = afqmcpy.estimators.gab(trial[:,nup:], trial[:,nup:]).T
                enew = afqmcpy.estimators.local_energy(system, [Gup,Gdown])[0].real
                if self.self_consistant(enew, eold, niup, niup_old, nidown,
                                        nidown_old, it, deps):
                    if all(abs(numpy.array(minima))-abs(enew) < -deps):
                        minima.append(enew)
                        psi_accept = copy.deepcopy(trial)
                        e_accept = numpy.append(e_up, e_down)
                    break
                else:
                    mixup = self.mix_density(niup, niup_old, alpha)
                    mixdown = self.mix_density(nidown, nidown_old, alpha)
                    niup_old = niup
                    nidown_old = nidown
                    niup = mixup
                    nidown = mixdown
                    eold = enew
            print ("# SCF cycle: {:3d}. After {:4d} steps the minimum UHF"
                    " energy found is: {: 8f}".format(attempt, it, eold))

        system.U = uold
        print ("# Minimum energy found: {: 8f}".format(min(minima)))
        try:
            return (psi_accept, e_accept, min(minima))
        except UnboundLocalError:
            print ("Warning: No UHF wavefunction found.")
            print ("%f"%(enew-emin))
            sys.exit()


    def self_consistant(self, enew, eold, niup, niup_old, nidown, nidown_old,
                        it, deps=1e-8):
        '''Check if system parameters are converged'''

        e_cond= abs(enew-eold) < deps
        nup_cond = sum(abs(niup-niup_old))/len(niup) < deps
        ndown_cond = sum(abs(nidown-nidown_old))/len(nidown) < deps

        return e_cond and nup_cond and ndown_cond

    def mix_density(self, new, old, alpha):
        return (1-alpha)*new + alpha*old
