import numpy
import scipy.optimize
import scipy.linalg
import math
import cmath
import time
import copy
import sys
import afqmcpy.utils
import afqmcpy.estimators
import afqmcpy.hubbard


class FreeElectron:

    def __init__(self, system, cplx, trial):
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

    def __init__(self, system, cplx, trial):
        print ("# Constructing trial wavefunction")
        init_time = time.time()
        if cplx:
            self.trial_type = complex
        else:
            self.trial_type = float
        # Unpack input options.
        self.ninitial = trial.get('ninitial', 100)
        self.nconv = trial.get('nconv', 5000)
        self.ueff = trial.get('ueff', 0.4)
        self.deps = trial.get('deps', 1e-8)
        self.alpha = trial.get('alpha', 0.5)
        (self.psi, self.eigs, self.emin) = self.find_uhf_wfn(system, cplx,
                                                             self.ueff,
                                                             self.ninitial,
                                                             self.nconv,
                                                             self.alpha,
                                                             self.deps)
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

class MultiDeterminant:

    def __init__(self, system, cplx, trial, expansion_type='read'):
        init_time = time.time()
        (self.eigs, self.eigv) = afqmcpy.utils.diagonalise_sorted(system.T)
        if cplx:
            self.trial_type = complex
        else:
            self.trial_type = float
        if expansion_type == 'free_electron':
            self.free_electron(system)
        self.initialisation_time = time.time() - init_time

    def free_electron(self, system):
        # assume nup = ndown
        deg_e = self.eigs[system.nup]
        # "core" orbitals, i.e., those filled states where there aren't any
        # unoccupied states of the same energy.
        core = [i for i, e in enumerate(self.eigs[:system.nup]) if abs(e-deg_e) > 1e-8]
        icore = core[-1]
        ecore = self.eigs[icore]
        # indices of degenerate orbitals
        nactive_elec = system.nup - len(core)
        active = [i for i, e in enumerate(eigs) if abs(e-deg_e) < 1e-8]
        combs = [list(c) for c in itertools.combinations(active, nactive_elec)]
        # Assuming nup = ndown, then there are len(combs)**2.0 combinations, we
        # need to take the sqrt of this for the normalisation of the trial
        # function.
        coeff = 1.0 / len(combs)
        # We need to find all ways of occupying the up and down spins amongst
        # the degenerate orbitals. Note this can be done independently, hence
        # we use itertools.product rather than itertools.combinations.
        spin_comb = list(itertools.product(combs, combs))
        # This is stored as [ndets, nspin(=2), M, nsigma]
        psi_trial = numpy.zeros((len(spin_comb), 2, system.nbasis, system.nup),
                                dtype=trial_type)
        for (ic, cup) in enumerate(spin_comb):
            occ_orbs_up = core + cup[0]
            occ_orbs_down = core + cup[1]
            # select appropriate columns of the unitary matrix which
            # diagonalises H_1
            self.psi_trial[ic, 0, :, :] = eigv[:, occ_orbs_up]
            self.psi_trial[ic, 1, :, :] = eigv[:, occ_orbs_down]
            self.coeff[ic] = coeff

        self.combinations = spin_comb
