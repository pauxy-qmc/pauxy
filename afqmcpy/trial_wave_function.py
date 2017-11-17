import numpy
import scipy.optimize
import scipy.linalg
import math
import cmath
import time
import copy
import sys
import ast
import afqmcpy.utils
import afqmcpy.estimators
import afqmcpy.hubbard


class FreeElectron:

    def __init__(self, system, cplx, trial):
        init_time = time.time()
        self.name = "free_electron"
        self.type = "free_electron"
        self.read_init = trial.get('inititial_wavefunction', None)
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
        G = afqmcpy.estimators.gab(self.psi[:,:system.nup],
                                   self.psi[:,:system.nup])
        self.emin = sum(self.eigs[:system.nup]) + sum(self.eigs[:system.ndown])
        # For interface compatability
        self.coeffs = 1.0
        self.bp_wfn = trial.get('bp_wfn', None)
        self.initialisation_time = time.time() - init_time


class UHF:
    r"""UHF trial wavefunction.

    Search for UHF trial wavefunction by self consistenly solving the mean field
    Hamiltonian:

        .. math::
            H^{\sigma} = \sum_{\langle ij\rangle} \left(
                    c^{\dagger}_{i\sigma}c_{j\sigma} + h.c.\right) +
                    U_{\mathrm{eff}} \sum_i \hat{n}_{i\sigma}\langle\hat{n}_{i\bar{\sigma}}\rangle -
                    \frac{1}{2} U_{\mathrm{eff}} \sum_i \langle\hat{n}_{i\sigma}\rangle
                    \langle\hat{n}_{i\bar{\sigma}}\rangle.

    See [Xu11]_ for more details.

    .. Warning::
        This is for the Hubbard model only

    .. todo:: We should generalise in the future perhaps.

    Parameters
    ----------
    system : :class:`afqmcpy.Hubbard` object
        System parameters.
    cplx : bool
        True if the trial wavefunction etc is complex.
    trial : dict
        Trial wavefunction input options.

    Attributes
    ----------
    psi : :class:`numpy.ndarray`
        Trial wavefunction.
    eigs : :class:`numpy.array`
        One-electron eigenvalues.
    emin : float
        Ground state mean field total energy of trial wavefunction.
    """

    def __init__(self, system, cplx, trial):
        print ("# Constructing trial wavefunction")
        init_time = time.time()
        self.name = "UHF"
        self.type = "UHF"
        self.read_init = trial.get('initial_wavefunction', None)
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
        # For interface compatability
        self.coeffs = 1.0
        (self.psi, self.eigs, self.emin) = self.find_uhf_wfn(system, cplx,
                                                             self.ueff,
                                                             self.ninitial,
                                                             self.nconv,
                                                             self.alpha,
                                                             self.deps)
        self.bp_wfn = trial.get('bp_wfn', None)
        self.initialisation_time = time.time() - init_time

    def find_uhf_wfn(self, system, cplx, ueff, ninit, nit_max, alpha, deps=1e-8):
        emin = 0
        uold = system.U
        system.U = ueff
        minima= [0] # Local minima
        nup = system.nup
        # Search over different random starting points.
        for attempt in range(0, ninit):
            # Set up initial (random) guess for the density.
            (trial, eold) = self.initialise(system.nbasis, system.nup,
                                            system.ndown, cplx)
            niup = self.density(trial[:,:nup])
            nidown = self.density(trial[:,nup:])
            niup_old = self.density(trial[:,:nup])
            nidown_old = self.density(trial[:,nup:])
            for it in range(0, nit_max):
                # mean field Hamiltonians.
                HMFU = system.T + numpy.diag(ueff*nidown)
                HMFD = system.T + numpy.diag(ueff*niup)
                (e_up, ev_up) = afqmcpy.utils.diagonalise_sorted(HMFU)
                (e_down, ev_down) = afqmcpy.utils.diagonalise_sorted(HMFD)
                # Construct new wavefunction given new density.
                trial[:,:system.nup] = ev_up[:,:system.nup]
                trial[:,system.nup:] = ev_down[:,:system.ndown]
                # Construct corresponding site densities.
                niup = self.density(trial[:,:nup])
                nidown = self.density(trial[:,nup:])
                # Construct Green's function to compute the energy.
                Gup = afqmcpy.estimators.gab(trial[:,:nup], trial[:,:nup]).T
                Gdown = afqmcpy.estimators.gab(trial[:,nup:], trial[:,nup:]).T
                enew = afqmcpy.estimators.local_energy(system, [Gup,Gdown])[0].real
                if self.self_consistant(enew, eold, niup, niup_old, nidown,
                                        nidown_old, it, deps):
                    # Global minimum search.
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


    def initialise(self, nbasis, nup, ndown, cplx):
        (e_up, ev_up) = self.random_starting_point(nbasis)
        (e_down, ev_down) = self.random_starting_point(nbasis)

        if cplx:
            trial_type = complex
        else:
            trial_type = float
        trial = numpy.zeros(shape=(nbasis, nup+ndown),
                            dtype=trial_type)
        trial[:,:nup] = ev_up[:,:nup]
        trial[:,nup:] = ev_down[:,:ndown]
        eold = sum(e_up[:nup]) + sum(e_down[:ndown])

        return (trial, eold)

    def random_starting_point(self, nbasis):
        random = numpy.random.random((nbasis, nbasis))
        random = 0.5*(random + random.T)
        (energies, eigv) = afqmcpy.utils.diagonalise_sorted(random)
        return (energies, eigv)

    def density(self, wfn):
        return numpy.diag(wfn.dot((wfn.conj()).T))

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

    def __init__(self, system, cplx, trial):
        init_time = time.time()
        self.name = "multi_determinant"
        self.expansion = "multi_determinant"
        self.type = trial.get('type')
        self.ndets = trial.get('ndets', None)
        self.eigs = numpy.array([0.0])
        self.read_init = trial.get('initial_wavefunction', None)
        self.bp_wfn = trial.get('bp_wfn', 'init')
        if cplx or self.type == 'GHF':
            self.trial_type = complex
        else:
            self.trial_type = float
        if self.type == 'UHF':
            nbasis = system.nbasis
        else:
            nbasis = 2 * system.nbasis
        self.GAB = numpy.zeros(shape=(self.ndets, self.ndets, nbasis, nbasis),
                               dtype=self.trial_type)
        self.weights = numpy.zeros(shape=(self.ndets, self.ndets),
                                   dtype=self.trial_type)
        # For debugging purposes.
        if self.type == 'free_electron':
            (self.eigs, self.eigv) = afqmcpy.utils.diagonalise_sorted(system.T)
            psi = numpy.zeros(shape=(self.ndets, system.nbasis, system.ne))
            psi[:,:system.nup] = self.eigv[:,:system.nup]
            psi[:,system.nup:] = self.eigv[:,:system.ndown]
            self.psi = numpy.array([copy.deepcopy(psi) for i in range(0,self.ndets)])
            self.emin = sum(self.eigs[:system.nup]) + sum(self.eigs[:system.ndown])
            self.coeffs = numpy.ones(self.ndets)
        else:
            self.orbital_file = trial.get('orbitals')
            self.coeffs_file = trial.get('coefficients')
            # Store the complex conjugate of the multi-determinant trial
            # wavefunction expansion coefficients for ease later.
            self.coeffs = read_fortran_complex_numbers(self.coeffs_file)
            self.psi = numpy.zeros(shape=(self.ndets, nbasis, system.ne),
                                   dtype=self.coeffs.dtype)
            orbitals = read_fortran_complex_numbers(self.orbital_file)
            start = 0
            skip = nbasis * system.ne
            end = skip
            for i in range(self.ndets):
                self.psi[i] = orbitals[start:end].reshape((nbasis, system.ne),
                                                          order='F')
                start = end
                end += skip
            afqmcpy.estimators.gab_multi_det_full(self.psi, self.psi,
                                                  self.coeffs, self.coeffs,
                                                  self.GAB, self.weights)
            self.emin = (
                afqmcpy.estimators.local_energy_ghf_full(system, self.GAB,
                                                         self.weights)[0].real
            )
        self.initialisation_time = time.time() - init_time

def read_fortran_complex_numbers(filename):
    with open (filename) as f:
        content = f.readlines()
    # Converting fortran complex numbers to python. ugh
    # Be verbose for clarity.
    useable = [c.strip() for c in content]
    tuples = [ast.literal_eval(u) for u in useable]
    orbs = [complex(t[0], t[1]) for t in tuples]
    return numpy.array(orbs)
