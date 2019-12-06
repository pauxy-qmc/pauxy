import copy
import numpy
import time
from pauxy.estimators.mixed import local_energy
from pauxy.estimators.greens_function import gab
from pauxy.utils.linalg import diagonalise_sorted

class UHF(object):
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
    system : :class:`pauxy.systems.hubbard.Hubbard` object
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

    def __init__(self, system, cplx, trial, parallel=False, verbose=0):
        if verbose:
            print("# Constructing UHF trial wavefunction")
        self.verbose = verbose
        init_time = time.time()
        self.name = "UHF"
        self.type = "UHF"
        self.initial_wavefunction = trial.get('initial_wavefunction',
                                              'trial')
        if cplx:
            self.trial_type = complex
        else:
            self.trial_type = float
        # Unpack input options.
        self.ninitial = trial.get('ninitial', 10)
        self.nconv = trial.get('nconv', 5000)
        self.ueff = trial.get('ueff', 0.4)
        self.deps = trial.get('deps', 1e-8)
        self.alpha = trial.get('alpha', 0.5)
        # For interface compatability
        self.coeffs = 1.0
        self.type = 'UHF'
        self.ndets = 1
        (self.psi, self.eigs, self.emin, self.error, self.nav) = (
            self.find_uhf_wfn(system, cplx, self.ueff, self.ninitial,
                              self.nconv, self.alpha, self.deps, verbose)
        )
        if self.error and not parallel:
            warnings.warn('Error in constructing trial wavefunction. Exiting')
            sys.exit()
        Gup = gab(self.psi[:,:system.nup], self.psi[:,:system.nup]).T
        Gdown = gab(self.psi[:,system.nup:], self.psi[:,system.nup:]).T
        self.G = numpy.array([Gup, Gdown])
        self.etrial = local_energy(system, self.G)[0].real
        self.bp_wfn = trial.get('bp_wfn', None)
        self.initialisation_time = time.time() - init_time
        self.init = self.psi

    def find_uhf_wfn(self, system, cplx, ueff, ninit,
                     nit_max, alpha, deps=1e-8, verbose=0):
        emin = 0
        uold = system.U
        system.U = ueff
        minima = []  # Local minima
        nup = system.nup
        # Search over different random starting points.
        for attempt in range(0, ninit):
            # Set up initial (random) guess for the density.
            (self.trial, eold) = self.initialise(system.nbasis, system.nup,
                                            system.ndown, cplx)
            niup = self.density(self.trial[:,:nup])
            nidown = self.density(self.trial[:,nup:])
            niup_old = self.density(self.trial[:,:nup])
            nidown_old = self.density(self.trial[:,nup:])
            for it in range(0, nit_max):
                (niup, nidown, e_up, e_down) = (
                    self.diagonalise_mean_field(system, ueff, niup, nidown)
                )
                # Construct Green's function to compute the energy.
                Gup = gab(self.trial[:,:nup], self.trial[:,:nup]).T
                Gdown = gab(self.trial[:,nup:], self.trial[:,nup:]).T
                enew = local_energy(system, numpy.array([Gup, Gdown]))[0].real
                if verbose > 1:
                    print("# %d %f %f" % (it, enew, eold))
                sc = self.self_consistent(enew, eold, niup, niup_old, nidown,
                                          nidown_old, it, deps, verbose)
                if sc:
                    # Global minimum search.
                    if attempt == 0:
                        minima.append(enew)
                        psi_accept = copy.deepcopy(self.trial)
                        e_accept = numpy.append(e_up, e_down)
                    elif all(numpy.array(minima) - enew > deps):
                        minima.append(enew)
                        psi_accept = copy.deepcopy(self.trial)
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
            if verbose > 1:
                print("# SCF cycle: {:3d}. After {:4d} steps the minimum UHF"
                      " energy found is: {: 8f}".format(attempt, it, eold))
                MS = numpy.abs(system.nup - system.ndown) / 2.0
                S2exact = MS * (MS+1.)
                Sij = self.trial[:,:nup].T.dot(self.trial[:,nup:])
                S2 = S2exact + min(system.nup, system.ndown) - numpy.sum(numpy.abs(Sij).ravel())
                print("# <S^2> = {: 3f}".format(S2))

        system.U = uold
        print("# Minimum energy found: {: 8f}".format(min(minima)))
        MS = numpy.abs(system.nup - system.ndown) / 2.0
        S2exact = MS * (MS+1.)
        Sij = self.trial[:,:nup].T.dot(self.trial[:,nup:])
        S2 = S2exact + min(system.nup, system.ndown) - numpy.sum(numpy.abs(Sij).ravel())
        print("# <S^2> = {: 3f}".format(S2))

        try:
            return (psi_accept, e_accept, min(minima), False, [niup, nidown])
        except UnboundLocalError:
            warnings.warn("Warning: No UHF wavefunction found."
                          "Delta E: %f" % (enew - emin))
            return (trial, numpy.append(e_up, e_down), None, True, None)

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
        random = 0.5 * (random + random.T)
        (energies, eigv) = diagonalise_sorted(random)
        return (energies, eigv)

    def density(self, wfn):
        return numpy.diag(wfn.dot((wfn.conj()).T))

    def self_consistent(self, enew, eold, niup, niup_old, nidown, nidown_old,
                        it, deps=1e-8, verbose=0):
        '''Check if system parameters are converged'''

        depsn = deps**0.5
        ediff = abs(enew-eold)
        nup_diff = sum(abs(niup-niup_old))/len(niup)
        ndown_diff = sum(abs(nidown-nidown_old))/len(nidown)
        if verbose > 1:
            print("# de: %.10e dniu: %.10e dnid: %.10e"%(ediff, nup_diff, ndown_diff))

        return (ediff < deps) and (nup_diff < depsn) and (ndown_diff < depsn)

    def mix_density(self, new, old, alpha):
        return (1-alpha)*new + alpha*old

    def diagonalise_mean_field(self, system, ueff, niup, nidown):
        # mean field Hamiltonians.
        HMFU = system.T[0] + numpy.diag(ueff*nidown)
        HMFD = system.T[1] + numpy.diag(ueff*niup)

        # if (system.name == "HubbardHolstein"):
            # print("HHMODEL")

        (e_up, ev_up) = diagonalise_sorted(HMFU)
        (e_down, ev_down) = diagonalise_sorted(HMFD)
        # Construct new wavefunction given new density.
        self.trial[:,:system.nup] = ev_up[:,:system.nup]
        self.trial[:,system.nup:] = ev_down[:,:system.ndown]
        # Construct corresponding site densities.
        niup = self.density(self.trial[:,:system.nup])
        nidown = self.density(self.trial[:,system.nup:])
        return (niup, nidown, e_up, e_down)

    def calculate_energy(self, system):
        if self.verbose:
            print ("# Computing trial energy.")
        (self.energy, self.e1b, self.e2b) = local_energy(system, self.G)
        if self.verbose:
            print ("# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                   %(self.energy.real, self.e1b.real, self.e2b.real))


def unit_test():
    import itertools
    from pauxy.systems.hubbard import Hubbard
    from pauxy.estimators.ci import simple_fci_bose_fermi, simple_fci
    from pauxy.systems.hubbard_holstein import HubbardHolstein
    import scipy
    import numpy
    import scipy.sparse.linalg
    options1 = {
    "name": "Hubbard",
    # "name": "HubbardHolstein",
    "nup": 7,
    "ndown": 7,
    "nx": 7,
    "ny": 7,
    "U": 8.0,
    "w0": 0.5,
    "lambda": 1.0
    }
    options2 = {
    # "name": "Hubbard",
    "name": "HubbardHolstein",
    "nup": 7,
    "ndown": 7,
    "nx": 7,
    "ny": 7,
    "U": 8.0,
    "w0": 0.5,
    "lambda": 1.0
    }
    system = Hubbard (options1, verbose=True)
    system = HubbardHolstein (options2, verbose=True)

    uhf_driver = UHF(system, False, options1, parallel=False, verbose=0)
    uhf_driver = UHF(system, False, options2, parallel=False, verbose=0)



if __name__=="__main__":
    unit_test()
