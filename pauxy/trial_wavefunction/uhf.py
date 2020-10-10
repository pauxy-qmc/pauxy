import copy
import time

from scipy.linalg import expm
from scipy.optimize import minimize

from pauxy.estimators.mixed import local_energy
from pauxy.utils.linalg import diagonalise_sorted
from pauxy.systems.hubbard import decode_basis
from pauxy.utils.io import get_input_value
from pauxy.utils.linalg import reortho

try:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax
    from jax import grad, jit
    import jax.numpy as np
    import jax.scipy.linalg as LA
    import numpy
except ModuleNotFoundError:
    import numpy
    np = numpy
    def jit(function):
       def wrapper():
           function
       return wrapper()


import math

def projected_virtuals(psia):

    nup = psia.shape[1]
    nbasis = psia.shape[0]
    Ca = numpy.zeros((nbasis, nbasis), dtype = numpy.complex128)
    Ca[:,:nup] = psia.copy()

    # projected virtual
    P = numpy.eye(nbasis) - psia.dot(psia.T)
    u, s, v = numpy.linalg.svd(P)
    count = 0
    for i, sv in enumerate(s):
        if (sv > 1e-10):
            Ca[:,nup+count] = u[:,i]
            count += 1
        if (nup + count == nbasis):
            break
    
    return Ca

@jit
def gab(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    # Todo: check energy evaluation at later point, i.e., if this needs to be
    # transposed. Shouldn't matter for Hubbard model.
    inv_O = np.linalg.inv((A.conj().T).dot(B))
    GAB = B.dot(inv_O.dot(A.conj().T))
    return GAB

@jit
def local_energy_hubbard_jax(T,U,G,Ghalf=None):
    r"""Calculate local energy of walker for the Hubbard-Hostein model.

    Parameters
    ----------
    system : :class:`HubbardHolstein`
        System information for the HubbardHolstein model.
    G : :class:`numpy.ndarray`
        Walker's "Green's function"

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    nbasis = T[0].shape[1]
    ke = np.sum(T[0] * G[0] + T[1] * G[1])

    pe = U * np.dot(G[0].diagonal(), G[1].diagonal())

    etot = ke + pe 

    return (etot, ke, pe)

def gradient(x, nbasis, nup, ndown, T, U, c0, restricted):
    grad = numpy.array(jax.grad(objective_function)(x, nbasis, nup, ndown, T, U, c0,restricted))
    return grad

@jit
def compute_exp(Ua, tmp, theta_a):
    for i in range(1,50):
        tmp = np.einsum("ij,jk->ik", theta_a, tmp)
        Ua += tmp / math.factorial(i)

    return Ua

def objective_function (x, nbasis, nup, ndown, T, U, c0, restricted):
    nbasis = int(round(nbasis))
    nup = int(round(nup))
    ndown = int(round(ndown))

    nbsf = nbasis
    nocca = nup
    noccb = ndown
    nvira = nbasis - nocca
    nvirb = nbasis - noccb
    
    nova = nocca*nvira
    novb = noccb*nvirb
    
    daia = np.array(x[:nova],dtype=np.float64)
    daib = np.array(x[nova:nova+novb],dtype=np.float64)

    daia = daia.reshape((nvira, nocca))
    daib = daib.reshape((nvirb, noccb))

    if (restricted):
        daib = jax.ops.index_update(daib, jax.ops.index[:,:], daia)

    theta_a = np.zeros((nbsf, nbsf),dtype=np.float64)
    theta_b = np.zeros((nbsf, nbsf),dtype=np.float64)

    theta_a = jax.ops.index_update(theta_a, jax.ops.index[nocca:nbsf,:nocca], daia)
    theta_a = jax.ops.index_update(theta_a, jax.ops.index[:nocca, nocca:nbsf], -np.transpose(daia))

    theta_b = jax.ops.index_update(theta_b, jax.ops.index[noccb:nbsf,:noccb], daib)
    theta_b = jax.ops.index_update(theta_b, jax.ops.index[:noccb, noccb:nbsf], -np.transpose(daib))

    Ua = np.eye(nbsf,dtype=np.float64)
    tmp = np.eye(nbsf,dtype=np.float64)
    Ua = compute_exp(Ua, tmp, theta_a)

    C0a = np.array(c0[:nbsf*nbsf].reshape((nbsf,nbsf)),dtype=np.float64)
    Ca = C0a.dot(Ua)
    Ga = gab(Ca[:,:nocca], Ca[:,:nocca])
    
    if (noccb > 0):
        C0b = np.array(c0[nbsf*nbsf:].reshape((nbsf,nbsf)),dtype=np.float64)
        Ub = np.eye(nbsf)
        tmp = np.eye(nbsf)
        Ub = compute_exp(Ub, tmp, theta_b)
        Cb = C0b.dot(Ub)
        Gb = gab(Cb[:,:noccb], Cb[:,:noccb])
    else:
        Gb = np.zeros_like(Ga)

    G = np.array([Ga, Gb],dtype=np.float64)

    etot,ke,pe = local_energy_hubbard_jax(T,U,G)

    return etot.real

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

    def __init__(self, system, trial={}, verbose=0):
        assert "Hubbard" in system.name
        if verbose:
            print("# Constructing UHF trial wavefunction")
        self.verbose = verbose
        init_time = time.time()
        self.name = "UHF"
        self.type = "UHF"
        self.initial_wavefunction = trial.get('initial_wavefunction',
                                              'trial')
        self.trial_type = complex
        # Unpack input options.
        self.ninitial = get_input_value(trial, 'ninitial', default=10,
                                        verbose=verbose)
        self.nconv = get_input_value(trial, 'nconv', default=5000,
                                    verbose=verbose)
        self.ueff = get_input_value(trial, 'ueff',
                                    default=0.4,
                                    verbose=verbose)
        self.deps = get_input_value(trial, 'deps', default=1e-8,
                                    verbose=verbose)
        self.alpha = get_input_value(trial, 'alpha', default=0.5,
                                     verbose=verbose)

        self.variational = get_input_value(trial, 'variational', default=True,
                                     verbose=verbose)
        self.restricted = get_input_value(trial, 'restricted', default=False,
                                     verbose=verbose)
        self.maxiter = get_input_value(trial, 'maxiter',default= 3, verbose=verbose)
        self.maxscf = get_input_value(trial, 'maxscf', default=500, verbose=verbose)
        # For interface compatability
        self.coeffs = 1.0
        self.type = 'UHF'
        self.ndets = 1
        self.initial_guess = trial.get('initial', 'kinetic')
        if self.verbose:
            print("# initial guess is {}".format(self.initial_guess))
        if self.initial_guess == 'random':
            # if self.verbose:
                # print("# Solving UHF equations.")
            # (self.psi, self.eigs, self.emin, self.error, self.nav) = (
            #     self.find_uhf_wfn(system, self.ueff, self.ninitial,
            #                       self.nconv, self.alpha, self.deps, verbose)
            # )
            # print("self.emin = {}".format(self.emin))
            # if self.error:
                # warnings.warn('Error in constructing trial wavefunction. Exiting')
                # sys.exit()
            (self.psi, eold) = self.initialise(system.nbasis, system.nup,
                                            system.ndown)
        elif self.initial_guess == 'kinetic':
            Ta = system.T[0]
            Tb = system.T[1]
            ea, va = numpy.linalg.eigh(Ta)
            eb, vb = numpy.linalg.eigh(Tb)
            (self.psi, eold) = self.initialise(system.nbasis, system.nup,
                                            system.ndown)
            self.psi[:,:system.nup] = va[:,:system.nup]
            self.psi[:,system.nup:] = vb[:,:system.ndown]

        elif self.initial_guess == 'checkerboard':
            if self.verbose:
                print("# Using checkerboard breakup.")
            self.psi, unused = self.checkerboard(system.nbasis, system.nup, system.ndown)

        self.virt = numpy.zeros((system.nbasis, 2*system.nbasis - system.nup - system.ndown), dtype = self.psi.dtype)
        psia = projected_virtuals(self.psi[:,:system.nup])
        psib = projected_virtuals(self.psi[:,system.nup:])
        self.virt[:,:system.nbasis-system.nup] = psia[:,system.nup:]
        self.virt[:,system.nbasis-system.nup:] = psib[:,system.ndown:]
        
        if (self.variational):
            if (verbose):
                print("# we will repeat SCF {} times".format(self.maxiter))
            self.run_variational(system, verbose)

        Gup = gab(self.psi[:,:system.nup], self.psi[:,:system.nup]).T
        if (system.ndown > 0):
            Gdown = gab(self.psi[:,system.nup:], self.psi[:,system.nup:]).T
        else:
            Gdown = numpy.zeros_like(Gup)
        self.le_oratio = 1.0
        self.G = numpy.array([Gup, Gdown])
        self.etrial = local_energy(system, self.G)[0].real
        
        if self.verbose:
            print("# Trial energy is {}".format(self.etrial))

        self.bp_wfn = trial.get('bp_wfn', None)
        self.initialisation_time = time.time() - init_time
        self.init = self.psi
        self._mem_required = 0.0
        self._rchol = None

    def find_uhf_wfn(self, system, ueff, ninit,
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
                                            system.ndown)
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
                if (system.ndown>0):
                    Gdown = gab(self.trial[:,nup:], self.trial[:,nup:]).T
                else:
                    Gdown = numpy.zeros((system.nbasis, system.nbasis))
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

        system.U = uold
        if verbose:
            print("# Minimum energy found: {: 8f}".format(min(minima)))
            nocca = system.nup
            noccb = system.ndown
            MS = numpy.abs(nocca-noccb) / 2.0
            S2exact = MS * (MS+1.)
            Sij = psi_accept[:,:nocca].T.dot(psi_accept[:,nocca:])
            S2 = S2exact + min(nocca, noccb) - numpy.sum(numpy.abs(Sij*Sij).ravel())
            print("# <S^2> = {: 3f}".format(S2))
        try:
            return (psi_accept, e_accept, min(minima), False, [niup, nidown])
        except UnboundLocalError:
            warnings.warn("Warning: No UHF wavefunction found."
                          "Delta E: %f" % (enew - emin))
            return (trial, numpy.append(e_up, e_down), None, True, None)

    def initialise(self, nbasis, nup, ndown):
        (e_up, ev_up) = self.random_starting_point(nbasis)
        (e_down, ev_down) = self.random_starting_point(nbasis)

        trial = numpy.zeros(shape=(nbasis, nup+ndown),
                            dtype=numpy.complex128)
        trial[:,:nup] = ev_up[:,:nup]
        trial[:,nup:] = ev_down[:,:ndown]
        eold = sum(e_up[:nup]) + sum(e_down[:ndown])

        return (trial, eold)

    def random_starting_point(self, nbasis):
        random = numpy.random.random((nbasis, nbasis))
        random = 0.5 * (random + random.T)
        (energies, eigv) = diagonalise_sorted(random)
        return (energies, eigv)

    def checkerboard(self, nbasis, nup, ndown):
        nalpha = 0
        nbeta = 0
        wfn = numpy.zeros(shape=(nbasis, nup+ndown),
                            dtype=numpy.complex128)
        for i in range(nbasis):
            x, y = decode_basis(4,4,i)
            if x % 2 == 0 and y % 2 == 0:
                wfn[i,nalpha] = 1.0
                nalpha += 1
            elif x % 2 == 0 and y % 2 == 1:
                wfn[i,nup+nbeta] = -1.0
                nbeta += 1
            elif x % 2 == 1 and y % 2 == 0:
                wfn[i,nup+nbeta] = -1.0
                nbeta += 1
            elif x % 2 == 1 and y % 2 == 1:
                wfn[i,nalpha] = 1.0
                nalpha += 1
        return wfn, 10


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


    def run_variational(self, system, verbose):
        nbsf = system.nbasis
        nocca = system.nup
        noccb = system.ndown
        nvira = system.nbasis - nocca
        nvirb = system.nbasis - noccb   
#         
        nova = nocca*nvira
        novb = noccb*nvirb
#         
        x = numpy.zeros(nova + novb, dtype=numpy.float64)

        Ca = numpy.zeros((nbsf,nbsf))
        Ca[:,:nocca] = numpy.real(self.psi[:,:nocca])
        Ca[:,nocca:] = numpy.real(self.virt[:,:nvira])
        Cb = numpy.zeros((nbsf,nbsf))
        Cb[:,:noccb] = numpy.real(self.psi[:,nocca:])
        Cb[:,noccb:] = numpy.real(self.virt[:,nvira:])
        
        if (self.restricted):
            Cb = Ca.copy()

        if (system.ndown > 0):
            c0 = numpy.zeros(nbsf*nbsf*2, dtype=numpy.float64)
            c0[:nbsf*nbsf] = Ca.ravel()
            c0[nbsf*nbsf:] = Cb.ravel()
        else:
            c0 = numpy.zeros(nbsf*nbsf, dtype=numpy.float64)
            c0[:nbsf*nbsf] = Ca.ravel()
#       

        self.energy = 1e6

        for i in range (self.maxiter): # Try 10 times
            res = minimize(objective_function, x, args=(float(system.nbasis), float(system.nup), float(system.ndown),\
                    system.T, self.ueff, c0, self.restricted), jac=gradient, tol=1e-10,\
                method='L-BFGS-B',\
                options={ 'maxls': 20, 'iprint': 2, 'gtol': 1e-10, 'eps': 1e-10, 'maxiter': self.maxscf,\
                'ftol': 1.0e-10, 'maxcor': 1000, 'maxfun': 15000,'disp':True})
            e = res.fun
            if (verbose):
                print("# macro iter {} energy is {}".format(i, e))
            if (e < self.energy and numpy.abs(self.energy - e) > 1e-6):
                self.energy = res.fun
                xconv = res.x.copy()
            else:
                break
            x[0:0+nova+novb] = numpy.random.randn(nova+novb) * 1e-1 + xconv[0:]

        daia = res.x[0:0+nova] 
        daib = res.x[0+nova:0+nova+novb]

        daia = daia.reshape((nvira, nocca))
        daib = daib.reshape((nvirb, noccb))

        if (self.restricted):
            daib = daia.copy()

        theta_a = numpy.zeros((nbsf, nbsf))
        theta_a[nocca:nbsf,:nocca] = daia.copy()
        theta_a[:nocca, nocca:nbsf] = -daia.T.copy()

        theta_b = numpy.zeros((nbsf, nbsf))
        theta_b[noccb:nbsf,:noccb] = daib.copy()
        theta_b[:noccb, noccb:nbsf] = -daib.T.copy()
        
        Ua = expm(theta_a)
        C0a = c0[:nbsf*nbsf].reshape((nbsf,nbsf))
        Ca = C0a.dot(Ua)

        if (noccb > 0):
            C0b = c0[nbsf*nbsf:].reshape((nbsf,nbsf))
            Ub = expm(theta_b)
            Cb = C0b.dot(Ub)

        Cocca, detpsi = reortho(Ca[:,:nocca])
        Coccb, detpsi = reortho(Cb[:,:noccb])

        self.psi[:,:nocca] = Cocca
        self.psi[:,nocca:] = Coccb

        self.update_electronic_greens_function(system)


        MS = numpy.abs(nocca-noccb) / 2.0
        S2exact = MS * (MS+1.)
        Sij = self.psi[:,:nocca].T.dot(self.psi[:,nocca:])
        S2 = S2exact + min(nocca, noccb) - numpy.sum(numpy.abs(Sij*Sij).ravel())


        # nocca = system.nup
        # noccb = system.ndown
        # MS = numpy.abs(nocca-noccb) / 2.0
        # S2exact = MS * (MS+1.)
        # Sij = psi_accept[:,:nocca].T.dot(psi_accept[:,nocca:])
        # S2 = S2exact + min(nocca, noccb) - numpy.sum(numpy.abs(Sij*Sij).ravel())

        print("# <S^2> = {: 3f}".format(S2))

    def update_electronic_greens_function(self, system, verbose=0):
        gup = gab(self.psi[:, :system.nup],
                                         self.psi[:, :system.nup]).T
        if (system.ndown == 0):
            gdown = numpy.zeros_like(gup)
        else:
            gdown = gab(self.psi[:, system.nup:],
                                           self.psi[:, system.nup:]).T
        self.G = numpy.array([gup, gdown])