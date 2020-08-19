import itertools
import cmath
import h5py
from pauxy.systems.hubbard import Hubbard
from pauxy.trial_wavefunction.free_electron import FreeElectron
from pauxy.trial_wavefunction.uhf import UHF
from pauxy.trial_wavefunction.harmonic_oscillator import HarmonicOscillator
from pauxy.estimators.ci import simple_fci_bose_fermi, simple_fci
from pauxy.estimators.hubbard import local_energy_hubbard_holstein, local_energy_hubbard
from pauxy.systems.hubbard_holstein import HubbardHolstein
from pauxy.utils.linalg import reortho

from pauxy.estimators.greens_function import gab_spin

import time
from pauxy.utils.linalg import diagonalise_sorted


from pauxy.estimators.greens_function import gab_spin

import scipy
from scipy.linalg import expm
import scipy.sparse.linalg
from scipy.optimize import minimize

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

import math

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
def local_energy_hubbard_holstein_jax(T,U,g,m,w0, G, X, Lap, Ghalf=None):
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

    
    pe_ph = 0.5 * w0 ** 2 * m * np.sum(X * X)

    ke_ph = -0.5 * np.sum(Lap) / m - 0.5 * w0 * nbasis
    
    rho = G[0].diagonal() + G[1].diagonal()
    e_eph = - g * np.sqrt(m * w0 * 2.0) * np.dot(rho, X)

    etot = ke + pe + pe_ph + ke_ph + e_eph

    Eph = ke_ph + pe_ph
    Eel = ke + pe
    Eeb = e_eph

    return (etot, ke+pe, ke_ph+pe_ph+e_eph)

def gradient(x, nbasis, nup, ndown, T, U, g, m, w0, c0,restricted):
    grad = numpy.array(jax.grad(objective_function)(x, nbasis, nup, ndown, T, U, g, m, w0, c0,restricted))
    return grad

def hessian(x, nbasis, nup, ndown, T, U, g, m, w0, c0, restricted):
    H = numpy.array(jax.hessian(objective_function)(x, nbasis, nup, ndown, T, U, g, m, w0, c0,restricted))
    return H

def hessian_product(x, p, nbasis, nup, ndown, T, U, g, m, w0, c0):
    h = 1e-5
    xph = x + p * h
    xmh = x - p * h
    gph = gradient(xph, nbasis, nup, ndown, T, U, g, m, w0, c0)
    gmh = gradient(xmh, nbasis, nup, ndown, T, U, g, m, w0, c0)

    Hx = (gph - gmh) / (2.0 * h)
    return Hx

@jit
def compute_exp(Ua, tmp, theta_a):
    for i in range(1,15):
        tmp = np.einsum("ij,jk->ik", theta_a, tmp)
        Ua += tmp / math.factorial(i)

    return Ua

def compute_greens_function_from_x (x, nbasis, nup, ndown, c0, restricted):
    shift = x[0:nbasis]
    nbsf = nbasis
    nocca = nup
    noccb = ndown
    nvira = nbasis - nocca
    nvirb = nbasis - noccb
    
    nova = nocca*nvira
    novb = noccb*nvirb
    
    daia = np.array(x[nbsf:nbsf+nova],dtype=np.float64)
    daib = np.array(x[nbsf+nova:nbsf+nova+novb],dtype=np.float64)

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

    G = np.array([Ga, Gb],dtype=np.float64)

    return G

def objective_function (x, nbasis, nup, ndown, T, U, g, m, w0, c0, restricted):
    nbasis = int(round(nbasis))
    nup = int(round(nup))
    ndown = int(round(ndown))

    shift = x[0:nbasis]

    nbsf = nbasis
    nocca = nup
    noccb = ndown
    nvira = nbasis - nocca
    nvirb = nbasis - noccb
    
    nova = nocca*nvira
    novb = noccb*nvirb
    
    daia = np.array(x[nbsf:nbsf+nova],dtype=np.float64)
    daib = np.array(x[nbsf+nova:nbsf+nova+novb],dtype=np.float64)

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

    G = np.array([Ga, Gb],dtype=np.float64)

    phi = HarmonicOscillator(m, w0, order=0, shift = shift)
    Lap = phi.laplacian(shift)

    etot, eel, eph = local_energy_hubbard_holstein_jax(T,U, g,m,w0, G, shift, Lap)

    return etot.real

class CoherentState(object):

    def __init__(self, system, options, verbose=False):
        self.verbose = verbose
        if verbose:
            print ("# Parsing free electron input options.")
        init_time = time.time()
        self.name = "coherent_state"
        self.type = "coherent_state"
        self.trial_type = complex

        self.initial_wavefunction = options.get('initial_wavefunction',
                                              'coherent_state')
        if verbose:
            print ("# Diagonalising one-body Hamiltonian.")

        (self.eigs_up, self.eigv_up) = diagonalise_sorted(system.T[0])
        (self.eigs_dn, self.eigv_dn) = diagonalise_sorted(system.T[1])

        self.reference = options.get('reference', None)
        self.exporder = options.get('exporder', 6)
        self.maxiter = options.get('maxiter', 3)
        
        if verbose:
            print("# exporder in CoherentState is 15 no matter what you entered like {}".format(self.exporder))

        self.psi = numpy.zeros(shape=(system.nbasis, system.nup+system.ndown),
                               dtype=self.trial_type)

        assert (system.name == "HubbardHolstein")

        self.m = system.m
        self.w0 = system.w0

        self.nbasis = system.nbasis
        self.nocca = system.nup
        self.noccb = system.ndown

        self.wfn_file = options.get('wfn_file', None)
        if self.wfn_file is not None:
            if verbose:
                print ("# Reading trial wavefunction from %s"%(self.wfn_file))
            f = h5py.File(self.wfn_file, "r")
            self.shift = f["shift"][()]
            self.psi = f["psi"][()]
            f.close()
            gup = gab(self.psi[:, :system.nup],
                                             self.psi[:, :system.nup]).T
            if (system.ndown > 0):
                gdown = gab(self.psi[:, system.nup:],
                                                   self.psi[:, system.nup:]).T
            else:
                gdown = numpy.zeros_like(gup)

            self.G = numpy.array([gup, gdown])
        else:
            free_electron = options.get('free_electron', False)
            if (free_electron):
                trial_elec = FreeElectron(system, trial=options, verbose=0)
            else:
                trial_elec = UHF(system, trial=options, verbose=0)

            self.psi[:, :system.nup] = trial_elec.psi[:, :system.nup]
            self.psi[:, system.nup:] = trial_elec.psi[:, system.nup:]

            Pa = self.psi[:, :system.nup].dot(self.psi[:, :system.nup].T)
            Va = (numpy.eye(system.nbasis) - Pa).dot(numpy.eye(system.nbasis))
            e, va = numpy.linalg.eigh(Va)

            Pb = self.psi[:, system.nup:].dot(self.psi[:, system.nup:].T)
            Vb = (numpy.eye(system.nbasis) - Pb).dot(numpy.eye(system.nbasis))
            e, vb = numpy.linalg.eigh(Vb)
                
            nocca = system.nup
            noccb = system.ndown
            nvira = system.nbasis-system.nup
            nvirb = system.nbasis-system.ndown

            self.virt = numpy.zeros((system.nbasis, nvira+nvirb))
            self.virt[:,:nvira] = numpy.real(va[:,system.nup:])
            self.virt[:,nvira:] = numpy.real(vb[:,system.ndown:])

            self.G = trial_elec.G.copy()

            gup = gab(self.psi[:, :system.nup],
                                             self.psi[:, :system.nup]).T
            if (system.ndown > 0):
                gdown = gab(self.psi[:, system.nup:],
                                                   self.psi[:, system.nup:]).T
            else:
                gdown = numpy.zeros_like(gup)

            self.G = numpy.array([gup, gdown])

            self.variational = options.get('variational',True)
            self.restricted = options.get('restricted',False)
            print("# restricted = {}".format(self.restricted))

            rho = [numpy.diag(self.G[0]), numpy.diag(self.G[1])]
            self.shift = numpy.sqrt(system.w0*2.0 * system.m) * system.g * (rho[0]+ rho[1]) / (system.m * system.w0**2)
            self.shift = self.shift.real
            print("# Initial shift = {}".format(self.shift[0:3]))

            if (self.variational):
                if (verbose):
                    print("# we will repeat SCF {} times".format(self.maxiter))
                self.run_variational(system, verbose)
                print("# Variational Coherent State Energy = {}".format(self.energy))

            print("# Optimized shift = {}".format(self.shift[0:3]))


        self.boson_trial = HarmonicOscillator(m = system.m, w = system.w0, order = 0, shift=self.shift)

        self.symmetrize = options.get('symmetrize',False)
        
        print("# Symmetrize Coherent State = {}".format(self.symmetrize))
        
        if (self.symmetrize):
            self.perms = numpy.array(list(itertools.permutations([i for i in range(system.nbasis)])))
            # self.perms = numpy.array([self.perms[0].copy()]) # Pick only one permanent
            self.nperms = self.perms.shape[0]
            norm = 1.0 / numpy.sqrt(self.nperms)
            self.coeffs = norm * numpy.ones(self.nperms)
            print("# Number of permutations = {}".format(self.nperms))
        else:
            self.coeffs = 1.0

        self.calculate_energy(system)
        
        if (self.symmetrize):
            print("# Coherent State energy (symmetrized) = {}".format(self.energy))
        else:
            print("# Coherent State energy = {}".format(self.energy))

        self.initialisation_time = time.time() - init_time

        self.spin_projection = options.get('spin_projection',True)
        if (self.spin_projection): # natural orbital
            print("# Spin projection is used")
            Pcharge = self.G[0] + self.G[1]
            e, v = numpy.linalg.eigh(Pcharge)
            self.init = numpy.zeros_like(self.psi)

            idx = e.argsort()[::-1]
            e = e[idx]
            v = v[:,idx]

            self.init[:, :system.nup] = v[:, :system.nup].copy()
            self.init[:, system.nup:] = v[:, :system.ndown].copy()
        else:
            self.init = self.psi.copy()

        # For interface compatability
        self.ndets = 1
        self.bp_wfn = options.get('bp_wfn', None)
        self.error = False
        self.eigs = numpy.append(self.eigs_up, self.eigs_dn)
        self.eigs.sort()

        self._mem_required = 0.0
        self._rchol = None

        if verbose:
            print ("# Updated coherent.")

        if verbose:
            print ("# Finished initialising Coherent State trial wavefunction.")

    def value(self, walker): # value
        if (self.symmetrize):
            phi = 0.0
            shift0 = self.shift.copy()
            for i, perm in enumerate(self.perms):
                shift = shift0[perm].copy()
                boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=shift)
                phi += boson_trial.value(walker.X) * walker.ots[i] * self.coeffs[i]
        else:
            boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=self.shift)
            phi = boson_trial.value(walker.X)
        return phi

    def gradient(self, walker): # gradient / value
        if (self.symmetrize):
            grad = numpy.zeros(self.nbasis, dtype=walker.phi.dtype)
            denom = self.value(walker)
            shift0 = self.shift.copy()
            for i, perm in enumerate(self.perms):
                shift = shift0[perm].copy()
                boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=shift)
                grad += boson_trial.value(walker.X) * boson_trial.gradient(walker.X) * walker.ots[i] * self.coeffs[i]
            grad /= denom
        else:
            boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=self.shift)
            grad = boson_trial.gradient(walker.X)
        return grad

    def laplacian(self, walker): # gradient / value
        if (self.symmetrize):
            lap = numpy.zeros(self.nbasis, dtype=walker.phi.dtype)
            denom = self.value(walker)
            shift0 = self.shift.copy()
            for i, perm in enumerate(self.perms):
                shift = shift0[perm].copy()
                boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=shift)
                walker.Lapi[i] = boson_trial.laplacian(walker.X)
                lap += boson_trial.value(walker.X) * walker.Lapi[i] * walker.ots[i] * self.coeffs[i]
            lap /= denom
        else:
            boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=self.shift)
            lap = boson_trial.laplacian(walker.X)
        return lap

    def bosonic_local_energy(self, walker):

        ke   = - 0.5 * numpy.sum(self.laplacian(walker)) / self.m
        pot  = 0.5 * self.m * self.w0 * self.w0 * numpy.sum(walker.X * walker.X)
        eloc = ke+pot - 0.5 * self.w0 * self.nbasis # No zero-point energy

        return eloc

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
        x = numpy.zeros(system.nbasis + nova + novb, dtype=numpy.float64)
        if (x.shape[0] == 0):
            gup = numpy.zeros((nbsf, nbsf))
            for i in range(nocca):
                gup[i,i] = 1.0
            gdown = numpy.zeros((nbsf, nbsf))
            for i in range(noccb):
                gdown[i,i] = 1.0
            self.G = numpy.array([gup, gdown])
            self.shift = numpy.zeros(nbsf)
            self.calculate_energy(system)
            return

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
        x[:system.nbasis] = self.shift.real.copy() # initial guess
        # self.shift = numpy.zeros(nbsf)
        self.energy = 1e6

        # xconv = numpy.zeros_like(x)
        
        # from jax.experimental import optimizers
        # opt_init, opt_update, get_params = optimizers.adagrad(step_size=0.5)

        # for i in range (self.maxiter): # Try 10 times
        #     ehistory = []
        #     x = numpy.zeros_like(x)
        #     x[:system.nbasis] = self.shift.real.copy() # initial guess
        #     x += numpy.random.randn(x.shape[0]) * 1e-2
        #     x_jax = np.array(x)
        #     opt_state = opt_init(x_jax)

        #     def update(i, opt_state):
        #         params = get_params(opt_state)
        #         gradient = jax.grad(objective_function)(params, float(system.nbasis), float(system.nup), float(system.ndown),\
        #             system.T, system.U, system.g, system.m, system.w0, c0, self.restricted)
        #         return opt_update(i, gradient, opt_state)

        #     eprev = 10000
            
        #     params = get_params(opt_state)
        #     Gprev = compute_greens_function_from_x(params, system.nbasis, system.nup, system.ndown, c0, self.restricted)
        #     shift_prev = x[:system.nbasis]

        #     for t in range(1000):
        #         params = get_params(opt_state)
        #         shift_curr = params[:system.nbasis]
        #         Gcurr = compute_greens_function_from_x(params, system.nbasis, system.nup, system.ndown, c0, self.restricted)
        #         ecurr = objective_function(params, float(system.nbasis), float(system.nup), float(system.ndown),\
        #             system.T, system.U, system.g, system.m, system.w0, c0, self.restricted)
        #         opt_state = update(t, opt_state)

        #         Gdiff = (Gprev-Gcurr).ravel()
        #         shift_diff = shift_prev - shift_curr

        #         rms = numpy.sum(Gdiff**2) + numpy.sum(shift_diff**2)
        #         echange = numpy.abs(ecurr - eprev)

        #         if (echange < 1e-8 and rms < 1e-8):

        #             if verbose:
        #                 print("# {} {} {} {} (converged)".format(t, ecurr, echange, rms))
        #                 self.energy = ecurr
        #                 ehistory += [ecurr]
        #             #     H = hessian(params, float(system.nbasis), float(system.nup), float(system.ndown),\
        #             # system.T, system.U, system.g, system.m, system.w0, c0, self.restricted)
        #             #     eigval, eigvec = numpy.linalg.eigh(H)
        #             #     print(eigval)
        #             break
        #         else:
        #             eprev = ecurr
        #             Gprev = Gcurr
        #             shift_prev = shift_curr
        #             if (verbose and t % 20 == 0):
        #                 print("# {} {} {} {}".format(t, ecurr, echange, rms))

        # x = numpy.array(params)
        # self.shift = x[:nbsf]

        # daia = x[nbsf:nbsf+nova] 
        # daib = x[nbsf+nova:nbsf+nova+novb]
        
        from scipy.optimize import basinhopping
        minimizer_kwargs = {"method":"L-BFGS-B", "jac":True, "args":(float(system.nbasis), float(system.nup), float(system.ndown),system.T, system.U, system.g, system.m, system.w0, c0, self.restricted),
                            "options":{ 'maxls': 20, 'iprint': 2, 'gtol': 1e-10, 'eps': 1e-10, 'maxiter': 15000,\
                                    'ftol': 1.0e-10, 'maxcor': 1000, 'maxfun': 15000,'disp':False}}

        def func(x, nbasis, nup, ndown,T, U, g, m, w0, c0, restricted):
            f = objective_function(x, nbasis, nup, ndown,T, U, g, m, w0, c0, restricted)
            df = gradient(x, nbasis, nup, ndown,T, U, g, m, w0, c0, restricted)
            return f, df
        
        def print_fun(x, f, accepted):
            print("at minimum %.4f accepted %d" % (f, int(accepted)))

        res = basinhopping(func, x, minimizer_kwargs=minimizer_kwargs, callback=print_fun,
                   niter=self.maxiter, niter_success=3)
        print("global minimium at {}".format(res.fun))

        # for i in range (self.maxiter): # Try 10 times
        #     res = minimize(objective_function, x, args=(float(system.nbasis), float(system.nup), float(system.ndown),\
        #             system.T, system.U, system.g, system.m, system.w0, c0, self.restricted), jac=gradient, tol=1e-10,\
        #         method='L-BFGS-B', options={ 'maxls': 20, 'iprint': 2, 'gtol': 1e-10, 'eps': 1e-10, 'maxiter': 15000,\
        #         'ftol': 1.0e-10, 'maxcor': 1000, 'maxfun': 15000,'disp':True})
        #     e = res.fun
        #     if (verbose):
        #         print("# macro iter {} energy is {}".format(i, e))
        #     if (e < self.energy and numpy.abs(self.energy - e) > 1e-6):
        #         self.energy = res.fun
        #         self.shift = self.shift
        #         xconv = res.x.copy()
        #     else:
        #         break
        #     x[:system.nbasis] = numpy.random.randn(self.shift.shape[0]) * 1e-1 + xconv[:nbsf]
        #     x[nbsf:nbsf+nova+novb] = numpy.random.randn(nova+novb) * 1e-1 + xconv[nbsf:]

        self.shift = res.x[:nbsf]
        daia = res.x[nbsf:nbsf+nova] 
        daib = res.x[nbsf+nova:nbsf+nova+novb]

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

        self.psi[:,:nocca] = Ca[:,:nocca]
        self.psi[:,nocca:] = Cb[:,:noccb]
        self.update_electronic_greens_function(system)


        MS = numpy.abs(nocca-noccb) / 2.0
        S2exact = MS * (MS+1.)
        Sij = self.psi[:,:nocca].T.dot(self.psi[:,nocca:])
        S2 = S2exact + min(nocca, noccb) - numpy.sum(numpy.abs(Sij*Sij).ravel())
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

    def update_wfn(self, system, V, verbose=0):
        (self.eigs_up, self.eigv_up) = diagonalise_sorted(system.T[0]+V[0])
        (self.eigs_dn, self.eigv_dn) = diagonalise_sorted(system.T[1]+V[1])

        # I think this is slightly cleaner than using two separate
        # matrices.
        if self.reference is not None:
            self.psi[:, :system.nup] = self.eigv_up[:, self.reference]
            self.psi[:, system.nup:] = self.eigv_dn[:, self.reference]
        else:
            self.psi[:, :system.nup] = self.eigv_up[:, :system.nup]
            self.psi[:, system.nup:] = self.eigv_dn[:, :system.ndown]
            nocca = system.nup
            noccb = system.ndown
            nvira = system.nbasis-system.nup
            nvirb = system.nbasis-system.ndown

            self.virt[:, :nvira] = self.eigv_up[:,nocca:nocca+nvira]
            self.virt[:, nvira:nvira+nvirb] = self.eigv_dn[:,noccb:noccb+nvirb]
        
        gup = gab(self.psi[:, :system.nup],
                                         self.psi[:, :system.nup]).T


        h1 = system.T[0] + V[0]

        if (system.ndown == 0):
            gdown = numpy.zeros_like(gup)
        else:
            gdown = gab(self.psi[:, system.nup:],
                                           self.psi[:, system.nup:]).T
        self.eigs = numpy.append(self.eigs_up, self.eigs_dn)
        self.eigs.sort()
        self.G = numpy.array([gup, gdown])

    def calculate_energy(self, system):
        if self.verbose:
            print ("# Computing trial energy.")
        
        if (self.symmetrize):
            energy = 0.
            e1b = 0. 
            e2b = 0.
            denom = 0.0

            # single coherent state energy
            phi = HarmonicOscillator(system.m, system.w0, order=0, shift = self.shift)
            Lap = phi.laplacian(self.shift)
            (energy_single, e1b_single, e2b_single) = local_energy_hubbard_holstein_jax(system.T,system.U, system.g,system.m,system.w0, self.G, self.shift, Lap)

            num_energy = 0.0
            num_e1b = 0.0
            num_e2b = 0.0

            psia = self.psi[:, :system.nup]
            psib = self.psi[:, system.nup:]

            beta = self.shift * numpy.sqrt(system.m * system.w0 /2.0)

            for iperm in range(self.nperms):
                ipermutation = self.perms[iperm]
                psia_iperm = psia[ipermutation, :].copy()
                psib_iperm = psib[ipermutation, :].copy()
                beta_iperm = beta[ipermutation].copy()

                num_energy += energy_single * self.coeffs[iperm]**2
                num_e1b += e1b_single * self.coeffs[iperm]**2
                num_e2b += e2b_single * self.coeffs[iperm]**2

                denom += self.coeffs[iperm]**2

                for jperm in range(iperm+1, self.nperms):
                    jpermutation = self.perms[jperm]
                    psia_jperm = psia[jpermutation, :].copy()
                    psib_jperm = psib[jpermutation, :].copy()
                    beta_jperm = beta[jpermutation].copy()
                    Ga = gab(psia_iperm, psia_jperm)
                    Gb = gab(psib_iperm, psib_jperm)
                    rho = Ga.diagonal() + Gb.diagonal()

                    ke = numpy.sum(system.T[0] * Ga + system.T[1] * Gb)
                    pe = system.U * numpy.dot(Ga.diagonal(), Gb.diagonal())
                    e_ph = system.w0 * numpy.sum(beta_iperm * beta_jperm) 
                    e_eph = - system.g * numpy.dot(rho, beta_iperm + beta_jperm)

                    overlap = numpy.linalg.det(psia_iperm.T.dot(psia_jperm)) * numpy.linalg.det(psib_iperm.T.dot(psib_jperm)) * numpy.prod(numpy.exp (- 0.5 * (beta_iperm**2 + beta_jperm**2) + beta_iperm*beta_jperm))

                    num_energy += (ke + pe + e_ph + e_eph)*overlap * self.coeffs[iperm] * self.coeffs[jperm] * 2.0 # 2.0 comes from hermiticity
                    
                    num_e1b += (ke + pe)*overlap * self.coeffs[iperm] * self.coeffs[jperm] * 2.0 # 2.0 comes from hermiticity
                    num_e2b += (e_ph + e_eph)*overlap * self.coeffs[iperm] * self.coeffs[jperm] * 2.0 # 2.0 comes from hermiticity

                    denom += overlap * self.coeffs[iperm] * self.coeffs[jperm] * 2.0

            self.energy = num_energy / denom
            self.e1b = num_e1b / denom
            self.e2b = num_e2b / denom

        else:            
                phi = HarmonicOscillator(system.m, system.w0, order=0, shift = self.shift)
                Lap = phi.laplacian(self.shift)
                (self.energy, self.e1b, self.e2b) = local_energy_hubbard_holstein_jax(system.T,system.U,system.g,system.m,system.w0, self.G, self.shift, Lap)

        self.energy = complex(self.energy)
        self.e1b = complex(self.e1b)
        self.e2b = complex(self.e2b)
