import itertools
import cmath
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
from pauxy.utils.io import read_fortran_complex_numbers
from pauxy.utils.linalg import diagonalise_sorted
from pauxy.estimators.mixed import local_energy
from pauxy.estimators.greens_function import gab


from pauxy.estimators.greens_function import gab_spin

import scipy
from scipy.linalg import expm
import scipy.sparse.linalg
from scipy.optimize import minimize
try:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as np
    import jax.scipy.linalg as LA
    import numpy
except ModuleNotFoundError:
    import numpy
    np = numpy
import math

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

def local_energy_hubbard_holstein_jax(system, G, X, Lap, Ghalf=None):
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
    ke = np.sum(system.T[0] * G[0] + system.T[1] * G[1])

    if system.symmetric:
        pe = -0.5*system.U*(G[0].trace() + G[1].trace())

    pe = system.U * np.dot(G[0].diagonal(), G[1].diagonal())

    
    pe_ph = 0.5 * system.w0 ** 2 * system.m * np.sum(X * X)

    ke_ph = -0.5 * np.sum(Lap) / system.m - 0.5 * system.w0 * system.nbasis
    
    rho = G[0].diagonal() + G[1].diagonal()
    e_eph = - system.g * cmath.sqrt(system.m * system.w0 * 2.0) * np.dot(rho, X)

    etot = ke + pe + pe_ph + ke_ph + e_eph

    Eph = ke_ph + pe_ph
    Eel = ke + pe
    Eeb = e_eph

    return (etot, ke+pe, ke_ph+pe_ph+e_eph)

def gradient(x, system, c0, psi,resctricted):
    grad = numpy.array(jax.grad(objective_function)(x, system, c0, psi,resctricted))
    return grad

def hessian(x, system, c0, psi,resctricted):
    H = numpy.array(jax.hessian(objective_function)(x, system, c0, psi,resctricted))
    return H

def hessian_product(x, p, system, c0, psi):
    h = 1e-5
    xph = x + p * h
    xmh = x - p * h
    gph = gradient(xph, system, c0, psi)
    gmh = gradient(xmh, system, c0, psi)

    Hx = (gph - gmh) / (2.0 * h)
    return Hx

def objective_function (x, system, c0, psi, resctricted):
    shift = x[0:system.nbasis]

    nbsf = system.nbasis
    nocca = system.nup
    noccb = system.ndown
    nvira = system.nbasis - nocca
    nvirb = system.nbasis - noccb
    
    nova = nocca*nvira
    novb = noccb*nvirb
    
    daia = np.array(x[nbsf:nbsf+nova] )
    daib = np.array(x[nbsf+nova:nbsf+nova+novb])

    daia = daia.reshape((nvira, nocca))
    daib = daib.reshape((nvirb, noccb))

    if (resctricted):
        daib = jax.ops.index_update(daib, jax.ops.index[:,:], daia)

    theta_a = np.zeros((nbsf, nbsf))
    theta_b = np.zeros((nbsf, nbsf))

    theta_a = jax.ops.index_update(theta_a, jax.ops.index[nocca:nbsf,:nocca], daia)
    theta_a = jax.ops.index_update(theta_a, jax.ops.index[:nocca, nocca:nbsf], -np.transpose(daia))

    theta_b = jax.ops.index_update(theta_b, jax.ops.index[noccb:nbsf,:noccb], daib)
    theta_b = jax.ops.index_update(theta_b, jax.ops.index[:noccb, noccb:nbsf], -np.transpose(daib))

    Ua = np.eye(nbsf)
    tmp = np.eye(nbsf)
    for i in range(1,6):
        tmp = np.einsum("ij,jk->ik", theta_a, tmp)
        Ua += tmp / math.factorial(i)

    C0a = np.array(c0[:nbsf*nbsf].reshape((nbsf,nbsf)))
    Ca = C0a.dot(Ua)
    Ga = gab(Ca[:,:nocca], Ca[:,:nocca])
    
    if (noccb > 0):
        C0b = np.array(c0[nbsf*nbsf:].reshape((nbsf,nbsf)))
        Ub = np.eye(nbsf)
        tmp = np.eye(nbsf)
        for i in range(1,6):
            tmp = np.einsum("ij,jk->ik", theta_b, tmp)
            Ub += tmp / math.factorial(i)
        Cb = C0b.dot(Ub)
        Gb = gab(Cb[:,:noccb], Cb[:,:noccb])

    G = np.array([Ga, Gb])
    phi = HarmonicOscillator(system.m, system.w0, order=0, shift = shift)

    Lap = phi.laplacian(shift)
    etot, eel, eph = local_energy_hubbard_holstein_jax(system, G, shift, Lap)

    return etot.real

class CoherentState(object):

    def __init__(self, system, cplx, trial, parallel=False, verbose=False):
        self.verbose = verbose
        if verbose:
            print ("# Parsing free electron input options.")
        init_time = time.time()
        self.name = "coherent_state"
        self.type = "coherent_state"
        self.initial_wavefunction = trial.get('initial_wavefunction',
                                              'coherent_state')
        if verbose:
            print ("# Diagonalising one-body Hamiltonian.")

        (self.eigs_up, self.eigv_up) = diagonalise_sorted(system.T[0])
        (self.eigs_dn, self.eigv_dn) = diagonalise_sorted(system.T[1])

        self.reference = trial.get('reference', None)
        if cplx:
            self.trial_type = complex
        else:
            self.trial_type = float
        self.read_in = trial.get('read_in', None)
        self.psi = numpy.zeros(shape=(system.nbasis, system.nup+system.ndown),
                               dtype=self.trial_type)

        assert (system.name == "HubbardHolstein")

        self.m = system.m
        self.w0 = system.w0

        self.nbasis = system.nbasis
        self.nocca = system.nup
        self.noccb = system.ndown

        if self.read_in is not None:
            if verbose:
                print ("# Reading trial wavefunction from %s"%(self.read_in))
            try:
                self.psi = numpy.load(self.read_in)
                self.psi = self.psi.astype(self.trial_type)
            except OSError:
                if verbose:
                    print("# Trial wavefunction is not in native numpy form.")
                    print("# Assuming Fortran GHF format.")
                orbitals = read_fortran_complex_numbers(self.read_in)
                tmp = orbitals.reshape((2*system.nbasis, system.ne),
                                       order='F')
                ups = []
                downs = []
                # deal with potential inconsistency in ghf format...
                for (i, c) in enumerate(tmp.T):
                    if all(abs(c[:system.nbasis]) > 1e-10):
                        ups.append(i)
                    else:
                        downs.append(i)
                self.psi[:, :system.nup] = tmp[:system.nbasis, ups]
                self.psi[:, system.nup:] = tmp[system.nbasis:, downs]
        else:
            free_electron = trial.get('free_electron', False)
            if (free_electron):
                trial_elec = FreeElectron(system, False, trial, parallel=False, verbose=0)
            else:
                trial_elec = UHF(system, False, trial, parallel=False, verbose=0)

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

        self.variational = trial.get('variational',True)
        self.restricted = trial.get('restricted',False)
        print("# restricted = {}".format(self.restricted))

        # For interface compatability
        self.coeffs = 1.0
        self.ndets = 1
        self.bp_wfn = trial.get('bp_wfn', None)
        self.error = False
        self.eigs = numpy.append(self.eigs_up, self.eigs_dn)
        self.eigs.sort()

        rho = [numpy.diag(self.G[0]), numpy.diag(self.G[1])]
        self.shift = numpy.sqrt(system.w0*2.0 * system.m) * system.g * (rho[0]+ rho[1]) / (system.m * system.w0**2)
        print("# Initial shift = {}".format(self.shift[0:3]))

        if (self.variational):
            self.run_variational(system)
            print("# Variational Coherent State Energy = {}".format(self.energy))

        print("# Optimized shift = {}".format(self.shift[0:3]))

        self.boson_trial = HarmonicOscillator(m = system.m, w = system.w0, order = 0, shift=self.shift)

        self.symmetrize = trial.get('symmetrize',False)
        print("# Symmetrize Coherent State = {}".format(self.symmetrize))
        if (self.symmetrize):
            self.perms = numpy.array(list(itertools.permutations([i for i in range(system.nbasis)])))
            self.nperms = self.perms.shape[0]
            norm = 1.0 / numpy.sqrt(self.nperms)
            self.coeffs = norm * numpy.ones(self.nperms)
                   
        self.calculate_energy(system)

        print("# Coherent State energy = {}".format(self.energy))

        self.initialisation_time = time.time() - init_time

        self.spin_projection = trial.get('spin_projection',True)
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

        if verbose:
            print ("# Updated coherent.")

        if verbose:
            print ("# Finished initialising Coherent State trial wavefunction.")

    def gradient(self, walker): # gradient / value
        if (self.symmetrize):
            grad = numpy.zeros(self.nbasis, dtype=walker.phi.dtype)
            denom = numpy.sum(walker.weights)
            shift0 = self.shift.copy()
            for i, perm in enumerate(self.perms):
                shift = shift0[perm].copy()
                boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=shift)
                grad += boson_trial.gradient(walker.X) * walker.weights[i]
            grad /= denom
        else:
            boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=self.shift)
            grad = boson_trial.gradient(walker.X)
        return grad

    def laplacian(self, walker): # gradient / value
        if (self.symmetrize):
            lap = numpy.zeros(self.nbasis, dtype=walker.phi.dtype)
            denom = numpy.sum(walker.weights)
            shift0 = self.shift.copy()
            for i, perm in enumerate(self.perms):
                shift = shift0[perm].copy()
                boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=shift)
                walker.Lapi[i] = boson_trial.laplacian(walker.X)
                lap += walker.Lapi[i] * walker.weights[i]
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

    def run_variational(self, system):
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
            c0 = numpy.zeros(nbsf*nbsf*2)
            c0[:nbsf*nbsf] = Ca.ravel()
            c0[nbsf*nbsf:] = Cb.ravel()
        else:
            c0 = numpy.zeros(nbsf*nbsf)
            c0[:nbsf*nbsf] = Ca.ravel()
#       
        x[:system.nbasis] = self.shift.copy() # initial guess
        # self.shift = numpy.zeros(nbsf)
        self.energy = 1e6

        xconv = numpy.zeros_like(x)
        for i in range (10): # Try 10 times
            res = minimize(objective_function, x, args=(system, c0, self, self.restricted), jac=gradient, method='L-BFGS-B', options={'disp':True})
            e = res.fun
            if (e < self.energy and numpy.abs(self.energy - e) > 1e-6):
                self.energy = res.fun
                self.shift = self.shift
                xconv = res.x.copy()
            else:
                break
            x[:system.nbasis] = numpy.random.randn(self.shift.shape[0]) * 1e-1 + xconv[:nbsf]
            x[nbsf:nbsf+nova+novb] = numpy.random.randn(nova+novb) * 1e-1 + xconv[nbsf:]
        
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
        # print("daia = {}".format(daia))
        # print("Ca = {}".format(Ca))
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

        phi = HarmonicOscillator(system.m, system.w0, order=0, shift = self.shift)
        Lap = phi.laplacian(self.shift)
        # (self.energy, self.e1b, self.e2b) = local_energy(system, self.G)
        # self.energy = etot
        (self.energy, self.e1b, self.e2b) = local_energy_hubbard_holstein_jax(system, self.G, self.shift, Lap)
        self.energy = complex(self.energy)
        self.e1b = complex(self.e1b)
        self.e2b = complex(self.e2b)

def unit_test():
    import itertools
    from pauxy.systems.hubbard import Hubbard
    from pauxy.estimators.hubbard import local_energy_hubbard_holstein, local_energy_hubbard_holstein_momentum
    from pauxy.estimators.ci import simple_fci_bose_fermi, simple_fci
    from pauxy.systems.hubbard_holstein import HubbardHolstein, kinetic_lang_firsov
    from pauxy.walkers.single_det import SingleDetWalker
    from pauxy.trial_wavefunction.free_electron import FreeElectron
    from pauxy.trial_wavefunction.harmonic_oscillator import HarmonicOscillator, HarmonicOscillatorMomentum
    import scipy
    import numpy
    import scipy.sparse.linalg
    
    options = {
    "name": "HubbardHolstein",
    "nup": 10,
    "ndown": 10,
    "nx": 20,
    "ny": 1,
    "t": 1.0,
    "U": 0.0,
    "w0": 0.1,
    # "m": 10.0,
    "lambda": 0.5,
    "lang_firsov":False,
    "variational":True,
    "restricted":False
    }

    system = HubbardHolstein (options, verbose=True)
    driver = CoherentState(system, False, options, parallel=False, verbose=1)


if __name__=="__main__":
    unit_test()
