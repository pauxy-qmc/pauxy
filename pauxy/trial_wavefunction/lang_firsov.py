import itertools
import cmath
import h5py
from pauxy.systems.hubbard import Hubbard
from pauxy.trial_wavefunction.free_electron import FreeElectron
from pauxy.trial_wavefunction.coherent_state import CoherentState, gab, compute_exp
from pauxy.trial_wavefunction.harmonic_oscillator import HarmonicOscillator
from pauxy.estimators.ci import simple_fci_bose_fermi, simple_fci
from pauxy.estimators.hubbard import local_energy_hubbard_holstein, local_energy_hubbard
from pauxy.systems.hubbard_holstein import HubbardHolstein
from pauxy.utils.linalg import reortho, overlap

from pauxy.estimators.greens_function import gab_spin

import scipy
import numpy
import scipy.sparse.linalg
from scipy.linalg import expm
from scipy.optimize import minimize

import time
from pauxy.utils.io import read_fortran_complex_numbers
from pauxy.utils.linalg import diagonalise_sorted
from pauxy.estimators.mixed import local_energy

try:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax
    from jax import grad, jit
    import jax.numpy as np
    import jax.scipy.linalg as LA
    import numpy
    from pauxy.trial_wavefunction.coherent_state import gab, compute_exp

except ModuleNotFoundError:
    from pauxy.estimators.greens_function import gab
    import numpy
    np = numpy

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

def gradient(x, nbasis, nup, ndown, T, U, g, m, w0, c0,restricted):
    grad = numpy.array(jax.grad(objective_function)(x, nbasis, nup, ndown, T, U, g, m, w0, c0,restricted))
    return grad

def objective_function (x, nbasis, nup, ndown, T, U, g, m, w0, c0, restricted):
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
    else:
        Gb = np.zeros_like(Ga)

    G = np.array([Ga, Gb],dtype=np.float64)


    beta = x[0:nbasis]

    ti = x[nbsf+nova+novb:]

    rho = G[0].diagonal() + G[1].diagonal()
    rhoa = G[0].diagonal()
    rhob = G[1].diagonal()

    Tscaling = np.zeros_like(T[0])

    factor = np.exp(-0.25*ti**2)
    Tscaling = np.einsum("i,j->ij",factor,factor)
    
    ke = np.sum(T[0] * G[0] * Tscaling + T[1] * G[1] * Tscaling)
    ke += np.sum(rho *(ti*ti*w0/2.0 - g * ti * np.sqrt(2.0)))
    
    eph = w0 * np.sum(beta*beta) 
    
    eeph = 2.0 * np.sum(rho * beta * (ti * w0  / np.sqrt(2.0) - g ))

    pe = U * np.dot(G[0].diagonal(), G[1].diagonal())
    pe += w0 * np.sum(G[0].diagonal()*G[1].diagonal()*ti*ti)
    pe -= 2.0 * g * np.sqrt(2.0) * np.sum(G[0].diagonal()*G[1].diagonal()*ti)
    
    etot = ke + pe + eeph + eph

    return etot.real

class LangFirsov(object):

    def __init__(self, system, options, verbose=False):
        self.verbose = verbose
        if verbose:
            print ("# Parsing free electron input options.")
        init_time = time.time()
        self.name = "lang_firsov"
        self.type = "lang_firsov"
        self.trial_type = complex

        self.initial_wavefunction = options.get('initial_wavefunction',
                                              'lang_firsov')
        if verbose:
            print ("# Diagonalising one-body Hamiltonian.")

        (self.eigs_up, self.eigv_up) = diagonalise_sorted(system.T[0])
        (self.eigs_dn, self.eigv_dn) = diagonalise_sorted(system.T[1])

        self.reference = options.get('reference', None)
        self.exporder = options.get('exporder', 6)
        self.maxiter = options.get('maxiter', 3)
        self.maxscf = options.get('maxscf', 500)
        self.ueff = options.get('ueff', system.U)
        
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

        self.algorithm = options.get('algorithm',"bfgs")
        self.random_guess = options.get('random_guess',False)
        self.symmetrize = options.get('symmetrize',False)

        self.linearize = options.get('linearize',True)
        if verbose:
            print("# random guess = {}".format(self.random_guess))

        self.dry_run = options.get('dry_run',False)
        if verbose:
            print("# dry_run = {}".format(self.dry_run))
            if (self.dry_run):
                print("# dry run should reproduce the coherent state result (i.e., no Lang-Firsov amplitudes)")

        if (self.linearize):
            print("# Linearize Lang-Firsov State for the evaluation of local energy and overlap")
        else:
            print("# exact application of Lang-Firsov AFQMC is unsupported")
            exit()

        print("# Linearize Lang-Firsov State = {}".format(self.linearize))
        print("# Symmetrize Lang-Firsov State = {}".format(self.symmetrize))

        if (self.symmetrize):
            print("# Symmetrize Lang-Firsov State currently unsupported")
            exit()


        self.wfn_file = options.get('wfn_file', None)
        self.variational = options.get('variational', True)
        
        self.coeffs = None
        self.perms = None

        if self.wfn_file is not None:
            if verbose:
                print ("# Reading trial wavefunction from %s"%(self.wfn_file))
            f = h5py.File(self.wfn_file, "r")
            self.shift = f["shift"][()].real
            self.psi = f["psi"][()]
            self.tis = f["tis"][()] # Lang-Firsov amplitudes
            f.close()

            if (len(self.psi.shape) == 3):
                if verbose:
                    print("# MultiLangFirsov trial detected")
                print("# MultiLangFirsov trial currently unsupported")
                exit()
                self.symmetrize = True
                self.perms = None

                f = h5py.File(self.wfn_file, "r")
                self.coeffs = f["coeffs"][()]
                f.close()

                self.nperms = self.coeffs.shape[0]

                assert(self.nperms == self.psi.shape[0])
                assert(self.nperms == self.shift.shape[0])

                self.G = None
                if verbose:
                    print("# A total of {} coherent states are used".format(self.nperms))

            else:
                gup = gab(self.psi[:, :system.nup],
                                                 self.psi[:, :system.nup]).T
                if (system.ndown > 0):
                    gdown = gab(self.psi[:, system.nup:],
                                                       self.psi[:, system.nup:]).T
                else:
                    gdown = numpy.zeros_like(gup)

                self.G = numpy.array([gup, gdown], dtype=self.psi.dtype)

        else:
            trial = CoherentState(system, options=options, verbose=self.verbose)
            self.boson_trial = trial.boson_trial
            self.restricted = trial.restricted
            self.init_guess_file = trial.init_guess_file
            self.psi = trial.psi.copy()
            self.shift = trial.shift.copy()
            self.beta = self.shift * numpy.sqrt(system.m * system.w0 /2.0)
            self.coeffs = trial.coeffs
            self.tis = numpy.zeros_like(self.shift)
            self.G = trial.G.copy()
            nocca = system.nup
            noccb = system.ndown
            nvira = system.nbasis - system.nup
            nvirb = system.nbasis - system.ndown
            self.virt = numpy.zeros((system.nbasis, nvira+nvirb),dtype=self.psi.dtype)
            Ca = projected_virtuals(self.psi[:,:system.nup])
            Cb = projected_virtuals(self.psi[:,system.nup:])
            self.virt[:,:nvira] = Ca[:,system.nup:]
            self.virt[:,nvira:] = Cb[:,system.ndown:]

            if (self.variational and not self.dry_run):
                if (verbose):
                    print("# we will repeat SCF {} times".format(self.maxiter))
                self.run_variational(system, verbose)
                print("# Variational Lang-Firsov state energy = {}".format(self.energy))

            print("# Optimized shift = {}".format(self.shift[0:5]))
            print("# Optimized amplitudes = {}".format(self.tis[0:5]))

            self.boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=self.shift)

            ovlp_a = numpy.linalg.det(overlap(trial.psi[:,:nocca],self.psi[:,:nocca]))
            ovlp_b = numpy.linalg.det(overlap(trial.psi[:,nocca:],self.psi[:,nocca:]))
            print("# Overlap with coherent state = {}".format(ovlp_a*ovlp_b))

        self.calculate_energy(system)
        print("# Lang-Firsov trial state energy = {}".format(self.energy))

        self.initialisation_time = time.time() - init_time

        self.spin_projection = options.get('spin_projection',False)
        if (self.spin_projection and not self.symmetrize): # natural orbital
            print("# Spin projection is used")
            Pcharge = self.G[0] + self.G[1]
            e, v = numpy.linalg.eigh(Pcharge)
            self.init = numpy.zeros_like(self.psi)

            idx = e.argsort()[::-1]
            e = e[idx]
            v = v[:,idx]

            self.init[:, :system.nup] = v[:, :system.nup].copy()
            if (system.ndown > 0):
                self.init[:, system.nup:] = v[:, :system.ndown].copy()
        else:
            if (len(self.psi.shape) == 3):
                self.init = self.psi[0,:,:].copy()
            else:
                self.init = self.psi.copy()
        
        nocca = system.nup
        noccb = system.ndown
        nvira = system.nbasis-system.nup
        nvirb = system.nbasis-system.ndown

        MS = numpy.abs(nocca-noccb) / 2.0
        S2exact = MS * (MS+1.)
        Sij = self.psi[:,:nocca].T.dot(self.psi[:,nocca:])
        self.S2 = S2exact + min(nocca, noccb) - numpy.sum(numpy.abs(Sij*Sij).ravel())
        if (verbose):
            print("# <S^2> = {: 3f}".format(self.S2))

        # For interface compatability
        self.ndets = 1
        self.bp_wfn = options.get('bp_wfn', None)
        self.error = False
        self.eigs = numpy.append(self.eigs_up, self.eigs_dn)
        self.eigs.sort()

        self._mem_required = 0.0
        self._rchol = None
        self._eri = None
        self._UVT = None

        if verbose:
            print ("# Finished initialising variational Lang-Firsov trial wavefunction.")

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
        x = numpy.zeros(system.nbasis + nova + novb + system.nbasis, dtype=numpy.float64)
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

        x[:system.nbasis] = self.beta.real.copy() # initial guess
        self.energy = 1e6

        if self.algorithm == "bfgs":
            for i in range (self.maxiter): # Try 10 times
                res = minimize(objective_function, x, args=(float(system.nbasis), float(system.nup), float(system.ndown),\
                        system.T, self.ueff, system.g, system.m, system.w0, c0, self.restricted), jac=gradient, tol=1e-10,\
                    method='L-BFGS-B',\
                    options={ 'maxls': 20, 'iprint': 2, 'gtol': 1e-10, 'eps': 1e-10, 'maxiter': self.maxscf,\
                    'ftol': 1.0e-10, 'maxcor': 1000, 'maxfun': 15000,'disp':False})
                e = res.fun
                if (verbose):
                    print("# macro iter {} energy is {}".format(i, e))
                if (e < self.energy and numpy.abs(self.energy - e) > 1e-6):
                    self.energy = res.fun
                    self.shift = self.shift
                    xconv = res.x.copy()
                else:
                    break
                x[:system.nbasis] = numpy.random.randn(nbsf) * 1e-1 + xconv[:nbsf]
                x[nbsf:nbsf+nova+novb] = numpy.random.randn(nova+novb) * 1e-1 + xconv[nbsf:nbsf+nova+novb]
                x[nbsf+nova+novb:] = numpy.random.randn(nbsf) * 1e-1 + xconv[nbsf+nova+novb:]

            self.beta = res.x[:nbsf]
            self.shift = self.beta / numpy.sqrt(system.m * system.w0 /2.0)
            daia = res.x[nbsf:nbsf+nova] 
            daib = res.x[nbsf+nova:nbsf+nova+novb]
            self.tis = res.x[nbsf+nova+novb:]
        else:
            print("# Unknown optimizer")
            exit()

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

    def calc_elec_overlap(self, walker):
        """Caculate overlap with walker wavefunction (electronic only).

        Parameters
        ----------
        walker : object
            walker wavefunction object.

        Returns
        -------
        ot : float / complex
            Overlap.
        """
        na = walker.nup
        Oalpha = numpy.dot(self.psi[:,:na].conj().T, walker.phi[:,:na])
        sign_a, logdet_a = numpy.linalg.slogdet(Oalpha)
        nb = walker.ndown
        logdet_b, sign_b = 0.0, 1.0
        if nb > 0:
            Obeta = numpy.dot(self.psi[:,na:].conj().T, walker.phi[:,na:])
            sign_b, logdet_b = numpy.linalg.slogdet(Obeta)
        
        ot = sign_a*sign_b*numpy.exp(logdet_a+logdet_b-walker.log_shift)

        return ot

    def value(self, walker): # value

        if (self.linearize):
            # boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=self.shift)
            phi = self.boson_trial.value(walker.X)
            dphi = self.boson_trial.gradient(walker.X)

            elec_ot = self.calc_elec_overlap(walker)

            self.greens_function(walker)
            rho = (walker.G[0].diagonal() + walker.G[1].diagonal())

            term1 = phi * elec_ot
            term2 = -numpy.sum(self.tis * rho * dphi) * phi * elec_ot

            overlap = term1 + term2 

        return overlap

#   Compute  <\psi_T | \partial_i D | \psi> / <\psi_T| D | \psi>
    def gradient(self, walker): # gradient / value
        if (self.linearize):
            # boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=self.shift)
            phi = self.boson_trial.value(walker.X)
            dphi = self.boson_trial.gradient(walker.X)
            ddphi = self.boson_trial.hessian(walker.X) # hessian

            grad = numpy.zeros(self.nbasis, dtype=walker.phi.dtype)
            
            denom = self.value(walker)
            
            elec_ot = self.calc_elec_overlap(walker)

            self.greens_function(walker)
            rho = (walker.G[0].diagonal() + walker.G[1].diagonal())

            term1 = dphi * elec_ot * phi
            term2 = -numpy.einsum("i,ik->k",self.tis*rho,ddphi) * elec_ot * phi
            grad = (term1+term2) / denom

        return grad

#   Compute  <\psi_T | \partial_i^2 D | \psi> / <\psi_T| D | \psi>
    def laplacian(self, walker):
        if (self.linearize):
            # boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=self.shift)
            phi = self.boson_trial.value(walker.X)
            d2phi = self.boson_trial.laplacian(walker.X)
            dd2phi = self.boson_trial.grad_laplacian(walker.X)

            lap = numpy.zeros(self.nbasis, dtype=walker.phi.dtype)
            
            denom = self.value(walker)
            
            elec_ot = self.calc_elec_overlap(walker)
            
            self.greens_function(walker)
            rho = (walker.G[0].diagonal() + walker.G[1].diagonal())

            term1 = d2phi * elec_ot * phi
            term2 = -numpy.einsum("i,ik->k", self.tis*rho, dd2phi) * elec_ot * phi

            lap = (term1+term2) / denom

        return lap

    def calculate_energy(self, system):
        if self.verbose:
            print ("# Computing trial energy.")

        beta = self.beta
        tis = self.tis

        rho = self.G[0].diagonal() + self.G[1].diagonal()
        rhoa = self.G[0].diagonal()
        rhob = self.G[1].diagonal()

        Tscaling = numpy.zeros_like(system.T[0])
        factor = numpy.exp(-0.25*tis**2)
        Tscaling = numpy.einsum("i,j->ij",factor,factor)
        
        ke = numpy.sum(system.T[0] * self.G[0] * Tscaling + system.T[1] * self.G[1] * Tscaling)
        ke += numpy.sum(rho *(tis*tis*system.w0/2.0 - system.g * tis * numpy.sqrt(2.0)))
        
        eph = system.w0 * numpy.sum(beta*beta) 
        
        eeph = 2.0 * numpy.sum(rho * beta * (tis * system.w0  / numpy.sqrt(2.0) - system.g ))

        pe = system.U * numpy.dot(rhoa, rhob)
        pe += system.w0 * numpy.sum(rhoa*rhob*tis*tis)
        pe -= 2.0 * system.g * numpy.sqrt(2.0) * numpy.sum(rhoa*rhob*tis)
        
        self.energy = ke + pe + eeph + eph
        self.energy = self.energy.real
        
        print("# Eee, Ekin, Eph, Eeph = {}, {}, {}, {}".format(pe.real, ke.real, eph.real, eeph.real))

    def calc_otrial(self, walker):

        ot = self.value(walker)

        return ot

    def inverse_overlap(self, walker):
        ''' place holder. this should never be actually used'''
        return
    
    def calc_overlap(self, walker):

        ot = self.value(walker)

        return ot

    def greens_function(self, walker):
        """Compute walker's green's function.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        Returns
        -------
        det : float64 / complex128
            Determinant of overlap matrix.
        """
        nup = walker.nup
        ndown = walker.ndown

        ovlp = numpy.dot(walker.phi[:,:nup].T, self.psi[:,:nup].conj())
        walker.Gmod[0] = numpy.dot(scipy.linalg.inv(ovlp), walker.phi[:,:nup].T)
        walker.G[0] = numpy.dot(self.psi[:,:nup].conj(), walker.Gmod[0])
        sign_a, log_ovlp_a = numpy.linalg.slogdet(ovlp)
        sign_b, log_ovlp_b = 1.0, 0.0
        if ndown > 0:
            ovlp = numpy.dot(walker.phi[:,nup:].T, self.psi[:,nup:].conj())
            sign_b, log_ovlp_b = numpy.linalg.slogdet(ovlp)
            walker.Gmod[1] = numpy.dot(scipy.linalg.inv(ovlp), walker.phi[:,nup:].T)
            walker.G[1] = numpy.dot(self.psi[:,nup:].conj(), walker.Gmod[1])
        det = sign_a*sign_b*numpy.exp(log_ovlp_a+log_ovlp_b-walker.log_shift)
        return det

    def fb_greens_function(self, walker):
        """Compute walker's green's function for applying the force bias.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        Returns
        -------
        det : float64 / complex128
            Determinant of overlap matrix.
        """
        G = self.full_greens_function(walker)
        return G

    def full_greens_function(self, walker):
        """Compute walker's green's function accounting for both bosons and fermions

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        Returns
        -------
        det : float64 / complex128
            Determinant of overlap matrix.
        """
        self.greens_function(walker)

        if (self.linearize):
            # boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=self.shift)
            phi = self.boson_trial.value(walker.X)
            dphi = self.boson_trial.gradient(walker.X)

            elec_ot = self.calc_elec_overlap(walker)
            term1 = [walker.G[0] * elec_ot * phi, walker.G[1] * elec_ot * phi]

            tdphi = self.tis*dphi
            
            # note that here we don't multiply it by elec_ot
            rho = (walker.G[0].diagonal() + walker.G[1].diagonal())

            term2_a = -numpy.einsum("i,ij->ij",tdphi, walker.G[0]) * elec_ot\
            - numpy.sum(tdphi*rho) * walker.G[0] * elec_ot * elec_ot\
            + numpy.einsum("ik,k,kj->ij",walker.G[0], tdphi, walker.G[0], optimize=True) * elec_ot * elec_ot
            term2_b = -numpy.einsum("i,ij->ij",tdphi, walker.G[1]) * elec_ot\
            - numpy.sum(tdphi*rho) * walker.G[1] * elec_ot * elec_ot\
            + numpy.einsum("ik,k,kj->ij",walker.G[1], tdphi, walker.G[1], optimize=True) * elec_ot * elec_ot

            term2_a *= phi
            term2_b *= phi

            denom = self.value(walker)

            G = [(term1[0]+term2_a)/denom, (term1[1]+term2_b)/denom]
        return G


    def bosonic_local_energy(self, walker):

        self.greens_function(walker) # update walker's electronic Green's function
        denom = self.value(walker)

        ke   = - 0.5 * numpy.sum(self.laplacian(walker)) / self.m
        

        # pot  = 0.5 * self.m * self.w0 * self.w0 * numpy.sum(walker.X * walker.X)

        phi = self.boson_trial.value(walker.X)
        dphi = self.boson_trial.gradient(walker.X)
        elec_ot = self.calc_elec_overlap(walker)
        rho = walker.G[0].diagonal() + walker.G[1].diagonal()

        pot_term1 = 0.5 * self.m * self.w0 * self.w0 * numpy.sum(walker.X * walker.X) * elec_ot * phi
        pot_term2 = -0.5 * self.m * self.w0 * self.w0 * numpy.sum(walker.X * walker.X) * numpy.sum(rho*self.tis*dphi) * elec_ot * phi
        pot_term3 = -self.m * self.w0 * self.w0 * numpy.sum(walker.X * rho * self.tis) * elec_ot * phi

        pot = (pot_term1 + pot_term2 + pot_term3)/denom

        eloc = ke+pot - 0.5 * self.w0 * self.nbasis # No zero-point energy

        return eloc


    def local_energy(self, system, walker):
        
        assert (self.linearize)

        # boson_trial = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=self.shift)
        phi = self.boson_trial.value(walker.X)
        dphi = self.boson_trial.gradient(walker.X)
        elec_ot = self.calc_elec_overlap(walker)

        denom = self.value(walker)
        lap = self.laplacian(walker) # compute walker's laplacian
        self.greens_function(walker) # update walker's electronic Green's function
        G = self.full_greens_function(walker) # compute the true walker's one-body greens function (not just electronic)

        ke = numpy.sum(system.T[0] * G[0] + system.T[1] * G[1])

        # two-body needs to be coded up!
        # if system.symmetric: # super dumb thing that I never understood
        #     pe = -0.5*system.U*(G[0].trace() + G[1].trace())
        # pe = system.U * numpy.dot(G[0].diagonal(), G[1].diagonal()) 
        pe = 0.0
        assert(numpy.abs(system.U) < 1e-8)

        # e_eph here...
        rho = walker.G[0].diagonal() + walker.G[1].diagonal()
        e_eph_term1 = - system.g * numpy.sqrt(system.m * system.w0 * 2.0) * numpy.dot(rho, walker.X) * elec_ot * phi

        kdelta = numpy.eye(system.nbasis)
        nkni = numpy.einsum("ki,ki->ki", kdelta, walker.G[0]+walker.G[1]) * elec_ot\
             + numpy.einsum("i,k->ki", rho,rho) * elec_ot * elec_ot\
             - walker.G[0] * walker.G[0].T * elec_ot * elec_ot\
             - walker.G[1] * walker.G[1].T * elec_ot * elec_ot
        
        e_eph_term2 = +system.g * numpy.sqrt(system.m * system.w0 * 2.0) * numpy.einsum("k,ki,i->", self.tis*dphi, nkni, walker.X, optimize=True) * phi\
                      +system.g * numpy.sqrt(system.m * system.w0 * 2.0) * numpy.einsum("i,ii->",self.tis,nkni) * phi

        e_eph = (e_eph_term1+ e_eph_term2) / denom

        # phonon energy here
        ke_ph = -0.5 * numpy.sum(lap) / system.m - 0.5 * system.w0 * system.nbasis
        pe_ph = 0.5 * system.w0 ** 2 * system.m * numpy.sum(walker.X * walker.X)
        
        etot = ke + pe + pe_ph + ke_ph + e_eph

        Eph = ke_ph + pe_ph
        Eel = ke + pe
        Eeb = e_eph

        return (etot, ke+pe, ke_ph+pe_ph+e_eph)



