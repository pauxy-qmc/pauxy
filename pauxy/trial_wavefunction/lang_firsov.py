import itertools
import cmath
from pauxy.systems.hubbard import Hubbard
from pauxy.trial_wavefunction.free_electron import FreeElectron
from pauxy.trial_wavefunction.harmonic_oscillator import HarmonicOscillator
from pauxy.estimators.ci import simple_fci_bose_fermi, simple_fci
from pauxy.estimators.hubbard import local_energy_hubbard_holstein, local_energy_hubbard
from pauxy.systems.hubbard_holstein import HubbardHolstein
from pauxy.utils.linalg import reortho

from pauxy.estimators.greens_function import gab_spin

import scipy
import numpy
import scipy.sparse.linalg
from scipy.linalg import expm
from scipy.optimize import minimize

import numpy
import time
from pauxy.utils.io import read_fortran_complex_numbers
from pauxy.utils.linalg import diagonalise_sorted
from pauxy.estimators.mixed import local_energy
from pauxy.estimators.greens_function import gab

def decode_basis(nx, ny, i):
    """Return cartesian lattice coordinates from basis index.

    Consider a 3x3 lattice then we index lattice sites like::

        (0,2) (1,2) (2,2)       6 7 8
        (0,1) (1,1) (2,1)  ->   3 4 5
        (0,0) (1,0) (2,0)       0 1 2

    i.e., i = i_x + n_x * i_y, and i_x = i%n_x, i_y = i//nx.

    Parameters
    ----------
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.
    i : int
        Basis index (same for up and down spins).
    """
    if ny == 1:
        return numpy.array([i%nx])
    else:
        return numpy.array([i%nx, i//nx])


def objective_function_rotation (x, system, psi, c0):

    phi = x[0:system.nbasis]

    nbsf = system.nbasis
    nocca = system.nup
    noccb = system.ndown
    nvira = system.nbasis - nocca
    nvirb = system.nbasis - noccb
    
    nova = nocca*nvira
    novb = noccb*nvirb
    
    daia = x[nbsf:nbsf+nova] 
    daib = x[nbsf+nova:nbsf+nova+novb]

    daia = daia.reshape((nvira, nocca))
    daib = daib.reshape((nvirb, noccb))

    Ua = numpy.zeros((nbsf, nbsf))
    Ub = numpy.zeros((nbsf, nbsf))

    Ua[nocca:nbsf,:nocca] = daia.copy()
    Ua[:nocca, nocca:nbsf] = -daia.T.copy()

    Ub[noccb:nbsf,:noccb] = daib.copy()
    Ub[:noccb, noccb:nbsf] = -daib.T.copy()

    C0a = c0[:nbsf*nbsf].reshape((nbsf,nbsf))
    C0b = c0[nbsf*nbsf:].reshape((nbsf,nbsf))

    Ua = expm(Ua)
    Ub = expm(Ub)

    Ca = C0a.dot(Ua)
    Cb = C0b.dot(Ub)

    psi.psi[:,:nocca] = Ca[:,:nocca].copy()
    psi.psi[:,nocca:] = Cb[:,:noccb].copy()

    psi.update_electronic_greens_function(system)

    ni = numpy.diag(psi.G[0]+psi.G[1])
    nia = numpy.diag(psi.G[0])
    nib = numpy.diag(psi.G[1])

    sqrttwomw = numpy.sqrt(system.m * system.w0*2.0)
    
    # gamma = x[-1]
    # gamma = 0.0
    gamma = system.g * numpy.sqrt(2.0 * system.m / system.w0)
    
    alpha = gamma * numpy.sqrt(system.m * system.w0 / 2.0)

    Eph = system.w0 * numpy.sum(phi*phi)
    Eeph = (gamma * system.w0 - system.g * sqrttwomw) * numpy.sum (2.0 * phi / sqrttwomw * ni)
    Eeph += (gamma**2 * system.w0 / 2.0 - system.g * gamma * sqrttwomw) * numpy.sum(ni)

    Eee = (system.U + gamma**2 * system.w0 - 2.0 * system.g * gamma * sqrttwomw) * numpy.sum(nia * nib)

    Ekin = numpy.exp (-alpha * alpha) * numpy.sum(system.T[0] * psi.G[0] + system.T[1] * psi.G[1])
    # Ekin = numpy.sum(system.T[0] * psi.G[0] + system.T[1] * psi.G[1])
    etot = Eph + Eeph + Eee + Ekin
    return etot.real

class LangFirsov(object):

    def __init__(self, system, cplx, trial, parallel=False, verbose=False):
        self.verbose = verbose
        if verbose:
            print ("# Parsing free electron input options.")
        init_time = time.time()
        self.name = "lang_firsov"
        self.type = "lang_firsov"
        self.initial_wavefunction = trial.get('initial_wavefunction',
                                              'lang_firsov')
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
                self.virt = numpy.zeros((system.nbasis, nvira+nvirb))

                self.virt[:, :nvira] = self.eigv_up[:,nocca:nocca+nvira]
                self.virt[:, nvira:nvira+nvirb] = self.eigv_dn[:,noccb:noccb+nvirb]

        gup = gab(self.psi[:, :system.nup],
                                         self.psi[:, :system.nup]).T
        gdown = gab(self.psi[:, system.nup:],
                                           self.psi[:, system.nup:]).T

        self.G = numpy.array([gup, gdown])

        # For interface compatability
        self.coeffs = 1.0
        self.ndets = 1
        self.bp_wfn = trial.get('bp_wfn', None)
        self.error = False
        self.eigs = numpy.append(self.eigs_up, self.eigs_dn)
        self.eigs.sort()

        self.gamma = system.g * numpy.sqrt(2.0 * system.m / system.w0)
        self.shift = numpy.zeros(system.nbasis)
        
        self.initialisation_time = time.time() - init_time
        self.init = self.psi.copy()

        self.calculate_energy(system)

        print("# Lang-Firsov gamma = {}".format(self.gamma))
        print("# Lang-Firsov shift = {}".format(self.shift))
        print("# Lang-Firsov energy = {}".format(self.energy))

        if verbose:
            print ("# Updated lang_firsov.")

        if verbose:
            print ("# Finished initialising Lang-Firsov trial wavefunction.")

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
        x = numpy.zeros(system.nbasis + nova+novb + 1)
        
        rho = [numpy.diag(self.G[0]), numpy.diag(self.G[1])]
        self.shift = numpy.sqrt(system.w0*2.0 * system.m) * system.g * (rho[0]+ rho[1]) / (system.m * system.w0**2)
        nX = numpy.array([numpy.diag(self.shift), numpy.diag(self.shift)], dtype=numpy.float64)
        V = - numpy.real(system.g * cmath.sqrt(system.m * system.w0 * 2.0) * nX)
        self.update_wfn(system, V)
#         
#         
        Ca = numpy.zeros((nbsf,nbsf))
        Ca[:,:nocca] = self.psi[:,:nocca]
        Ca[:,nocca:] = self.virt[:,:nvira]
        Cb = numpy.zeros((nbsf,nbsf))
        Cb[:,:noccb] = self.psi[:,nocca:]
        Cb[:,noccb:] = self.virt[:,nvira:]
#         
        c0 = numpy.zeros(nbsf*nbsf*2)
        c0[:nbsf*nbsf] = Ca.ravel()
        c0[nbsf*nbsf:] = Cb.ravel()
#       
        self.calculate_energy(system)
        print("self.shift = {}".format(self.shift))
        for i in range (10): # Try 10 times
            res = minimize(objective_function_rotation, x, args=(system, self, c0), method='BFGS', options={'disp':False})
            e = res.fun
            if (e < self.energy):
                self.energy = res.fun
                self.shift = res.x[:system.nbasis] / numpy.sqrt(system.m * system.w0 / 2.0)
                self.gamma = res.x[-1]
            x[:system.nbasis] = self.shift + numpy.random.randn(self.shift.shape[0])
            x[nbsf:nbsf+nova+novb] = numpy.random.randn(nova+novb)
            x[-1] = numpy.random.randn(1)

        daia = res.x[nbsf:nbsf+nova] 
        daib = res.x[nbsf+nova:nbsf+nova+novb]

        daia = daia.reshape((nvira, nocca))
        daib = daib.reshape((nvirb, noccb))

        Ua = numpy.zeros((nbsf, nbsf))
        Ub = numpy.zeros((nbsf, nbsf))

        Ua[nocca:nbsf,:nocca] = daia.copy()
        Ua[:nocca, nocca:nbsf] = -daia.T.copy()

        Ub[noccb:nbsf,:noccb] = daib.copy()
        Ub[:noccb, noccb:nbsf] = -daib.T.copy()

        C0a = c0[:nbsf*nbsf].reshape((nbsf,nbsf))
        C0b = c0[nbsf*nbsf:].reshape((nbsf,nbsf))

        Ua = expm(Ua)
        Ub = expm(Ub)

        Ca = C0a.dot(Ua)
        Cb = C0b.dot(Ub)

        self.psi[:,:nocca] = Ca[:,:nocca]
        self.psi[:,nocca:] = Cb[:,:noccb]

        self.update_electronic_greens_function(system)
        
    def update_electronic_greens_function(self, system, verbose=0):
        gup = gab(self.psi[:, :system.nup],
                                         self.psi[:, :system.nup]).T
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
        gdown = gab(self.psi[:, system.nup:],
                                           self.psi[:, system.nup:]).T
        self.eigs = numpy.append(self.eigs_up, self.eigs_dn)
        self.eigs.sort()
        self.G = numpy.array([gup, gdown])

#   Compute D_{jj} 
    def compute_Dvec(self, walker):

        phi0 = self.shift.copy()
        nbsf = walker.X.shape[0]
        D = numpy.zeros(nbsf)
        for i in range(nbsf):
            phi = phi0.copy()
            phi[i] += self.gamma
            
            # QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=X)
            # D[i] = QHO.value(walker.X)

            QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=phi0[i])
            denom = QHO.value(walker.X[i])
            
            QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=phi[i])
            num = QHO.value(walker.X[i])

            D[i] = num/denom

        return D

#   Compute \sum_i \partial_i D_{jj} = A_{jj}
    def compute_dDvec(self, walker):

        phi0 = self.shift.copy()
        nbsf = walker.X.shape[0]
        dD = numpy.zeros(nbsf)
        
        for i in range(nbsf):
            phi = phi0.copy()
            phi[i] += self.gamma
            # QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=X[i])
            # dD[i] = QHO.gradient(walker.X[i]) * QHO.value(walker.X[i]) # gradient is actually grad / value
            
            QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=phi0[i])
            denom = QHO.gradient(walker.X[i])

            QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=phi[i])
            num = QHO.gradient(walker.X[i])

            dD[i] = num/denom
        return dD


#   Compute \sum_i \partial_i^2 D_{jj} = A_{jj}
    def compute_d2Dvec(self, walker):

        phi0 = self.shift.copy()
        nbsf = walker.X.shape[0]
        d2D = numpy.zeros(nbsf)
        
        for i in range(nbsf):
            phi = phi0.copy()
            phi[i] += self.gamma

            # QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=phi[i])
            # d2D[i] = QHO.laplacian(walker.X[i]) * QHO.value(walker.X[i]) # gradient is actually grad / value
            
            QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=phi0[i])
            denom = QHO.laplacian(walker.X[i])

            QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=phi[i])
            num = QHO.laplacian(walker.X[i])

            d2D[i] = num/denom

        return d2D


#   Compute  <\psi_T | \partial_i D | \psi> / <\psi_T| D | \psi>
    def gradient(self, walker):

        psi0 = self.psi.copy()
        
        nbsf = walker.X.shape[0]
        
        grad = numpy.zeros(nbsf)

        # Compute denominator
        # Dvec = self.compute_Dvec(walker)
        # self.psi = numpy.einsum("m,mi->mi",Dvec, psi0)
        # ot_denom = walker.calc_otrial(self)
        self.psi[:,:self.nocca] = numpy.einsum("m,mi->mi",Dvec, psi0a)
        self.psi[:,self.nocca:] = psi0b
        walker.inverse_overlap(self)
        ot_denom = walker.calc_otrial(self)

        self.psi[:,:self.nocca] = psi0a
        self.psi[:,self.nocca:] = numpy.einsum("m,mi->mi",Dvec, psi0b)
        walker.inverse_overlap(self)
        ot_denom += walker.calc_otrial(self)

        # Compute numerator
        dD = self.compute_dDvec(walker)

        for i in range (nbsf):
            dDvec = numpy.zeros_like(dD)
            dDvec[i] = dD[i]
            
            self.psi[:,:self.nocca] = numpy.einsum("m,mi->mi",dDvec, psi0a)
            self.psi[:,self.nocca:] = psi0b
            walker.inverse_overlap(self)
            ot_num = walker.calc_otrial(self)
            
            self.psi[:,:self.nocca] = psi0a
            self.psi[:,self.nocca:] = numpy.einsum("m,mi->mi",dDvec, psi0b)
            walker.inverse_overlap(self)
            ot_num += walker.calc_otrial(self)
            grad[i] = ot_num / ot_denom
        
        self.psi = psi0.copy()
        return grad

#   Compute  <\psi_T | \partial_i^2 D | \psi> / <\psi_T| D | \psi>
    def laplacian(self, walker):

        psi0 = self.psi.copy()

        psi0a = psi0[:,:self.nocca]
        psi0b = psi0[:,self.nocca:]
        
        nbsf = walker.X.shape[0]
        lap = numpy.zeros(nbsf)

        # Compute denominator
        Dvec = self.compute_Dvec(walker)
        self.psi[:,:self.nocca] = numpy.einsum("m,mi->mi",Dvec, psi0a)
        self.psi[:,self.nocca:] = numpy.einsum("m,mi->mi",Dvec, psi0b)
        walker.inverse_overlap(self)
        ot_denom = walker.calc_otrial(self)
        self.psi = psi0.copy()
        
        # Compute numerator
        d2D = self.compute_d2Dvec(walker)

        QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=self.shift)
        QHO_lap = QHO.laplacian(walker.X)

        for i in range (nbsf):
            d2Dvec = Dvec.copy()
            d2Dvec[i] = d2D[i]

            self.psi[:,:self.nocca] = numpy.einsum("m,mi->mi",d2Dvec, psi0a)
            self.psi[:,self.nocca:] = numpy.einsum("m,mi->mi",d2Dvec, psi0b)
            walker.inverse_overlap(self)
            ot_num = walker.calc_otrial(self)

            lap[i] = ot_num / ot_denom * QHO_lap[i]
        
        self.psi = psi0.copy()

        return lap

    def calculate_energy(self, system):
        if self.verbose:
            print ("# Computing trial energy.")
        sqrttwomw = numpy.sqrt(system.m * system.w0*2.0)
        alpha = self.gamma * numpy.sqrt(system.m * system.w0 / 2.0)
        phi = self.shift * numpy.sqrt(system.m * system.w0 / 2.0)

        nia = numpy.diag(self.G[0])
        nib = numpy.diag(self.G[1])
        ni = nia + nib

        Eph = system.w0 * numpy.sum(phi*phi)
        Eeph = (self.gamma * system.w0 - system.g * sqrttwomw) * numpy.sum (2.0 * phi / sqrttwomw * ni)
        # print("Eeph1 = {}".format(Eeph))
        Eeph += (self.gamma**2 * system.w0 / 2.0 - system.g * self.gamma * sqrttwomw) * numpy.sum(ni)
        # print("Eeph2 = {}".format(Eeph))

        Eee = (system.U + self.gamma**2 * system.w0 - 2.0 * system.g * self.gamma * sqrttwomw) * numpy.sum(nia * nib)

        Ekin = numpy.exp (-alpha * alpha) * numpy.sum(system.T[0] * self.G[0] + system.T[1] * self.G[1])
        # Ekin = numpy.sum(system.T[0] * self.G[0] + system.T[1] * self.G[1])
        self.energy = Eph + Eeph + Eee + Ekin
        # print("self.G[1] = {}".format(self.G[1]))
        # print("self.T[1] = {}".format(system.T[1]))

        # print("alpha = {}".format(alpha))
        # print("Eph = {}".format(Eph))
        # print("Eeph = {}".format(Eeph))
        # print("Eee = {}".format(Eee))
        # print("Ekin = {}".format(Ekin))


def unit_test():
    import itertools
    from pauxy.systems.hubbard import Hubbard
    from pauxy.estimators.hubbard import local_energy_hubbard_holstein, local_energy_hubbard_holstein_momentum
    from pauxy.estimators.ci import simple_fci_bose_fermi, simple_fci
    from pauxy.systems.hubbard_holstein import HubbardHolstein
    from pauxy.walkers.single_det import SingleDetWalker
    from pauxy.trial_wavefunction.free_electron import FreeElectron
    from pauxy.trial_wavefunction.harmonic_oscillator import HarmonicOscillator, HarmonicOscillatorMomentum
    import scipy
    import numpy
    import scipy.sparse.linalg
    
    options = {
    "name": "HubbardHolstein",
    "nup": 1,
    "ndown": 1,
    "nx": 2,
    "ny": 1,
    "t": 1.0,
    "U": 400.0,
    "w0": 10.0,
    "lambda": 100.0,
    "lang_firsov":True
    }

    system = HubbardHolstein (options, verbose=True)
        
    lf_driver = LangFirsov(system, False, options, parallel=False, verbose=1)

    walker = SingleDetWalker(options, system, lf_driver)

    G = lf_driver.G.copy()
    # Lap = lf_driver.laplacian(walker)
    boson_trial = HarmonicOscillatorMomentum(m = system.m, w = system.w0, order = 0, shift=lf_driver.shift)
    Lap = boson_trial.laplacian(walker.P)

    etot = local_energy_hubbard_holstein_momentum(system, G, walker.P, Lap)[0]
    print("etot = {}".format(etot))
    
    # boson_trial = HarmonicOscillator(m = system.m, w = system.w0, order = 0, shift=lf_driver.shift)
    # Lap = boson_trial.laplacian(walker.X)
    # etot = local_energy_hubbard_holstein(system, G, walker.X, Lap)[0]

    # print("etot = {}".format(etot))



if __name__=="__main__":
    unit_test()
