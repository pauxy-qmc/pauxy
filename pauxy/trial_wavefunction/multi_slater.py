import numpy
import scipy.linalg
import time
from pauxy.estimators.mixed import (
        variational_energy, variational_energy_ortho_det
        )
from pauxy.estimators.greens_function import gab, gab_mod, gab_mod_ovlp
from pauxy.estimators.ci import get_hmatel, get_one_body_matel
from pauxy.utils.io import get_input_value

class MultiSlater(object):

    def __init__(self, system, wfn, nbasis=None, options={},
                 init=None, parallel=False, verbose=False, orbs=None):
        self.verbose = verbose
        if verbose:
            print ("# Parsing MultiSlater trial wavefunction input options.")
        init_time = time.time()
        self.name = "MultiSlater"
        # TODO : Fix for MSD.
        self.ortho_expansion = False
        rediag = get_input_value(options, 'recompute_ci',
                                 default=False, alias=['rediag'],
                                 verbose=verbose)
        if len(wfn) == 3:
            # CI type expansion.
            self.from_phmsd(system, wfn, orbs)
            self.ortho_expansion = True
        else:
            self.psi = wfn[1]
            self.coeffs = wfn[0]
        self.ndets = len(self.coeffs)
        if rediag:
            if self.verbose:
                print("# Recomputing CI coefficients.")
            self.recompute_ci_coeffs(system)
        if init is not None:
            self.init = init
        else:
            if len(self.psi.shape) == 3:
                self.init = self.psi[0].copy()
            else:
                self.init = self.psi.copy()
        self.error = False
        self.initialisation_time = time.time() - init_time
        if verbose:
            print ("# Finished setting up trial wavefunction.")

    def calculate_energy(self, system):
        if self.verbose:
            print ("# Computing trial wavefunction energy.")
        start = time.time()
        # Cannot use usual energy evaluation routines if trial is orthogonal.
        if self.ortho_expansion:
            self.energy, self.e1b, self.e2b = (
                    variational_energy_ortho_det(system,
                                                 self.spin_occs,
                                                 self.coeffs)
                    )
        else:
            (self.energy, self.e1b, self.e2b) = (
                    variational_energy(system, self.psi, self.coeffs)
                    )
        if self.verbose:
            print("# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                   %(self.energy.real, self.e1b.real, self.e2b.real))
            print("# Time to evaluate local energy: %f s"%(time.time()-start))

    def from_phmsd(self, system, wfn, orbs):
        ndets = len(wfn[0])
        self.psi = numpy.zeros((ndets,system.nbasis,system.ne),
                                dtype=numpy.complex128)
        if self.verbose:
            print("# Creating trial wavefunction from CI-like expansion.")
        if orbs is None:
            if self.verbose:
                print("# Assuming RHF reference.")
            I = numpy.eye(system.nbasis, dtype=numpy.complex128)
        # Store determinants occupation strings in spin orbital indexing scheme.
        # (alpha,beta,alpha,beta...) for convenience later during energy
        # evaluation.
        soa = [[2*x for x in w] for w in wfn[1]]
        spocc = [alp+[2*x+1 for x in w] for (alp,w) in zip(soa,wfn[2])]
        self.spin_occs = [numpy.sort(numpy.array(x)) for x in spocc]
        self.coeffs = wfn[0]
        for idet, (occa, occb) in enumerate(zip(wfn[1], wfn[2])):
            self.psi[idet,:,:system.nup] = I[:,occa]
            self.psi[idet,:,system.nup:] = I[:,occb]

    def recompute_ci_coeffs(self, system):
        H = numpy.zeros((self.ndets, self.ndets), dtype=numpy.complex128)
        S = numpy.zeros((self.ndets, self.ndets), dtype=numpy.complex128)
        if self.ortho_expansion:
            for i in range(self.ndets):
                for j in range(i,self.ndets):
                    di = self.spin_occs[i]
                    dj = self.spin_occs[j]
                    H[i,j] = get_hmatel(system,di,dj)
            e, ev = scipy.linalg.eigh(H, lower=False)
        else:
            na = system.nup
            for i, di in enumerate(self.psi):
                for j, dj in enumerate(self.psi):
                    if j >= i:
                        ga, ioa = gab_mod_ovlp(di[:,:na], dj[:,:na])
                        gb, iob = gab_mod_ovlp(di[:,na:], dj[:,na:])
                        G = numpy.array([ga,gb])
                        ovlp = 1.0/(scipy.linalg.det(ioa)*scipy.linalg.det(iob))
                        H[i,j] = ovlp * local_energy(system, G, opt=False)[0]
                        S[i,j] = ovlp
            e, ev = scipy.linalg.eigh(H, S, lower=False)
        self.coeffs = ev[:,0]

    def contract_one_body(self, ints):
        numer = 0.0
        denom = 0.0
        for i in range(self.ndets):
            for j in range(self.ndets):
                cfac = self.coeffs[i].conj()*self.coeffs[j].conj()
                if self.ortho_expansion:
                    di = self.spin_occs[i]
                    dj = self.spin_occs[j]
                    tij = get_one_body_matel(ints,di,dj)
                    numer += cfac * tij
                    denom += cfac
                else:
                    ga, ioa = gab_mod_ovlp(di[:,:na], dj[:,:na])
                    gb, iob = gab_mod_ovlp(di[:,na:], dj[:,na:])
                    ovlp = 1.0/(scipy.linalg.det(ioa)*scipy.linalg.det(iob))
                    tij = numpy.dot(ints.ravel(), ga.ravel()+gb.ravel())
                    numer += cfac * ovlp * tij
                    numer += cfac * ovlp
        return numer / denom
