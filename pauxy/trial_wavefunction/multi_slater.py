import numpy
import scipy.linalg
import time
from pauxy.estimators.mixed import (
        variational_energy, variational_energy_ortho_det, local_energy
        )
from pauxy.estimators.greens_function import gab, gab_spin, gab_mod, gab_mod_ovlp
from pauxy.estimators.ci import get_hmatel, get_one_body_matel
from pauxy.utils.io import (
        get_input_value,
        write_qmcpack_wfn
        )

class MultiSlater(object):

    def __init__(self, system, wfn, nbasis=None, options={},
                 init=None, parallel=False, verbose=False, orbs=None):
        self.verbose = verbose
        if verbose:
            print ("# Parsing MultiSlater trial wavefunction input options.")
        init_time = time.time()
        self.name = "MultiSlater"
        self.type = "MultiSlater"
        # TODO : Fix for MSD.
        rediag = get_input_value(options, 'recompute_ci',
                                 default=False, alias=['rediag'],
                                 verbose=verbose)
        self.half_rot = get_input_value(options, 'half_rotate',
                                        default=False, alias=['rotate'],
                                        verbose=verbose)
        if len(wfn) == 3:
            # CI type expansion.
            self.from_phmsd(system, wfn, orbs)
            self.ortho_expansion = True
        else:
            self.psi = wfn[1]
            self.coeffs = numpy.array(wfn[0], dtype=numpy.complex128)
            self.ortho_expansion = False
        if self.verbose:
            if self.ortho_expansion:
                print("# Assuming orthogonal trial wavefunction expansion.")
            else:
                print("# Assuming non-orthogonal trial wavefunction expansion.")
            print("# Trial wavefunction shape: {}".format(self.psi.shape))
        self.ndets = len(self.coeffs)
        if self.ndets == 1:
            self.psi = self.psi[0]
            self.G, self.GH = gab_spin(self.psi, self.psi,
                                       system.nup, system.ndown)
        else:
            self.G = None
            self.GH = None
        if self.half_rot:
            self.half_rotate(system)
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
        self._nalpha = system.nup
        self._nelec = system.nelec
        self._nbasis = system.nbasis
        write_wfn = options.get('write_wavefunction', False)
        if write_wfn:
            self.write_wavefunction()
        if verbose:
            print ("# Finished setting up trial wavefunction.")

    def calculate_energy(self, system):
        if self.verbose:
            print("# Computing trial wavefunction energy.")
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
                    variational_energy(system, self.psi, self.coeffs,
                                       G=self.G, GH=self.GH)
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
        # Store alpha electrons first followed by beta electrons.
        nb = system.nbasis
        dets = [list(a) + [i+nb for i in c] for (a,c) in zip(wfn[1],wfn[2])]
        self.spin_occs = [numpy.sort(d) for d in dets]
        self.occa = wfn[1]
        self.occb = wfn[2]
        self.coeffs = numpy.array(wfn[0], dtype=numpy.complex128)
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
                    H[i,j] = get_hmatel(system,di,dj)[0]
            e, ev = scipy.linalg.eigh(H, lower=False)
        else:
            na = system.nup
            for i, di in enumerate(self.psi):
                for j, dj in enumerate(self.psi):
                    if j >= i:
                        ga, gha, ioa = gab_mod_ovlp(di[:,:na], dj[:,:na])
                        gb, ghb, iob = gab_mod_ovlp(di[:,na:], dj[:,na:])
                        G = numpy.array([ga,gb])
                        Ghalf = numpy.array([gha,ghb])
                        ovlp = 1.0/(scipy.linalg.det(ioa)*scipy.linalg.det(iob))
                        if abs(ovlp) > 1e-12:
                            H[i,j] = ovlp * local_energy(system, G, Ghalf=Ghalf,
                                                         rchol=self.rot_chol[i])[0]
                            S[i,j] = ovlp
                            H[j,i] = numpy.conjugate(H[i,j])
                            S[j,i] = numpy.conjugate(S[i,j])
            e, ev = scipy.linalg.eigh(H, S, lower=False)
        if self.verbose > 1:
            print("Old and New CI coefficients: ")
            for co,cn in zip(self.coeffs,ev[:,0]):
                print("{} {}".format(co, cn))
        self.coeffs = numpy.array(ev[:,0], dtype=numpy.complex128)

    def contract_one_body(self, ints):
        numer = 0.0
        denom = 0.0
        na = self._nalpha
        for i in range(self.ndets):
            for j in range(self.ndets):
                cfac = self.coeffs[i].conj()*self.coeffs[j].conj()
                if self.ortho_expansion:
                    di = self.spin_occs[i]
                    dj = self.spin_occs[j]
                    tij = get_one_body_matel(ints,di,dj)
                    numer += cfac * tij
                    if i == j:
                        denom += self.coeffs[i].conj()*self.coeffs[i].conj()
                else:
                    di = self.psi[i]
                    dj = self.psi[j]
                    ga, ioa = gab_mod_ovlp(di[:,:na], dj[:,:na])
                    gb, iob = gab_mod_ovlp(di[:,na:], dj[:,na:])
                    ovlp = 1.0/(scipy.linalg.det(ioa)*scipy.linalg.det(iob))
                    tij = numpy.dot(ints.ravel(), ga.ravel()+gb.ravel())
                    numer += cfac * ovlp * tij
                    denom += cfac * ovlp
        return numer / denom

    def write_wavefunction(self, filename='wfn.h5', init=None, occs=False):
        if occs:
            wfn = (self.coeffs, self.occa, self.occb)
        else:
            wfn = (self.coeffs, self.psi)
        write_qmcpack_wfn(filename, wfn, 'uhf', self._nelec, self._nbasis,
                          init=init)

    def half_rotate(self, system):
        # Half rotated cholesky vectors (by trial wavefunction).
        M = system.nbasis
        na = system.nup
        nb = system.ndown
        if self.verbose:
            print("# Constructing half rotated Cholesky vectors.")

        chol = system.chol_vecs.reshape((-1,M*M)).T.reshape((M,M,-1))
        start = time.time()
        self.rot_chol = []
        for i, psi in enumerate(self.psi):
            start = time.time()
            if self.verbose:
                print("# Rotating Cholesky for determinant {}".format(i))
            rup = numpy.tensordot(psi[:,:na].conj(),
                                  chol,
                                  axes=((0),(0)))
            rdn = numpy.tensordot(psi[:,na:].conj(),
                                  chol,
                                  axes=((0),(0)))
            self.rot_chol.append([rup.reshape(M*na,-1), rdn.reshape((M*nb,-1))])
            if self.verbose:
                print("# Time to half rotate {}".format(time.time()-start))
