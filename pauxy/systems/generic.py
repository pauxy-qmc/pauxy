import ast
import h5py
import numpy
import sys
import scipy.linalg
import time
from scipy.sparse import csr_matrix
from pauxy.utils.linalg import modified_cholesky
from pauxy.utils.io import from_qmcpack_cholesky
from pauxy.estimators.generic import (
        local_energy_generic, core_contribution,
        local_energy_generic_cholesky, core_contribution_cholesky
)
from pauxy.estimators.mixed import local_energy_multi_det_full


class Generic(object):
    """Generic system class (integrals read from fcidump)

    Parameters
    ----------
    nup : int
        Number of up electrons.
    ndown : int
        Number of down electrons.
    integrals : string
        Path to FCIDUMP containing one- and two-electron integrals.
    decomposition : string
        Method by which to decompose two-electron integrals. Options:

            - cholesky: Use cholesky decomposition. Default.
            - eigenvalue: Use eigenvalue decomposition. Not implemented.

    threshold : float
        Cutoff for cholesky decomposition or minimum eigenvalue.
    verbose : bool
        Print extra information.

    Attributes
    ----------
    T : :class:`numpy.ndarray`
        One-body part of the Hamiltonian.
    h2e : :class:`numpy.ndarray`
        Two-electron integrals.
    ecore : float
        Core contribution to the total energy.
    h1e_mod : :class:`numpy.ndarray`
        Modified one-body Hamiltonian.
    chol_vecs : :class:`numpy.ndarray`
        Cholesky vectors.
    nchol_vec : int
        Number of cholesky vectors.
    nfields : int
        Number of field configurations per walker for back propagation.
    """

    def __init__(self, inputs, verbose=False):
        if verbose:
            print("# Parsing input options.")
        self.name = "Generic"
        self.atom = inputs.get('atom', None)
        self.verbose = verbose
        self.nup = inputs['nup']
        self.ndown = inputs['ndown']
        self.ne = self.nup + self.ndown
        self.integral_file = inputs.get('integrals')
        self.cutoff = inputs.get('sparse_cutoff', None)
        self.sparse = inputs.get('sparse', True)
        self.threshold = inputs.get('cholesky_threshold', 1e-5)
        self.cplx_chol = inputs.get('complex_cholesky', False)
        if verbose:
            print("# Reading integrals from %s." % self.integral_file)
        self.chol_vecs = self.read_integrals()
        self.nchol = self.chol_vecs.shape[0]
        self.construct_h1e_mod()
        self.ktwist = numpy.array([None])
        start = time.time()
        if self.cplx_chol:
            self.nfields = 2 * self.nchol_vec
            self.hs_pot = numpy.zeros(shape=(self.nfields,self.nbasis,self.nbasis),
                                      dtype=numpy.complex128)
            for (n,cn) in enumerate(self.chol_vecs):
                vplus = 0.5*(cn+cn.conj().T)
                vminus = 0.5j*(cn-cn.conj().T)
                self.hs_pot[n] = vplus
                self.hs_pot[self.nchol_vec+n] = vminus
        else:
            self.hs_pot = self.chol_vecs
            self.nfields = self.nchol_vec
        if verbose:
            print("# Number of Cholesky vectors: %d"%(self.nchol))
            print("# Number of fields: %d"%(self.nfields))
            print("# Time to construct Hubbard--Stratonovich potentials: "
                  "%f s"%(time.time()-start))
        self.mu = None
        if self.sparse:
            if self.cutoff is not None:
                self.hs_pot[numpy.abs(self.hs_pot) < self.cutoff] = 0
            tmp = numpy.transpose(self.hs_pot, axes=(1,2,0))
            tmp = tmp.reshape(self.nbasis*self.nbasis, self.nfields)
            self.hs_pot = csr_matrix(tmp)
        if verbose:
            print("# Finished setting up Generic system object.")

    def read_integrals(self):
        (h1e, schol_vecs, self.ecore,
        self.nbasis, nup, ndown) = from_qmcpack_cholesky(self.integral_file)
        if ((nup != self.nup) or ndown != self.ndown):
            print("Number of electrons is inconsistent")
            print("%d %d vs. %d %d"%(nup, ndown, self.nup, self.ndown))
        self.nchol_vec = int(schol_vecs.shape[-1])
        chol_vecs = schol_vecs.toarray().T.reshape((-1,self.nbasis,self.nbasis))
        if numpy.max(numpy.abs(chol_vecs.imag)) > 1e-6:
            print("# Found complex integrals.")
            self.cplx_chol = True
        else:
            if self.cplx_chol:
                print("# Using Hermitian Cholesky decomposition.")
            else:
                print("# Using real symmetric Cholesky decomposition.")
                self.cplx_chol= False
        mem = chol_vecs.nbytes / (1024.0**3)
        print("# Approximate memory required by Cholesky vectors %f GB"%mem)
        self.H1 = numpy.array([h1e, h1e])
        # These will be reconstructed later.
        self.orbs = None
        self.h2e = None
        return chol_vecs

    def construct_h1e_mod(self):
        # Subtract one-body bit following reordering of 2-body operators.
        # Eqn (17) of [Motta17]_
        v0 = 0.5 * numpy.einsum('lik,ljk->ij', self.chol_vecs,
                                self.chol_vecs.conj())
        self.h1e_mod = numpy.array([self.H1[0]-v0, self.H1[1]-v0])


    def construct_integral_tensors_real(self, trial):
        # Half rotated cholesky vectors (by trial wavefunction).
        # Assuming nup = ndown here
        M = self.nbasis
        na = self.nup
        nb = self.ndown
        if self.verbose:
            print("# Constructing half rotated Cholesky vectors.")
        rup = numpy.zeros(shape=(self.nchol_vec, na, M),
                          dtype=numpy.complex128)
        rdn = numpy.zeros(shape=(self.nchol_vec, nb, M),
                          dtype=numpy.complex128)
        # rup = numpy.einsum('ia,lik->lak',
                           # trial.psi[:,:na].conj(),
                           # self.chol_vecs)
        # rdn = numpy.einsum('ia,lik->lak',
                           # trial.psi[:,na:].conj(),
                           # self.chol_vecs)
        # This is much faster than einsum.
        if self.sparse:
            self.hs_pot = self.hs_pot.toarray().reshape(M,M,self.nfields)
            self.hs_pot = self.hs_pot.transpose(2,0,1)
        for l in range(self.nchol_vec):
            rup[l] = numpy.dot(trial.psi[:,:na].conj().T, self.chol_vecs[l])
            rdn[l] = numpy.dot(trial.psi[:,na:].conj().T, self.chol_vecs[l])
        if self.verbose:
            print("# Constructing half rotated V_{(ab)(kl)}.")
        # vaklb_a = (numpy.einsum('gak,gbl->akbl', rup, rup) -
                   # numpy.einsum('gbk,gal->akbl', rup, rup))
        # vaklb_b = (numpy.einsum('gak,gbl->akbl', rdn, rdn) -
                   # numpy.einsum('gbk,gal->akbl', rdn, rdn))
        # This is also much faster than einsum.
        start = time.time()
        rup = rup.reshape((self.nchol_vec, -1))
        rdn = rdn.reshape((self.nchol_vec, -1))
        Ma = numpy.dot(rup.T, rup)
        Mb = numpy.dot(rdn.T, rdn)
        vakbl_a = Ma - Ma.reshape(na,M,na,M).transpose((2,1,0,3)).reshape(na*M,na*M)
        vakbl_b = Mb - Mb.reshape(nb,M,nb,M).transpose((2,1,0,3)).reshape(nb*M,nb*M)
        tvakbl = time.time() - start
        self.rot_hs_pot = [csr_matrix(rup.T.reshape((M*na, -1))),
                           csr_matrix(rdn.T.reshape((M*nb, -1)))]
        self.rchol_vecs = self.rot_hs_pot
        self.vakbl = [csr_matrix(vakbl_a.reshape((M*na, M*na))),
                      csr_matrix(vakbl_b.reshape((M*nb, M*nb)))]
        if self.sparse:
            if self.cutoff is not None:
                self.hs_pot[numpy.abs(self.hs_pot) < self.cutoff] = 0
            tmp = numpy.transpose(self.hs_pot, axes=(1,2,0))
            tmp = tmp.reshape(M*M, self.nfields)
            self.hs_pot = csr_matrix(tmp)
        if self.verbose:
            print("# Time to construct V_{(ak)(bl)}: %f"%tvakbl)
            nnz = self.rchol_vecs[0].nnz
            print("# Number of non-zero elements in rotated cholesky: %d"%nnz)
            nelem = self.rchol_vecs[0].shape[0] * self.rchol_vecs[0].shape[1]
            print("# Sparsity: %f"%(1-float(nnz)/nelem))
            mem = (2*nnz*16/(1024.0**3))
            print("# Approximate memory used %f" " GB"%mem)
            nnz = self.vakbl[0].nnz
            print("# Number of non-zero elements in V_{(ak)(bl)}: %d"%nnz)
            mem = (2*nnz*16/(1024.0**3))
            print("# Approximate memory used %f GB"%mem)
            nelem = self.vakbl[0].shape[0] * self.vakbl[0].shape[1]
            print("# Sparsity: %f"%(1-float(nnz)/nelem))

    def construct_integral_tensors_cplx(self, trial):
        # Half rotated cholesky vectors (by trial wavefunction).
        # Assuming nup = ndown here
        M = self.nbasis
        na = self.nup
        nb = self.ndown
        if self.verbose:
            print("# Constructing complex half rotated HS Potentials.")
        rup = numpy.zeros(shape=(self.nfields, na, M),
                          dtype=numpy.complex128)
        rdn = numpy.zeros(shape=(self.nfields, nb, M),
                          dtype=numpy.complex128)
        # rup = numpy.einsum('ia,lik->lak',
                           # trial.psi[:,:na].conj(),
                           # self.hs_pot)
        # rdn = numpy.einsum('ia,lik->lak',
                           # trial.psi[:,na:].conj(),
                           # self.hs_pot)
        # This is much faster than einsum.
        start = time.time()
        if self.sparse:
            self.hs_pot = self.hs_pot.toarray().reshape(M,M,self.nfields)
            self.hs_pot = self.hs_pot.transpose(2,0,1)
        for (n,cn) in enumerate(self.hs_pot):
            rup[n] = numpy.dot(trial.psi[:,:na].conj().T, self.hs_pot[n])
            rdn[n] = numpy.dot(trial.psi[:,na:].conj().T, self.hs_pot[n])
        self.rot_hs_pot = [csr_matrix(rup.reshape((-1,M*na)).T),
                           csr_matrix(rdn.reshape((-1,M*nb)).T)]
        if self.verbose:
            print("# Time to construct half-rotated HS potentials: "
                  "%f s"%(time.time()-start))
            nnz = self.rot_hs_pot[0].nnz
            print("# Number of non-zero elements in rotated potentials: %d"%nnz)
            nelem = self.rot_hs_pot[0].shape[0] * self.rot_hs_pot[0].shape[1]
            print("# Sparsity: %f"%(1-float(nnz)/nelem))
            mem = (2*nnz*16/(1024.0**3))
            print("# Approximate memory required %f" " GB"%mem)
            print("# Constructing half rotated V_{(ab)(kl)}.")
        # This is also much faster than einsum.
        Qak = numpy.zeros((self.nchol, M*na), dtype=numpy.complex128)
        Rbl = numpy.zeros((self.nchol, M*na), dtype=numpy.complex128)
        start = time.time()
        for (n,cn) in enumerate(self.chol_vecs):
            Qak[n] = numpy.dot(trial.psi[:,:na].conj().T, cn).ravel()
            Rbl[n] = numpy.dot(trial.psi[:,:na].conj().T, cn.conj()).ravel()
        if self.verbose:
            print("# Time to construct Qak, Rbl: %f s"%(time.time()-start))
        Makbl = numpy.dot(Qak.T,Rbl)
        vakbl_a = (
            Makbl -
            Makbl.reshape(na,M,na,M).transpose((2,1,0,3)).reshape(na*M,na*M)
        )
        Qak = numpy.zeros((self.nchol, M*nb), dtype=numpy.complex128)
        Rbl = numpy.zeros((self.nchol, M*nb), dtype=numpy.complex128)
        for (n,cn) in enumerate(self.chol_vecs):
            Qak[n] = numpy.dot(trial.psi[:,na:].conj().T, cn).ravel()
            Rbl[n] = numpy.dot(trial.psi[:,na:].conj().T, cn.conj()).ravel()
        Makbl = numpy.dot(Qak.T,Rbl)
        vakbl_b = (
            Makbl -
            Makbl.reshape(nb,M,nb,M).transpose((2,1,0,3)).reshape(nb*M,nb*M)
        )
        self.vakbl = [csr_matrix(vakbl_a.reshape((M*na, M*na))),
                      csr_matrix(vakbl_b.reshape((M*nb, M*nb)))]
        tvakbl = time.time() - start
        # TODO: Stop converting hs pot to dense
        if self.sparse:
            if self.cutoff is not None:
                self.hs_pot[numpy.abs(self.hs_pot) < self.cutoff] = 0
            tmp = numpy.transpose(self.hs_pot, axes=(1,2,0))
            tmp = tmp.reshape(self.nbasis*self.nbasis, self.nfields)
            self.hs_pot = csr_matrix(tmp)
        if self.verbose:
            print("# Time to construct V_{(ak)(bl)}: %f s"%(tvakbl))
            nnz = self.vakbl[0].nnz
            print("# Number of non-zero elements in V_{(ak)(bl)}: %d"%nnz)
            print("# Approximate memory used %f GB"%mem)
            nelem = self.vakbl[0].shape[0] * self.vakbl[0].shape[1]
            print("# Sparsity: %f"%(1-float(nnz)/nelem))
