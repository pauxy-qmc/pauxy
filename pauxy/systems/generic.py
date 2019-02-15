import ast
import h5py
import numpy
import sys
import scipy.linalg
import time
from scipy.sparse import csr_matrix
from pauxy.utils.linalg import modified_cholesky
from pauxy.utils.io import from_qmcpack_cholesky
from pauxy.utils.from_pyscf import write_fcidump
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
        self.ncore= inputs.get('nfrozen_core', 0)
        self.nfv = inputs.get('nfrozen_virt', 0)
        self.write = inputs.get('write', False)
        self.frozen_core = self.ncore > 0
        if self.frozen_core:
            self.nup = self.nup - self.ncore
            self.ndown = self.ndown - self.ncore
        self.ne = self.nup + self.ndown
        self.integral_file = inputs.get('integrals')
        self.total_mem = 0
        self.decomopsition = inputs.get('decomposition', 'cholesky')
        self.cutoff = inputs.get('sparse_cutoff', None)
        self.sparse = inputs.get('sparse', True)
        self.threshold = inputs.get('cholesky_threshold', 1e-5)
        self.cplx_chol = inputs.get('complex_cholesky', False)
        if verbose:
            print("# Reading integrals from %s." % self.integral_file)
        self.schol_vecs = None
        self.read_integrals()
        self.nactive = self.nbasis - self.ncore - self.nfv
        if not self.frozen_core:
            if self.chol_vecs is None:
                if verbose:
                    print("# Decomposing two-body operator.")
                init = time.time()
                self.chol_vecs = self.construct_decomposition()
                self.h2e = None
                if verbose:
                    print("# Time to perform Cholesky decomposition: %f s"
                          %(time.time()-init))
                self.nchol_vec = self.chol_vecs.shape[0]
            self.construct_h1e_mod()
            self.nfields = self.nchol_vec
        self.ktwist = numpy.array(inputs.get('ktwist'))
        self.mu = None
        if verbose:
            print("# Finished setting up Generic system object.")

    def read_integrals(self):
        try:
            self.read_qmcpack_integrals()
        except KeyError:
            self.read_hdf5_integrals()
        except OSError:
            self.read_ascii_integrals()

    def read_ascii_integrals(self):
        """Read in integrals from file.

        Returns
        -------
        T : :class:`numpy.ndarray`
            One-body part of the Hamiltonian.
        h2e : :class:`numpy.ndarray`
            Two-electron integrals.
        ecore : float
            Core contribution to the total energy.
        """
        if self.verbose:
            print ("# Reading integrals in plain text FCIDUMP format.")
        f = open(self.integral_file)
        while True:
            line = f.readline()
            if 'END' in line:
                break
            for i in line.split(','):
                if 'NORB' in i:
                    self.nbasis = int(i.split('=')[1])
                elif 'NELEC' in i:
                    nelec = int(i.split('=')[1])
                    if nelec != self.ne:
                        print("Number of electrons is inconsistent")
                        sys.exit()
        self.h1e = numpy.zeros((self.nbasis, self.nbasis))
        self.h2e = numpy.zeros((self.nbasis, self.nbasis,
                                self.nbasis, self.nbasis))
        lines = f.readlines()
        for l in lines:
            s = l.split()
            # ascii fcidump uses chemist's notation for integrals.
            # each line contains v_{ijkl} i k j l
            # Note (ik|jl) = <ij|kl>.
            # Assuming real integrals
            try:
                integral = float(s[0])
            except ValueError:
                ig = ast.literal_eval(s[0].strip())
                # Hack for the moment, not dealing with complex fcidumps, just
                # the format
                integral = ig[0]
            i, k, j, l = [int(x) for x in s[1:]]
            if i == j == k == l == 0:
                self.ecore = integral
            elif j == 0 and l == 0:
                # <i|k> = <k|i>
                self.h1e[i-1,k-1] = integral
                self.h1e[k-1,i-1] = integral
            elif i > 0  and j > 0 and k > 0 and l > 0:
                # <ij|kl> = <ji|lk> = <kl|ij> = <lk|ji> =
                # <kj|il> = <li|jk> = <il|kj> = <jk|li>
                # (ik|jl)
                self.h2e[i-1,k-1,j-1,l-1] = integral
                # (jl|ik)
                self.h2e[j-1,l-1,i-1,k-1] = integral
                # (ki|lj)
                self.h2e[k-1,i-1,l-1,j-1] = integral
                # (lj|ki)
                self.h2e[l-1,j-1,k-1,i-1] = integral
                # (ki|jl)
                self.h2e[k-1,i-1,j-1,l-1] = integral
                # (lj|ik)
                self.h2e[l-1,j-1,i-1,k-1] = integral
                # (ik|lj)
                self.h2e[i-1,k-1,l-1,j-1] = integral
                # (jl|ki)
                self.h2e[j-1,l-1,k-1,i-1] = integral
        self.H1 = numpy.array([self.h1e, self.h1e])
        self.orbs = None
        self.chol_vecs = None

    def read_hdf5_integrals(self):
        """Read in integrals from file.

        Returns
        -------
        T : :class:`numpy.ndarray`
            One-body part of the Hamiltonian.
        h2e : :class:`numpy.ndarray`
            Two-electron integrals.
        ecore : float
            Core contribution to the total energy.
        """
        if self.verbose:
            print ("# Reading integrals in PAUXY HDF5 format.")
        with h5py.File(self.integral_file, 'r') as fh5:
            h1e = fh5['hcore'][:]
            self.nbasis = int(h1e.shape[-1])
            self.h2e = fh5['eri'][:]
            self.ecore = fh5['enuc'][:][0]
            nelec = fh5['nelec'][:]
            self.orbs = fh5['orbs'][:]
            try:
                self.coeffs = fh5['coeffs'][:]
            except KeyError:
                self.coeffs = None
        fc = self.frozen_core
        if (nelec[0] != self.nup or nelec[1] != self.ndown) and not fc:
            print("Number of electrons is inconsistent")
            print("%d %d vs. %d %d"%(nelec[0], nelec[1], self.nup, self.ndown))
            # sys.exit()
        self.H1 = numpy.array([h1e, h1e])

    def read_qmcpack_integrals(self):
        (h1e, self.schol_vecs, self.ecore,
        self.nbasis, nup, ndown) = from_qmcpack_cholesky(self.integral_file)
        if ((nup != self.nup) or ndown != self.ndown) and not self.frozen_core:
            print("Number of electrons is inconsistent")
            print("%d %d vs. %d %d"%(nelec[0], nelec[1], self.nup, self.ndown))
        self.nchol_vec = int(self.schol_vecs.shape[-1])
        self.chol_vecs = self.schol_vecs.toarray().T.reshape((-1, self.nbasis,
                                                         self.nbasis))
        if numpy.max(numpy.abs(self.chol_vecs.imag)) > 1e-6:
            print("# Found complex integrals.")
            self.cplx_chol = True
        else:
            if not self.cplx_chol:
                self.cplx_chol= False
        mem = self.chol_vecs.nbytes / (1024.0**3)
        self.total_mem += mem
        print("# Memory required by Cholesky vectors %f GB"%mem)
        self.H1 = numpy.array([h1e, h1e])
        # These will be reconstructed later.
        self.schol_vecs = None
        self.orbs = None
        self.h2e = None

    def construct_decomposition(self):
        """Decompose two-electron integrals.

        Returns
        -------
        h1e_mod : :class:`numpy.ndarray`
            Modified one-body Hamiltonian.
        chol_vecs : :class:`numpy.ndarray`
            Cholesky vectors.
        """
        # Super matrix of v_{ijkl}. V[mu(ik),nu(jl)] = v_{ijkl}.
        V = self.h2e.reshape((self.nbasis**2, self.nbasis**2))
        if (abs(numpy.sum(V - V.conj().T)) > 1e-12):
            print("Warning: Supermatrix is not Hermitian")
        chol_vecs = modified_cholesky(V, self.threshold, verbose=self.verbose)
        chol_vecs = chol_vecs.reshape((chol_vecs.shape[0],
                                       self.nbasis,
                                       self.nbasis))
        return chol_vecs

    def construct_h1e_mod(self):
        # Subtract one-body bit following reordering of 2-body operators.
        # Eqn (17) of [Motta17]_
        v0 = 0.5 * numpy.einsum('lik,ljk->ij', self.chol_vecs, self.chol_vecs)
        self.h1e_mod = numpy.array([self.H1[0]-v0, self.H1[1]-v0])

    def frozen_core_hamiltonian(self, trial):
        # 1. Construct one-body hamiltonian
        self.ecore = local_energy_generic_cholesky(self, trial.Gcore)[0]
        (hc_a, hc_b) = core_contribution_cholesky(self, trial.Gcore)
        self.T[0] = self.T[0] + 2*hc_a
        self.T[1] = self.T[1] + 2*hc_b
        # 3. Cholesky Decompose ERIs.
        nfv = self.nfv
        nc = self.ncore
        nb = self.nbasis
        # if len(self.orbs.shape) == 3:
            # self.orbs = self.orbs[:,nc:nb-nfv,nc:nb-nfv]
        # else:
            # self.orbs = self.orbs[nc:nb-nfv,nc:nb-nfv]
        self.T = self.T[:,nc:nb-nfv,nc:nb-nfv]
        self.nbasis = self.nbasis - self.ncore - self.nfv
        if self.h2e is not None:
            self.h2e = self.h2e[nc:nb-nfv,nc:nb-nfv,nc:nb-nfv,nc:nb-nfv]
            self.chol_vecs = self.construct_decomposition()
            self.h2e = None
        else:
            self.chol_vecs = self.chol_vecs[:,nc:nb-nfv,nc:nb-nfv]
        self.eactive = local_energy_generic_cholesky(self, trial.G)[0] - self.ecore
        self.nchol_vec = self.chol_vecs.shape[0]
        # 4. Subtract one-body term from writing H2 as sum of squares.
        self.construct_h1e_mod()
        self.nfields = self.nchol_vec
        if self.write:
            write_fcidump(self)
        if self.verbose:
            print("# Freezing core.")
            print("# Freezing %d core states and %d virtuals."
                  %(self.ncore, self.nfv))
            print("# Number of active electrons : (%d, %d)."
                  %(self.nup, self.ndown))
            print("# Number of active virtuals : %d"%self.nbasis)
            print("# Frozen core energy : %13.8e"%self.ecore.real)
            print("# Active space energy : %13.8e"%self.eactive.real)
            print("# Total HF energy : %13.8e"%(self.eactive+self.ecore).real)

    def construct_integral_tensors_real(self, trial):
        if self.schol_vecs is None:
            if self.cutoff is not None:
                self.chol_vecs[numpy.abs(self.chol_vecs) < self.cutoff] = 0
            tmp = numpy.transpose(self.chol_vecs, axes=(1,2,0))
            tmp = tmp.reshape(self.nbasis*self.nbasis, self.nchol_vec)
            self.schol_vecs = csr_matrix(tmp)
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
        rup = rup.reshape((self.nchol_vec, -1))
        rdn = rdn.reshape((self.nchol_vec, -1))
        Ma = numpy.dot(rup.T, rup)
        Mb = numpy.dot(rdn.T, rdn)
        tvaklb = time.time()
        vakbl_a = Ma - Ma.reshape(na,M,na,M).transpose((2,1,0,3)).reshape(na*M,na*M)
        vakbl_b = Mb - Mb.reshape(nb,M,nb,M).transpose((2,1,0,3)).reshape(nb*M,nb*M)
        self.rchol_vecs = [csr_matrix(rup.T.reshape((M*na, -1))),
                           csr_matrix(rdn.T.reshape((M*nb, -1)))]
        self.vaklb = [csr_matrix(vakbl_a.reshape((M*na, M*na))),
                      csr_matrix(vakbl_b.reshape((M*nb, M*nb)))]
        if self.verbose:
            nnz = self.rchol_vecs[0].nnz
            print("# Number of non-zero elements in rotated cholesky: %d"%nnz)
            nelem = self.rchol_vecs[0].shape[0] * self.rchol_vecs[0].shape[1]
            print("# Sparsity: %f"%(nnz/nelem))
            mem = (2*nnz*16/(1024.0**3))
            self.total_mem += mem
            print("# Memory used %f" " GB"%mem)
            nnz = self.vaklb[0].nnz
            print("# Number of non-zero elements in V_{(ak)(bl)}: %d"%nnz)
            mem = (2*nnz*16/(1024.0**3))
            self.total_mem += mem
            print("# Memory used %f GB"%mem)
            nelem = self.vaklb[0].shape[0] * self.vaklb[0].shape[1]
            print("# Sparsity: %f"%(nnz/nelem))

    def construct_integral_tensors_cplx(self, trial):
        # Half rotated cholesky vectors (by trial wavefunction).
        # Assuming nup = ndown here
        M = self.nbasis
        na = self.nup
        nb = self.ndown
        if self.verbose:
            print("# Constructing complex half rotated Cholesky vectors.")
        self.nfields = 2 * self.nfields
        self.sym_chol_vecs = numpy.zeros(shape=(2*self.nchol_vec, M, M),
                                         dtype=numpy.complex128)
        if self.schol_vecs is None:
            if self.cutoff is not None:
                self.sym_chol_vecs[numpy.abs(self.sym_chol_vecs) < self.cutoff] = 0
            tmp = numpy.transpose(self.sym_chol_vecs, axes=(1,2,0))
            tmp = tmp.reshape(self.nbasis*self.nbasis, 2*self.nchol_vec)
            self.schol_vecs = csr_matrix(tmp)
        rup = numpy.zeros(shape=(2*self.nchol_vec, na, M),
                          dtype=numpy.complex128)
        rdn = numpy.zeros(shape=(2*self.nchol_vec, nb, M),
                          dtype=numpy.complex128)
        # rup = numpy.einsum('ia,lik->lak',
                           # trial.psi[:,:na].conj(),
                           # self.chol_vecs)
        # rdn = numpy.einsum('ia,lik->lak',
                           # trial.psi[:,na:].conj(),
                           # self.chol_vecs)
        # This is much faster than einsum.
        for (n,cn) in enumerate(self.chol_vecs):
            vplus = 0.5*(cn+cn.conj().T)
            vminus = 0.5j*(cn-cn.conj().T)
            self.sym_chol_vecs[n] = vplus
            self.sym_chol_vecs[self.nchol_vec+n] = vminus
            rup[n] = numpy.dot(trial.psi[:,:na].conj().T, vplus)
            rup[self.nchol_vec+n] = numpy.dot(trial.psi[:,:na].conj().T, vminus)
            rdn[n] = numpy.dot(trial.psi[:,na:].conj().T, vplus)
            rdn[self.nchol_vec+n] = numpy.dot(trial.psi[:,na:].conj().T, vminus)
        if self.verbose:
            print("# Constructing half rotated V_{(ab)(kl)}.")
        # vaklb_a = (numpy.einsum('gak,gbl->akbl', rup, rup) -
                   # numpy.einsum('gbk,gal->akbl', rup, rup))
        # vaklb_b = (numpy.einsum('gak,gbl->akbl', rdn, rdn) -
                   # numpy.einsum('gbk,gal->akbl', rdn, rdn))
        # This is also much faster than einsum.
        self.rchol_vecs = [csr_matrix(rup.reshape((-1,M*na)).T),
                           csr_matrix(rdn.reshape((-1,M*nb)).T)]
        # vaklb_a = numpy.zeros((M*na,M*na), dtype=numpy.complex128)
        # vaklb_b = numpy.zeros((M*nb,M*nb), dtype=numpy.complex128)
        # tmp_1 = numpy.zeros((self.nchol_vecs, M*na, M*na),
                            # dtype=numpy.complex128)
        # tmp_2 = numpy.zeros((self.nchol_vecs, M*na, M*na),
                            # dtype=numpy.complex128)
        # for (n,cn) in enumerate(self.chol_vecs):
            # tmp_1[n] = numpy.dot(trial.psi[:,:na].conj().T, cn)
            # tmp_2[n] = numpy.dot(cn.conj().T, trial.psi[:,:na].conj().T)
        # self.vaklb = [csr_matrix(vakbl_a.reshape((M*na, M*na))),
                      # csr_matrix(vakbl_b.reshape((M*nb, M*nb)))]
        # if self.verbose:
            # nnz = self.rchol_vecs[0].nnz
            # print("# Number of non-zero elements in rotated cholesky: %d"%nnz)
            # nelem = self.rchol_vecs[0].shape[0] * self.rchol_vecs[0].shape[1]
            # print("# Sparsity: %f"%(nnz/nelem))
            # mem = (2*nnz*16/(1024.0**3))
            # self.total_mem += mem
            # print("# Memory used %f" " GB"%mem)
            # nnz = self.vaklb[0].nnz
            # print("# Number of non-zero elements in V_{(ak)(bl)}: %d"%nnz)
            # mem = (2*nnz*16/(1024.0**3))
            # self.total_mem += mem
            # print("# Memory used %f GB"%mem)
            # nelem = self.vaklb[0].shape[0] * self.vaklb[0].shape[1]
            # print("# Sparsity: %f"%(nnz/nelem))
