import numpy
import sys
import scipy.linalg
import time
from scipy.sparse import csr_matrix
from pauxy.utils.linalg import modified_cholesky

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

    def __init__(self, inputs, dt, verbose=False):
        if verbose:
            print ("# Parsing input options.")
        self.name = "Generic"
        self.verbose = verbose
        self.nup = inputs['nup']
        self.ndown = inputs['ndown']
        self.ne = self.nup + self.ndown
        self.integral_file = inputs.get('integrals')
        self.decomopsition = inputs.get('decomposition', 'cholesky')
        self.cutoff = inputs.get('sparse_cutoff', None)
        self.sparse = inputs.get('sparse', True)
        self.threshold = inputs.get('cholesky_threshold', 1e-5)
        if verbose:
            print ("# Reading integrals from %s." % self.integral_file)
        (self.T, self.h2e, self.ecore) = self.read_integrals()
        if verbose:
            print ("# Decomposing two-body operator.")
        init = time.time()
        (self.h1e_mod, self.chol_vecs) = self.construct_decomposition(verbose)
        if verbose:
            print ("# Time to perform Cholesky decomposition: %f s"%(time.time()-init))
        self.nchol_vec = self.chol_vecs.shape[0]
        if self.cutoff is not None:
            self.chol_vecs[numpy.abs(self.chol_vecs) < self.cutoff] = 0
        tmp = numpy.transpose(self.chol_vecs, axes=(1,2,0))
        tmp = tmp.reshape(self.nbasis*self.nbasis, self.nchol_vec)
        self.schol_vecs = csr_matrix(tmp)
        self.nfields = self.nchol_vec
        self.ktwist = numpy.array(inputs.get('ktwist'))
        self.mu = None
        if verbose:
            print ("# Finished setting up Generic system object.")

    def read_integrals(self):
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
        h1e = numpy.zeros((self.nbasis, self.nbasis))
        h2e = numpy.zeros((self.nbasis, self.nbasis, self.nbasis, self.nbasis))
        lines = f.readlines()
        for l in lines:
            s = l.split()
            # ascii fcidump uses chemist's notation for integrals.
            # each line contains v_{ijkl} i k j l
            # Note (ik|jl) = <ij|kl>.
            # Assuming real integrals
            integral = float(s[0])
            i, k, j, l = [int(x) for x in s[1:]]
            if i == j == k == l == 0:
                ecore = integral
            elif j == 0 and l == 0:
                # <i|k> = <k|i>
                h1e[i-1,k-1] = integral
                h1e[k-1,i-1] = integral
            elif i > 0  and j > 0 and k > 0 and l > 0:
                # <ij|kl> = <ji|lk> = <kl|ij> = <lk|ji> =
                # <kj|il> = <li|jk> = <il|kj> = <jk|li>
                h2e[i-1,j-1,k-1,l-1] = integral
                h2e[j-1,i-1,l-1,k-1] = integral
                h2e[k-1,l-1,i-1,j-1] = integral
                h2e[l-1,k-1,j-1,i-1] = integral
                h2e[k-1,j-1,i-1,l-1] = integral
                h2e[l-1,i-1,j-1,k-1] = integral
                h2e[i-1,l-1,k-1,j-1] = integral
                h2e[j-1,k-1,l-1,i-1] = integral

        return (numpy.array([h1e, h1e]), h2e, ecore)

    def construct_decomposition(self, verbose):
        """Decompose two-electron integrals.

        Returns
        -------
        h1e_mod : :class:`numpy.ndarray`
            Modified one-body Hamiltonian.
        chol_vecs : :class:`numpy.ndarray`
            Cholesky vectors.
        """
        # Super matrix of v_{ijkl}. V[mu(ik),nu(jl)] = v_{ijkl}.
        V = numpy.transpose(self.h2e, (0, 2, 1, 3)).reshape(self.nbasis**2,
                                                            self.nbasis**2)
        if (numpy.sum(V - V.T) != 0):
            print("Warning: Supermatrix is not symmetric")
        chol_vecs = modified_cholesky(V, self.threshold, verbose=verbose)
        chol_vecs = chol_vecs.reshape((chol_vecs.shape[0], self.nbasis, self.nbasis))
        # Subtract one-body bit following reordering of 2-body operators.
        # Eqn (17) of [Motta17]_
        h1e_mod = self.T[0] - 0.5 * numpy.einsum('lik,ljk->ij', chol_vecs, chol_vecs)
        h1e_mod = numpy.array([h1e_mod, h1e_mod])
        return (h1e_mod, chol_vecs)

    def construct_integral_tensors(self, trial):
        # Half rotated cholesky vectors (by trial wavefunction).
        # Assuming nup = ndown here
        M = self.nbasis
        na = self.nup
        nb = self.ndown
        if self.verbose:
            print ("# Constructing half rotated Cholesky vectors.")
        rotated_up = numpy.einsum('ia,lik->akl', trial.psi[:,:na].conj(),
                                  self.chol_vecs)
        rotated_down = numpy.einsum('ia,lik->akl', trial.psi[:,na:].conj(),
                                    self.chol_vecs)
        # Todo: Fix for complex Cholesky
        if self.verbose:
            print ("# Constructing half rotated V_{(ab)(kl)}.")
        vaklb_alpha = (numpy.einsum('akg,blg->akbl', rotated_up, rotated_up) -
                       numpy.einsum('bkg,alg->akbl', rotated_up, rotated_up))
        vaklb_beta = (numpy.einsum('akg,blg->akbl', rotated_down, rotated_down) -
                      numpy.einsum('bkg,alg->akbl', rotated_down, rotated_down))
        self.rchol_vecs = [csr_matrix(rotated_up.reshape((M*na, -1))),
                           csr_matrix(rotated_down.reshape((M*nb, -1)))]
        vaklb_alpha = (numpy.einsum('akg,blg->akbl', rotated_up, rotated_up) -
                       numpy.einsum('bkg,alg->akbl', rotated_up, rotated_up))
        vaklb_beta = (numpy.einsum('akg,blg->akbl', rotated_down, rotated_down) -
                      numpy.einsum('bkg,alg->akbl', rotated_down, rotated_down))
        self.rchol_vecs = [csr_matrix(rotated_up.reshape((M*na, -1))),
                           csr_matrix(rotated_down.reshape((M*nb, -1)))]
        self.vaklb = [csr_matrix(vaklb_alpha.reshape((M*na, M*na))),
                      csr_matrix(vaklb_beta.reshape((M*nb, M*nb)))]
        if self.verbose:
            nnz = self.rchol_vecs[0].nnz
            print ("# Number of non-zero elements in rotated cholesky: %d"%nnz)
            nelem = self.rchol_vecs[0].shape[0] * self.rchol_vecs[0].shape[1]
            print ("# Sparsity: %f"%(nnz/nelem))
            nnz = self.vaklb[0].nnz
            print ("# Number of non-zero elements in V_{(ak)(bl)}: %d"%nnz)
            nelem = self.vaklb[0].shape[0] * self.rchol_vecs[0].shape[1]
            print ("# Sparsity: %f"%(nnz/nelem))
