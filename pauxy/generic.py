import sys
import numpy
import scipy.linalg
import pauxy.utils


class Generic:
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

    def __init__(self, inputs, dt, verbose):
        if verbose:
            print ("# Parsing input options.")
        self.name = "Generic"
        self.nup = inputs['nup']
        self.ndown = inputs['ndown']
        self.ne = self.nup + self.ndown
        self.integral_file = inputs.get('integrals')
        self.decomopsition = inputs.get('decomposition', 'cholesky')
        self.threshold = inputs.get('threshold', 1e-5)
        if verbose:
            print ("# Reading integrals from %s." % self.integral_file)
        (self.T, self.h2e, self.ecore) = self.read_integrals()
        if verbose:
            print ("# Decomposing two-body operator.")
        (self.h1e_mod, self.chol_vecs) = self.construct_decomposition(verbose)
        self.nchol_vec = self.chol_vecs.shape[0]
        self.nfields = self.nchol_vec
        self.ktwist = numpy.array(inputs.get('ktwist'))
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

        Attributes
        ----------
        h1e_mod : :class:`numpy.ndarray`
            Modified one-body Hamiltonian.
        chol_vecs : :class:`numpy.ndarray`
            Cholesky vectors.
        """
        # Subtract one-body bit following reordering of 2-body operators.
        # Eqn (17) of [Motta17]_
        h1e_mod = self.T[0] - 0.5 * numpy.einsum('ijjl->il', self.h2e)
        h1e_mod = numpy.array([h1e_mod, h1e_mod])
        # Super matrix of v_{ijkl}. V[mu(ik),nu(jl)] = v_{ijkl}.
        V = numpy.transpose(self.h2e, (0, 2, 1, 3)).reshape(self.nbasis**2,
                                                            self.nbasis**2)
        if (numpy.sum(V - V.T) != 0):
            print("Warning: Supermatrix is not symmetric")
        chol_vecs = pauxy.utils.modified_cholesky(V, self.threshold,
                                                  verbose=verbose)
        return (h1e_mod, chol_vecs.reshape((chol_vecs.shape[0], self.nbasis,
                                            self.nbasis)))
