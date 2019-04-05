import sys
import numpy
import scipy.linalg
import scipy.sparse
import pauxy.utils
import math
import time


class UEG(object):
    """UEG system class (integrals read from fcidump)
    Parameters
    ----------
    nup : int
        Number of up electrons.
    ndown : int
        Number of down electrons.
    rs : float
        Density parameter.
    ecut : float
        Scaled cutoff energy.
    ktwist : :class:`numpy.ndarray`
        Twist vector.
    verbose : bool
        Print extra information.
    Attributes
    ----------
    T : :class:`numpy.ndarray`
        One-body part of the Hamiltonian. This is diagonal in plane wave basis.
    ecore : float
        Madelung contribution to the total energy.
    h1e_mod : :class:`numpy.ndarray`
        Modified one-body Hamiltonian.
    nfields : int
        Number of field configurations per walker for back propagation.
    basis : :class:`numpy.ndarray`
        Basis vectors within a cutoff.
    kfac : float
        Scale factor (2pi/L).
    """

    def __init__(self, inputs, verbose=False):
        if verbose:
            print ("# Parsing input options.")
        self.name = "UEG"
        self.nup = inputs.get('nup')
        self.ndown = inputs.get('ndown')
        self.rs = inputs.get('rs')
        self.ecut = inputs.get('ecut')
        self.ktwist = numpy.array(inputs.get('ktwist', [0,0,0])).reshape(3)
        self.mu = inputs.get('mu', None)
        # if(verbose):
        print("# Number of spin-up electrons: %i"%self.nup)
        print("# Number of spin-down electrons: %i"%self.ndown)
        print("# rs: %10.5f"%self.rs)

        self.thermal = inputs.get('thermal', False)

        # total # of electrons
        self.ne = self.nup + self.ndown
        # core energy
        self.ecore = 0.5 * self.ne * self.madelung()
        # spin polarisation
        self.zeta = (self.nup - self.ndown) / self.ne
        # Density.
        self.rho = ((4.0*math.pi)/3.0*self.rs**3.0)**(-1.0)
        # Box Length.
        self.L = self.rs*(4.0*self.ne*math.pi/3.)**(1/3.)
        # Volume
        self.vol = self.L**3.0
        # k-space grid spacing.
        # self.kfac = 2*math.pi/self.L
        self.kfac = 2*math.pi/self.L
        # Fermi Wavevector (infinite system).
        self.kf = (3*(self.zeta+1)*math.pi**2*self.ne/self.L**3)**(1/3.)
        # Fermi energy (inifinite systems).
        self.ef = 0.5*self.kf**2

        if verbose:
            print("# zeta: %d"%self.zeta)
            print("# rho: %13.8e"%self.rho)
            print("# L: %13.8e"%self.L)
            print("# vol: %13.8e"%self.vol)
            print("# kfac: %13.8e"%self.kfac)
            print("# E_M: %13.8e"%self.ecore)

        # Single particle eigenvalues and corresponding kvectors
        (self.sp_eigv, self.basis, self.nmax) = self.sp_energies(self.kfac, self.ecut)

        self.shifted_nmax = 2*self.nmax
        self.imax_sq = numpy.dot(self.basis[-1], self.basis[-1])
        self.create_lookup_table()
        for (i, k) in enumerate(self.basis):
            assert(i==self.lookup_basis(k))

        # Number of plane waves.
        self.nbasis = len(self.sp_eigv)
        self.nactive = self.nbasis
        self.ncore = 0
        self.nfv = 0
        self.mo_coeff = None
        # if(verbose):
        print("# Number of plane waves: %i"%self.nbasis)
        # Allowed momentum transfers (4*ecut)
        (eigs, qvecs, self.qnmax) = self.sp_energies(self.kfac, 4*self.ecut)
        # Omit Q = 0 term.
        self.qvecs = numpy.copy(qvecs[1:])
        self.vqvec = numpy.array([self.vq(self.kfac*q) for q in self.qvecs])
        # Number of momentum transfer vectors / auxiliary fields.
        # Can reduce by symmetry but be stupid for the moment.
        self.nchol = len(self.qvecs)
        self.nfields = 2*len(self.qvecs)
        # For consistency with frozen core molecular code.
        self.orbs = None
        self.frozen_core = False
        T = numpy.diag(self.sp_eigv)
        self.H1 = numpy.array([T, T]) # Making alpha and beta
        h1e_mod = self.mod_one_body(T)
        self.h1e_mod = numpy.array([h1e_mod, h1e_mod])
        self.orbs = None


        nlimit = self.nup

        if (self.thermal):
            nlimit = self.nbasis

        self.ikpq_i = []
        self.ikpq_kpq = []
        for (iq, q) in enumerate(self.qvecs):
            idxkpq_list_i =[]
            idxkpq_list_kpq =[]
            for i, k in enumerate(self.basis[0:nlimit]):
                kpq = k + q
                idxkpq = self.lookup_basis(kpq)
                if idxkpq is not None:
                    idxkpq_list_i += [i]
                    idxkpq_list_kpq += [idxkpq]
            self.ikpq_i += [idxkpq_list_i]
            self.ikpq_kpq += [idxkpq_list_kpq]

        self.ipmq_i = []
        self.ipmq_pmq = []
        for (iq, q) in enumerate(self.qvecs):
            idxpmq_list_i =[]
            idxpmq_list_pmq =[]
            for i, p in enumerate(self.basis[0:nlimit]):
                pmq = p - q
                idxpmq = self.lookup_basis(pmq)
                if idxpmq is not None:
                    idxpmq_list_i += [i]
                    idxpmq_list_pmq += [idxpmq]
            self.ipmq_i += [idxpmq_list_i]
            self.ipmq_pmq += [idxpmq_list_pmq]



        # self.ikpq_i = numpy.array(self.ikpq_i)
        # self.ikpq_kpq = numpy.array(self.ikpq_kpq)
        # self.ipmq_i = numpy.array(self.ipmq_i)
        # self.ipmq_pmq = numpy.array(self.ipmq_pmq)
        for (iq, q) in enumerate(self.qvecs):
            self.ikpq_i[iq]  = numpy.array(self.ikpq_i[iq], dtype=numpy.int64)
            self.ikpq_kpq[iq] = numpy.array(self.ikpq_kpq[iq], dtype=numpy.int64)
            self.ipmq_i[iq]  = numpy.array(self.ipmq_i[iq], dtype=numpy.int64)
            self.ipmq_pmq[iq] = numpy.array(self.ipmq_pmq[iq], dtype=numpy.int64)


        # if(verbose):
        print("# Constructing two-body potentials incore.")
        (self.chol_vecs, self.iA, self.iB) = self.two_body_potentials_incore()
        print("# Approximate memory required for "
              "two-body potentials: %f GB."%(3*self.iA.nnz*16/(1024**3)))
        print("# Constructing two_body_potentials_incore finished")
        if verbose:
            print ("# Finished setting up Generic system object.")


    def sp_energies(self, kfac, ecut):
        """Calculate the allowed kvectors and resulting single particle eigenvalues (basically kinetic energy)
        which can fit in the sphere in kspace determined by ecut.
        Parameters
        ----------
        kfac : float
            kspace grid spacing.
        ecut : float
            energy cutoff.
        Returns
        -------
        spval : :class:`numpy.ndarray`
            Array containing sorted single particle eigenvalues.
        kval : :class:`numpy.ndarray`
            Array containing basis vectors, sorted according to their
            corresponding single-particle energy.
        """

        # Scaled Units to match with HANDE.
        # So ecut is measured in units of 1/kfac^2.
        nmax = int(math.ceil(numpy.sqrt((2*ecut))))

        spval = []
        vec = []
        kval = []
        ks = self.ktwist

        for ni in range(-nmax, nmax+1):
            for nj in range(-nmax, nmax+1):
                for nk in range(-nmax, nmax+1):
                    spe = 0.5*(ni**2 + nj**2 + nk**2)
                    if (spe <= ecut):
                        kijk = [ni,nj,nk]
                        kval.append(kijk)
                        # Reintroduce 2 \pi / L factor.
                        ek = 0.5*numpy.dot(numpy.array(kijk)+ks,
                                       numpy.array(kijk)+ks)
                        spval.append(kfac**2*ek)

        # Sort the arrays in terms of increasing energy.
        spval = numpy.array(spval)
        ix = numpy.argsort(spval)
        spval = spval[ix]
        kval = numpy.array(kval)[ix]

        return (spval, kval, nmax)

    def create_lookup_table(self):
        basis_ix = []
        for k in self.basis:
            basis_ix.append(self.map_basis_to_index(k))
        self.lookup = numpy.zeros(max(basis_ix)+1, dtype=int)
        for i, b in enumerate(basis_ix):
            self.lookup[b] = i
        self.max_ix = max(basis_ix)

    def lookup_basis(self, vec):
        if (numpy.dot(vec,vec) <= self.imax_sq):
            ix = self.map_basis_to_index(vec)
            if ix >= len(self.lookup):
                ib = None
            else:
                ib = self.lookup[ix]
            return ib
        else:
            ib = None

    def map_basis_to_index(self, k):
        return ((k[0]+self.nmax) +
                self.shifted_nmax*(k[1]+self.nmax) +
                self.shifted_nmax*self.shifted_nmax*(k[2]+self.nmax))

    def madelung(self):
        """Use expression in Schoof et al. (PhysRevLett.115.130402) for the
        Madelung contribution to the total energy fitted to L.M. Fraser et al.
        Phys. Rev. B 53, 1814.
        Parameters
        ----------
        rs : float
            Wigner-Seitz radius.
        ne : int
            Number of electrons.
        Returns
        -------
        v_M: float
            Madelung potential (in Hartrees).
        """
        c1 = -2.837297
        c2 = (3.0/(4.0*math.pi))**(1.0/3.0)
        return c1 * c2 / (self.ne**(1.0/3.0) * self.rs)

    def vq(self, q):
        """The typical 3D Coulomb kernel
        Parameters
        ----------
        q : float
            a plane-wave vector
        Returns
        -------
        v_M: float
            3D Coulomb kernel (in Hartrees)
        """
        return 4*math.pi / numpy.dot(q, q)

    def mod_one_body(self, T):
        """ Add a diagonal term of two-body Hamiltonian to the one-body term
        Parameters
        ----------
        T : float
            one-body Hamiltonian (i.e. kinetic energy)
        Returns
        -------
        h1e_mod: float
            modified one-body Hamiltonian
        """
        h1e_mod = numpy.copy(T)

        fac = 1.0 / (2.0 * self.vol)
        for (i, ki) in enumerate(self.basis):
            for (j, kj) in enumerate(self.basis):
                if i != j:
                    q = self.kfac * (ki - kj)
                    h1e_mod[i,i] = h1e_mod[i,i] - fac * self.vq(q)
        return h1e_mod

    def density_operator(self, iq):
        """ Density operator as defined in Eq.(6) of PRB(75)245123
        Parameters
        ----------
        q : float
            a plane-wave vector
        Returns
        -------
        rho_q: float
            density operator
        """
        nnz = self.rho_ikpq_kpq[iq].shape[0] # Number of non-zeros
        ones = numpy.ones((nnz), dtype=numpy.complex128)
        rho_q = scipy.sparse.csc_matrix((ones, (self.rho_ikpq_kpq[iq], self.rho_ikpq_i[iq])),
            shape = (self.nbasis, self.nbasis) ,dtype=numpy.complex128 )
        return rho_q

    def scaled_density_operator_incore(self, transpose):
        """ Density operator as defined in Eq.(6) of PRB(75)245123
        Parameters
        ----------
        q : float
            a plane-wave vector
        Returns
        -------
        rho_q: float
            density operator
        """
        rho_ikpq_i = []
        rho_ikpq_kpq = []
        for (iq, q) in enumerate(self.qvecs):
            idxkpq_list_i =[]
            idxkpq_list_kpq =[]
            for i, k in enumerate(self.basis):
                kpq = k + q
                idxkpq = self.lookup_basis(kpq)
                if idxkpq is not None:
                    idxkpq_list_i += [i]
                    idxkpq_list_kpq += [idxkpq]
            rho_ikpq_i += [idxkpq_list_i]
            rho_ikpq_kpq += [idxkpq_list_kpq]

        for (iq, q) in enumerate(self.qvecs):
            rho_ikpq_i[iq]  = numpy.array(rho_ikpq_i[iq], dtype=numpy.int64)
            rho_ikpq_kpq[iq] = numpy.array(rho_ikpq_kpq[iq], dtype=numpy.int64)

        nq = len(self.qvecs)
        nnz = 0
        for iq in range(nq):
            nnz += rho_ikpq_kpq[iq].shape[0]

        col_index = []
        row_index = []

        values = []

        if (transpose):
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol
                piovol = math.pi / (self.vol)
                factor = (piovol/numpy.dot(qscaled,qscaled))**0.5

                for (innz, kpq) in enumerate(rho_ikpq_kpq[iq]):
                    row_index += [rho_ikpq_kpq[iq][innz] + rho_ikpq_i[iq][innz]*self.nbasis]
                    col_index += [iq]
                    values += [factor]
        else:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol
                piovol = math.pi / (self.vol)
                factor = (piovol/numpy.dot(qscaled,qscaled))**0.5

                for (innz, kpq) in enumerate(rho_ikpq_kpq[iq]):
                    row_index += [rho_ikpq_kpq[iq][innz]*self.nbasis + rho_ikpq_i[iq][innz]]
                    col_index += [iq]
                    values += [factor]

        rho_q = scipy.sparse.csc_matrix((values, (row_index, col_index)),
            shape = (self.nbasis*self.nbasis, nq) ,dtype=numpy.complex128 )

        return rho_q

    def two_body_potentials_incore(self):
        """Calculatate A and B of Eq.(13) of PRB(75)245123 for a given plane-wave vector q
        Parameters
        ----------
        system :
            system class
        q : float
            a plane-wave vector
        Returns
        -------
        iA : numpy array
            Eq.(13a)
        iB : numpy array
            Eq.(13b)
        """
        # qscaled = self.kfac * self.qvecs

        # # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol

        rho_q = self.scaled_density_operator_incore(False)
        rho_qH = self.scaled_density_operator_incore(True)

        iA = 1j * (rho_q + rho_qH)
        iB = - (rho_q - rho_qH)

        return (rho_q, iA, iB)

def unit_test():
    from scipy.sparse import csr_matrix
    # from openfermion.ops import FermionOperator
    # from openfermion.transforms import get_sparse_operator
    from scipy.linalg import eigvalsh
    from scipy.sparse.linalg import eigsh
    from numpy import linalg as LA

    inputs = {'nup':2,
    'ndown':2,
    'rs':1.0,
    'thermal':True,
    'ecut':21}
    system = UEG(inputs, True)

    # n_spatial = system.nbasis
    # n_qubits = system.nbasis * 2
    # n_electrons = system.nup + system.ndown

    # vacuum = csr_matrix((2 ** n_qubits, 1))
    # vacuum[0, 0] = 1

    # cre_op = FermionOperator()
    # coeff = 1.0

    # site_list = [0 for i in range (n_qubits)]
    # for i in range(n_qubits-n_electrons, n_qubits):
    #     site_list[i] = 1
    # term_list = []
    # for i, index in enumerate(reversed(site_list)):
    #     if index:
    #         term_list.insert(0, (i, 1))
    # term_tuple = tuple(term_list)
    # cre_op += coeff * FermionOperator(term_tuple)

    # cre_op_mat = get_sparse_operator(cre_op, n_qubits)
    # psi0 = cre_op_mat.dot(vacuum)

    # Hamiltonian = FermionOperator()
    # for p in range(n_spatial):
    #     pa = 2*p
    #     pb = 2*p+1
    #     tpp = system.H1[0][p,p]
    #     Hamiltonian += FermionOperator(((pa, 1),(pa, 0)), tpp)
    #     Hamiltonian += FermionOperator(((pb, 1),(pb, 0)), tpp)

    # nq = numpy.shape(system.qvecs)[0]

    # for iq in range(nq):
    #     vq = (1.0/(2.0*system.vol))*system.vqvec[iq]
    #     for (idxkpq,k) in zip(system.ikpq_kpq[iq],system.ikpq_i[iq]):
    #         for (idxpmq,p) in zip(system.ipmq_pmq[iq],system.ipmq_i[iq]):
    #             idxkpqa = (idxkpq*2)
    #             idxkpqb = (idxkpq*2+1)
    #             idxpmqa = (idxpmq*2)
    #             idxpmqb = (idxpmq*2+1)
    #             pa =      (p*2)
    #             pb =      (p*2+1)
    #             ka =      (k*2)
    #             kb =      (k*2+1)
    #             Hamiltonian += 0.5*FermionOperator(((idxkpqa,1),(idxpmqa,1),(pa, 0),(ka, 0)), vq)
    #             Hamiltonian += 0.5*FermionOperator(((idxkpqb,1),(idxpmqb,1),(pb, 0),(kb, 0)), vq)
    #             Hamiltonian += 0.5*FermionOperator(((idxkpqa,1),(idxpmqb,1),(pa, 0),(kb, 0)), vq)
    #             Hamiltonian += 0.5*FermionOperator(((idxkpqb,1),(idxpmqa,1),(pb, 0),(ka, 0)), vq)

    # hammat = get_sparse_operator(Hamiltonian, n_qubits)
    # Hpsi0 = hammat.dot(psi0)
    # E0 = psi0.H.dot(Hpsi0)
    # print(E0[0,0]) #((5.7804249284991878+0j), (6.0292135933516064+0j), (-0.24878866485241879+0j))

    # Number = FermionOperator()
    # for p in range(n_spatial):
    #     pa = 2*p
    #     pb = 2*p+1
    #     Number += FermionOperator(((pa, 1),(pa, 0)), 1.0)
    #     Number += FermionOperator(((pb, 1),(pb, 0)), 1.0)

    # evalues, evectors = eigsh(hammat, k=6, which='SA')
    # print(evalues)
    # number = get_sparse_operator(Number, n_qubits)

    # print(hammat.shape)
    # Nevec = number.dot(evectors)
    # enumbers = numpy.einsum("ij,ij->j",Nevec,numpy.conj(evectors))
    # print(enumbers)


if __name__=="__main__":
    unit_test()
