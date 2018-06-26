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

    def __init__(self, inputs, dt, verbose):
        if verbose:
            print ("# Parsing input options.")
        self.name = "UEG"
        self.nup = inputs.get('nup')
        self.ndown = inputs.get('ndown')
        self.rs = inputs.get('rs')
        self.ecut = inputs.get('ecut')
        self.ktwist = numpy.array(inputs.get('ktwist'))
        print("# Number of spin-up electrons = %i"%self.nup)
        print("# Number of spin-down electrons = %i"%self.ndown)
        print("# rs = %10.5f"%self.rs)

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

        # save dt
        self.dt = dt

        print("# zeta = %10.5f"%self.zeta)
        print("# rho = %10.5f"%self.rho)
        print("# L = %10.5f"%self.L)
        print("# vol = %10.5f"%self.vol)
        print("# kfac = %10.5f"%self.kfac)
        print("# ecore = %10.5f"%self.ecore)

        # Single particle eigenvalues and corresponding kvectors
        (self.sp_eigv, self.basis, self.nmax) = self.sp_energies(self.kfac, self.ecut)

        self.shifted_nmax = 2*self.nmax
        self.imax_sq = numpy.dot(self.basis[-1], self.basis[-1])
        self.create_lookup_table()
        for (i, k) in enumerate(self.basis):
            assert(i==self.lookup_basis(k))

        # Number of plane waves.
        self.nbasis = len(self.sp_eigv)
        print("# Number of plane waves = %i"%self.nbasis)
        # Allowed momentum transfers (4*ecut)
        (eigs, qvecs, self.qnmax) = self.sp_energies(self.kfac, 4*self.ecut)
        # Omit Q = 0 term.
        self.qvecs = numpy.copy(qvecs[1:])
        self.vqvec = numpy.array([self.vq(self.kfac*q) for q in self.qvecs])
        # Number of momentum transfer vectors / auxiliary fields.
        # Can reduce by symmetry but be stupid for the moment.
        self.nfields = 2*len(self.qvecs)
        T = numpy.diag(self.sp_eigv)
        self.T = numpy.array([T, T]) # Making alpha and beta
        h1e_mod = self.mod_one_body(T)
        self.h1e_mod = numpy.array([h1e_mod, h1e_mod])
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

        for ni in range(-nmax, nmax+1):
            for nj in range(-nmax, nmax+1):
                for nk in range(-nmax, nmax+1):
                    spe = 0.5*(ni**2 + nj**2 + nk**2)
                    if (spe <= ecut):
                        kval.append([ni,nj,nk])
                        # Reintroduce 2 \pi / L factor.
                        spval.append(kfac**2*spe)

        # Sort the arrays in terms of increasing energy.
        spval = numpy.array(spval)
        kval = [x for y, x in sorted(zip(spval, kval))]
        kval = numpy.array(kval)
        spval.sort()

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

    def density_operator(self, q):
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
        assert (q[0] != 0 or q[1] != 0 or q[2] !=0)
        rho_q = numpy.zeros(shape=(self.nbasis, self.nbasis))

        idxkpq = []
        for (i, ki) in enumerate(self.basis):
            kipq = ki+q
            e = numpy.sum(kipq**2 /2.0)
            if (e <= self.ecut):
                idx = self.lookup_basis(kipq)
                if (idx != None):
                    idxkpq += [(idx,i)]

        for (i,j) in idxkpq:
            rho_q[i,j] = 1

        return rho_q

def unit_test():
    inputs = {'nup':1, 
    'ndown':1,
    'rs':1.0,
    'ecut':1.0}
    system = UEG(inputs, True)

    for (i, qi) in enumerate(system.qvecs):
        rho_q = system.density_operator(qi)
        rho_mq = system.density_operator(-qi)
        print (numpy.linalg.norm(rho_q-rho_mq.T))

if __name__=="__main__":
    unit_test()
