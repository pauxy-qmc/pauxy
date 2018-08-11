import h5py
import numpy
try:
    from mpi4py import MPI
    mpi_sum = MPI.SUM
except ImportError:
    mpi_sum = None
import scipy.linalg
import time
from pauxy.estimators.utils import H5EstimatorHelper
from pauxy.utils.io import format_fixed_width_strings, format_fixed_width_floats


class Mixed(object):
    """Class for computing mixed estimates.

    Parameters
    ----------
    mixed : dict
        Input options for mixed estimates.
    root : bool
        True if on root/master processor.
    h5f : :class:`h5py.File`
        Output file object.
    qmc : :class:`pauxy.state.QMCOpts` object.
        Container for qmc input options.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    dtype : complex or float
        Output type.

    Attributes
    ----------
    nmeasure : int
        Max number of measurements.
    nreg : int
        Number of regular estimates (exluding iteration).
    G : :class:`numpy.ndarray`
        One-particle RDM.
    estimates : :class:`numpy.ndarray`
        Store for mixed estimates per processor.
    global_estimates : :class:`numpy.ndarray`
        Store for mixed estimates accross all processors.
    names : :class:`pauxy.estimators.EstimEnum`
        Enum for locating estimates in estimates array.
    header : int
        Output header.
    key : dict
        Explanation of output.
    output : :class:`pauxy.estimators.H5EstimatorHelper`
        Class for outputting data to HDF5 group.
    output : :class:`pauxy.estimators.H5EstimatorHelper`
        Class for outputting rdm data to HDF5 group.
    """

    def __init__(self, mixed, root, h5f, qmc, trial, dtype):
        self.rdm = mixed.get('rdm', False)
        self.verbose = mixed.get('verbose', True)
        self.nmeasure = qmc.nsteps // qmc.nmeasure
        self.header = ['iteration', 'Weight', 'E_num', 'E_denom', 'E',
                       'EKin', 'EPot', 'time']
        self.nreg = len(self.header[1:])
        self.G = numpy.zeros(trial.G.shape, trial.G.dtype)
        self.estimates = numpy.zeros(self.nreg + self.G.size, dtype=dtype)
        self.names = EstimatorEnum()
        self.estimates[self.names.time] = time.time()
        self.global_estimates = numpy.zeros(self.nreg + self.G.size,
                                            dtype=dtype)
        self.key = {
            'iteration': "Simulation iteration. iteration*dt = tau.",
            'Weight': "Total walker weight.",
            'E_num': "Numerator for projected energy estimator.",
            'E_denom': "Denominator for projected energy estimator.",
            'E': "Projected energy estimator.",
            'EKin': "Mixed kinetic energy estimator.",
            'EPot': "Mixed potential energy estimator.",
            'time': "Time per processor to complete one iteration.",
        }
        if root:
            energies = h5f.create_group('mixed_estimates')
            energies.create_dataset('headers',
                                    data=numpy.array(self.header[1:], dtype=object),
                                    dtype=h5py.special_dtype(vlen=str))
            self.output = H5EstimatorHelper(energies, 'energies',
                                            (self.nmeasure + 1, self.nreg),
                                            dtype)
            if self.rdm:
                name = 'single_particle_greens_function'
                self.dm_output = H5EstimatorHelper(energies, name,
                                                   (self.nmeasure + 1,) +
                                                   self.G.shape,
                                                   dtype)

    def update(self, system, qmc, trial, psi, step, free_projection=False):
        """Update mixed estimates for walkers.

        Parameters
        ----------
        system : system object.
            Container for model input options.
        qmc : :class:`pauxy.state.QMCOpts` object.
            Container for qmc input options.
        trial : :class:`pauxy.trial_wavefunction.X' object
            Trial wavefunction class.
        psi : :class:`pauxy.walkers.Walkers` object
            CPMC wavefunction.
        step : int
            Current simulation step
        free_projection : bool
            True if doing free projection.
        """
        if not free_projection:
            # When using importance sampling we only need to know the current
            # walkers weight as well as the local energy, the walker's overlap
            # with the trial wavefunction is not needed.
            for i, w in enumerate(psi.walkers):
                w.greens_function(trial)
                E, T, V = w.local_energy(system)
                self.estimates[self.names.enumer] += (
                        w.weight*E.real
                )
                self.estimates[self.names.ekin:self.names.epot+1] += (
                        w.weight*numpy.array([T,V]).real
                )
                self.estimates[self.names.weight] += w.weight
                self.estimates[self.names.edenom] += w.weight
                if self.rdm:
                    self.estimates[self.names.time+1:] += w.weight*w.G.flatten().real
        else:
            for i, w in enumerate(psi.walkers):
                w.greens_function(trial)
                E, T, V = w.local_energy(system)
                self.estimates[self.names.enumer] += w.weight*E*w.ot
                self.estimates[self.names.ekin:self.names.epot+1] += w.weight*numpy.array([T,V])*w.ot
                self.estimates[self.names.weight] += w.weight
                self.estimates[self.names.edenom] += (w.weight*w.ot)

    def print_step(self, comm, nprocs, step, nmeasure):
        """Print mixed estimates to file.

        This reduces estimates arrays over processors. On return estimates
        arrays are zerod.

        Parameters
        ----------
        comm :
            MPI communicator.
        nprocs : int
            Number of processors.
        step : int
            Current iteration number.
        nmeasure : int
            Number of steps between measurements.
        """
        es = self.estimates
        ns = self.names
        denom = es[ns.edenom]*nprocs / nmeasure
        es[ns.eproj] = es[ns.enumer] / denom
        es[ns.ekin:ns.epot+1] /= denom
        es[ns.weight:ns.enumer] = es[ns.weight:ns.enumer]
        es[ns.time] = (time.time()-es[ns.time]) / nprocs
        comm.Reduce(es, self.global_estimates, op=mpi_sum)
        if comm.Get_rank() == 0:
            if self.verbose:
                print (format_fixed_width_floats([step]+
                   list(self.global_estimates[:ns.time+1].real/nmeasure)))
            self.output.push(self.global_estimates[:ns.time+1]/nmeasure)
            if self.rdm:
                rdm = self.global_estimates[self.nreg:].reshape(self.G.shape)
                self.dm_output.push(rdm/denom/nmeasure)
        self.zero()

    def print_key(self, eol='', encode=False):
        """Print out information about what the estimates are.

        Parameters
        ----------
        eol : string, optional
            String to append to output, e.g., Default : ''.
        encode : bool
            In True encode output to be utf-8.
        """
        header = (
            eol + '# Explanation of output column headers:\n' +
            '# -------------------------------------' + eol
        )
        if encode:
            header = header.encode('utf-8')
        print(header)
        for (k, v) in self.key.items():
            s = '# %s : %s' % (k, v) + eol
            if encode:
                s = s.encode('utf-8')
            print(s)

    def print_header(self, eol='', encode=False):
        r"""Print out header for estimators

        Parameters
        ----------
        eol : string, optional
            String to append to output, Default : ''.
        encode : bool
            In True encode output to be utf-8.

        Returns
        -------
        None
        """
        s = format_fixed_width_strings(self.header) + eol
        if encode:
            s = s.encode('utf-8')
        print(s)

    def projected_energy(self):
        """Computes projected energy from estimator array.

        Returns
        -------
        eproj : float
            Mixed estimate for projected energy.
        """
        numerator = self.estimates[self.names.enumer]
        denominator = self.estimates[self.names.edenom]
        return (numerator / denominator).real

    def zero(self):
        """Zero (in the appropriate sense) various estimator arrays."""
        self.estimates[:] = 0
        self.global_estimates[:] = 0
        self.estimates[self.names.time] = time.time()

# Energy evaluation routines.

def local_energy(system, G):
    """Helper routine to compute local energy.

    Parameters
    ----------
    system : system object
        system object.
    G : :class:`numpy.ndarray`
        1RDM.

    Returns
    -------
    (E,T,V) : tuple
        Total, one-body and two-body energy.
    """
    ghf = (G.shape[-1] == 2*system.nbasis)
    if system.name == "Hubbard":
        if ghf:
            return local_energy_ghf(system, G)
        else:
            return local_energy_hubbard(system, G)
    else:
        return local_energy_generic(system, G)


def local_energy_hubbard(system, G):
    r"""Calculate local energy of walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    G : :class:`numpy.ndarray`
        Walker's "Green's function"

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    ke = numpy.sum(system.T[0] * G[0] + system.T[1] * G[1])
    pe = sum(system.U * G[0][i][i] * G[1][i][i]
             for i in range(0, system.nbasis))

    return (ke + pe, ke, pe)


def local_energy_ghf(system, Gi, weights, denom):
    """Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    Gi : :class:`numpy.ndarray`
        Array of Walker's "Green's function"
    denom : float
        Overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    ke = numpy.einsum('i,ikl,kl->', weights, Gi, system.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(Gi[:,:system.nbasis,:system.nbasis], axis1=1, axis2=2)
    gdd = numpy.diagonal(Gi[:,system.nbasis:,system.nbasis:], axis1=1, axis2=2)
    gud = numpy.diagonal(Gi[:,system.nbasis:,:system.nbasis], axis1=1, axis2=2)
    gdu = numpy.diagonal(Gi[:,:system.nbasis,system.nbasis:], axis1=1, axis2=2)
    gdiag = guu*gdd - gud*gdu
    pe = system.U * numpy.einsum('j,jk->', weights, gdiag) / denom
    return (ke+pe, ke, pe)


def local_energy_multi_det(system, Gi, weights):
    """Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    Gi : :class:`numpy.ndarray`
        Array of Walker's "Green's function"
    weights : :class:`numpy.ndarray`
        Components of overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    denom = numpy.sum(weights)
    ke = numpy.einsum('i,ikl,kl->', weights, Gi, system.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(Gi[:,:,:system.nup], axis1=1,
                         axis2=2)
    gdd = numpy.diagonal(Gi[:,:,system.nup:], axis1=1,
                         axis2=2)
    pe = system.U * numpy.einsum('j,jk->', weights, guu*gdd) / denom
    return (ke+pe, ke, pe)

def local_energy_ghf_full(system, GAB, weights):
    r"""Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    system : :class:`Hubbard`
        System information for the Hubbard model.
    GAB : :class:`numpy.ndarray`
        Matrix of Green's functions for different SDs A and B.
    weights : :class:`numpy.ndarray`
        Components of overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L, T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    denom = numpy.sum(weights)
    ke = numpy.einsum('ij,ijkl,kl->', weights, GAB, system.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(GAB[:,:,:system.nbasis,:system.nbasis], axis1=2,
                         axis2=3)
    gdd = numpy.diagonal(GAB[:,:,system.nbasis:,system.nbasis:], axis1=2,
                         axis2=3)
    gud = numpy.diagonal(GAB[:,:,system.nbasis:,:system.nbasis], axis1=2,
                         axis2=3)
    gdu = numpy.diagonal(GAB[:,:,:system.nbasis,system.nbasis:], axis1=2,
                         axis2=3)
    gdiag = guu*gdd - gud*gdu
    pe = system.U * numpy.einsum('ij,ijk->', weights, gdiag) / denom
    return (ke+pe, ke, pe)

def local_energy_generic(system, G):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the full form for the two-electron integrals.

    For testing purposes only.

    Parameters
    ----------
    system : :class:`hubbard`
        System information for the hubbard model.
    G : :class:`numpy.ndarray`
        Walker's "green's function"

    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    e1 = (numpy.einsum('ij,ji->', system.T[0], G[0]) +
          numpy.einsum('ij,ji->', system.T[1], G[1]))
    euu = 0.5*(numpy.einsum('pqrs,pr,qs->', system.h2e, G[0], G[0]) -
               numpy.einsum('pqrs,ps,qr->', system.h2e, G[0], G[0]))
    edd = 0.5*(numpy.einsum('pqrs,pr,qs->', system.h2e, G[1], G[1]) -
               numpy.einsum('pqrs,ps,qr->', system.h2e, G[1], G[1]))
    eud = 0.5*numpy.einsum('pqrs,pr,qs->', system.h2e, G[0], G[1])
    edu = 0.5*numpy.einsum('pqrs,pr,qs->', system.h2e, G[1], G[0])
    e2 = euu + edd + eud + edu
    return (e1+e2+system.ecore, e1+system.ecore, e2)

def local_energy_generic_cholesky(system, G):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals.

    Parameters
    ----------
    system : :class:`hubbard`
        System information for the hubbard model.
    G : :class:`numpy.ndarray`
        Walker's "green's function"

    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    e1 = (numpy.einsum('ij,ji->', system.T[0], G[0]) +
          numpy.einsum('ij,ji->', system.T[1], G[1]))
    euu = 0.5*(numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                            system.chol_vecs, G[0], G[0]) -
               numpy.einsum('lpr,lqs,ps,qr->', system.chol_vecs,
                            system.chol_vecs, G[0], G[0]))
    edd = 0.5*(numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                            system.chol_vecs, G[1], G[1]) -
               numpy.einsum('lpr,lqs,ps,qr->', system.chol_vecs,
                            system.chol_vecs, G[1], G[1]))
    eud = 0.5*numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                           system.chol_vecs, G[0], G[1])
    edu = 0.5*numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                           system.chol_vecs, G[1], G[0])
    e2 = euu + edd + eud + edu
    return (e1+e2+system.ecore, e1+system.ecore, e2)

def local_energy_generic_cholesky_opt(system, Theta, L):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals and the optimised
    algorithm using precomputed tensors.

    .. warning::

            Doesn't work.

    Parameters
    ----------
    system : :class:`hubbard`
        System information for the hubbard model.
    Theta : :class:`numpy.ndarray`
        Rotated Green's function.
    L : :class:`numpy.ndarray`
        Rotated Cholesky vectors.
    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    e1 = (numpy.einsum('ij,ji->', system.T[0], G[0]) +
          numpy.einsum('ij,ji->', system.T[1], G[1]))
    euu = 0.5*(numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                            system.chol_vecs, G[0], G[0]) -
               numpy.einsum('lpr,lqs,ps,qr->', system.chol_vecs,
                            system.chol_vecs, G[0], G[0]))
    edd = 0.5*(numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                            system.chol_vecs, G[1], G[1]) -
               numpy.einsum('lpr,lqs,ps,qr->', system.chol_vecs,
                            system.chol_vecs, G[1], G[1]))
    eud = 0.5*numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                           system.chol_vecs, G[0], G[1])
    edu = 0.5*numpy.einsum('lpr,lqs,pr,qs->', system.chol_vecs,
                           system.chol_vecs, G[1], G[0])
    e2 = euu + edd + eud + edu
    return (e1+e2+system.ecore, e1+system.ecore, e2)

# Green's functions

def gab(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    # Todo: check energy evaluation at later point, i.e., if this needs to be
    # transposed. Shouldn't matter for Hubbard model.
    inv_O = scipy.linalg.inv((A.conj().T).dot(B))
    GAB = B.dot(inv_O.dot(A.conj().T))
    return GAB


def gab_mod(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    # Todo: check energy evaluation at later point, i.e., if this needs to be
    # transposed. Shouldn't matter for Hubbard model.
    inv_O = scipy.linalg.inv((A.conj().T).dot(B))
    GAB = B.dot(inv_O)
    return GAB


def gab_multi_det(A, B, coeffs):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle = [B(A^{*T}B)^{-1}A^{*T}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent a multi-determinant trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.
    coeffs: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients. Assumed to be complex
        conjugated.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    # Todo: check energy evaluation at later point, i.e., if this needs to be
    # transposed. Shouldn't matter for Hubbard model.
    Gi = numpy.zeros(A.shape)
    overlaps = numpy.zeros(A.shape[1])
    for (ix, Aix) in enumerate(A):
        # construct "local" green's functions for each component of A
        # Todo: list comprehension here.
        inv_O = scipy.linalg.inv((Aix.conj().T).dot(B))
        Gi[ix] = (B.dot(inv_O.dot(Aix.conj().T))).T
        overlaps[ix] = 1.0 / scipy.linalg.det(inv_O)
    denom = numpy.dot(coeffs, overlaps)
    return numpy.einsum('i,ijk,i->jk', coeffs, Gi, overlaps) / denom


def construct_multi_ghf_gab_back_prop(A, B, coeffs, bp_weights):
    """Green's function for back propagation.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.
    coeffs: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients. Assumed to be complex
        conjugated.
    bp_weights : :class:`numpy.ndarray`
        Factors arising from GS orthogonalisation.

    Returns
    -------
    G : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    M = A.shape[1] // 2
    Gi, overlaps = construct_multi_ghf_gab(A, B, coeffs)
    scale = max(max(bp_weights), max(overlaps))
    full_weights = bp_weights * coeffs * overlaps / scale
    denom = sum(full_weights)
    G = numpy.einsum('i,ijk->jk', full_weights, Gi) / denom

    return G


def construct_multi_ghf_gab(A, B, coeffs, Gi=None, overlaps=None):
    """Construct components of multi-ghf trial wavefunction.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.
    Gi : :class:`numpy.ndarray`
        Array to store components of G. Default: None.
    overlaps : :class:`numpy.ndarray`
        Array to overlaps. Default: None.

    Returns
    -------
    Gi : :class:`numpy.ndarray`
        Array to store components of G. Default: None.
    overlaps : :class:`numpy.ndarray`
        Array to overlaps. Default: None.
    """
    M = B.shape[0] // 2
    if Gi is None:
        Gi = numpy.zeros(shape=(A.shape[0],A.shape[1],A.shape[1]), dtype=A.dtype)
    if overlaps is None:
        overlaps = numpy.zeros(A.shape[0], dtype=A.dtype)
    for (ix, Aix) in enumerate(A):
        # construct "local" green's functions for each component of A
        # Todo: list comprehension here.
        inv_O = scipy.linalg.inv((Aix.conj().T).dot(B))
        Gi[ix] = (B.dot(inv_O.dot(Aix.conj().T)))
        overlaps[ix] = 1.0 / scipy.linalg.det(inv_O)
    return (Gi, overlaps)


def gab_multi_det_full(A, B, coeffsA, coeffsB, GAB, weights):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle = [B(A^{*T}B)^{-1}A^{*T}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    .. todo: Fix docstring

    Here we assume both A and B are multi-determinant expansions.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Array containing elements of multi-determinant matrix representation of
        the ket used to construct G.
    coeffsA: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients for wavefunction A. Assumed to
        be complex conjugated.
    coeffsB: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients for wavefunction A. Assumed to
        be complex conjugated.
    GAB : :class:`numpy.ndarray`
        Matrix of Green's functions.
    weights : :class:`numpy.ndarray`
        Matrix of weights needed to construct G

    Returns
    -------
    G : :class:`numpy.ndarray`
        Full Green's function.
    """
    for ix, (Aix, cix) in enumerate(zip(A, coeffsA)):
        for iy, (Biy, ciy) in enumerate(zip(B, coeffsB)):
            # construct "local" green's functions for each component of A
            inv_O = scipy.linalg.inv((Aix.conj().T).dot(Biy))
            GAB[ix,iy] = (Biy.dot(inv_O)).dot(Aix.conj().T)
            weights[ix,iy] =  cix*(ciy.conj()) / scipy.linalg.det(inv_O)
    denom = numpy.sum(weights)
    G = numpy.einsum('ij,ijkl->kl', weights, GAB) / denom
    return G


class EstimatorEnum(object):
    """Enum structure for help with indexing Mixed estimators.

    python's support for enums doesn't help as it indexes from 1.
    """

    def __init__(self):
        # Exception for alignment of equal sign.
        self.weight = 0
        self.enumer = 1
        self.edenom = 2
        self.eproj = 3
        self.ekin = 4
        self.epot = 5
        self.time = 6


def eproj(estimates, enum):
    """Real projected energy.

    Parameters
    ----------
    estimates : numpy.array
        Array containing estimates averaged over all processors.
    enum : :class:`pauxy.estimators.EstimatorEnum` object
        Enumerator class outlining indices of estimates array elements.

    Returns
    -------
    eproj : float
        Projected energy from current estimates array.
    """

    numerator = estimates[enum.enumer]
    denominator = estimates[enum.edenom]
    return (numerator/denominator).real
