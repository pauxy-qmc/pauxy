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
from pauxy.estimators.thermal import particle_number, one_rdm_from_G
from pauxy.estimators.ueg import local_energy_ueg
from pauxy.estimators.hubbard import local_energy_hubbard, local_energy_hubbard_ghf
from pauxy.estimators.greens_function import gab_mod_ovlp
from pauxy.estimators.generic import (
    local_energy_generic_opt,
    local_energy_generic,
    local_energy_generic_cholesky
)
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
        self.thermal = mixed.get('thermal', False)
        self.average_gf = mixed.get('average_gf', False)
        self.rdm = mixed.get('rdm', False)
        self.verbose = mixed.get('verbose', True)
        self.nmeasure = qmc.nsteps // qmc.nmeasure
        if self.thermal:
            self.header = ['iteration', 'Weight', 'E_num', 'E_denom', 'E',
                           'EKin', 'EPot', 'Nav', 'time']
        else:
            self.header = ['iteration', 'Weight', 'E_num', 'E_denom', 'E',
                           'EKin', 'EPot', 'time']
        self.nreg = len(self.header[1:])
        self.dtype = dtype
        self.G = numpy.zeros(trial.G.shape, trial.G.dtype)
        self.estimates = numpy.zeros(self.nreg + self.G.size, dtype=dtype)
        self.names = EstimatorEnum(self.thermal)
        self.estimates[self.names.time] = time.time()
        self.global_estimates = numpy.zeros(self.nreg+self.G.size,
                                            dtype=dtype)
        self.key = {
            'iteration': "Simulation iteration. iteration*dt = tau.",
            'Weight': "Total walker weight.",
            'E_num': "Numerator for projected energy estimator.",
            'E_denom': "Denominator for projected energy estimator.",
            'E': "Projected energy estimator.",
            'EKin': "Mixed kinetic energy estimator.",
            'EPot': "Mixed potential energy estimator.",
            'Nav': "Average number of electrons.",
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
        if free_projection:
            for i, w in enumerate(psi.walkers):
                w.greens_function(trial)
                E, T, V = w.local_energy(system)
                # For T > 0 w.ot = 1 always.
                wfac = w.weight * w.ot * w.phase
                self.estimates[self.names.enumer] += wfac * E
                self.estimates[self.names.ekin:self.names.epot+1] += (
                        wfac * numpy.array([T,V])
                )
                if self.thermal:
                    nav = particle_number(one_rdm_from_G(w.G))
                    self.estimates[self.names.nav] += wfac * nav
                self.estimates[self.names.weight] += w.weight
                self.estimates[self.names.edenom] += wfac
        else:
            # When using importance sampling we only need to know the current
            # walkers weight as well as the local energy, the walker's overlap
            # with the trial wavefunction is not needed.
            for i, w in enumerate(psi.walkers):
                if self.thermal:
                    if self.average_gf:
                        E_sum = 0
                        T_sum = 0
                        V_sum = 0
                        nav = 0
                        print(w.stack_length)
                        for ts in range(w.stack_length):
                            w.greens_function(trial, slice_ix=ts*w.stack_size)
                            E, T, V = w.local_energy(system)
                            E_sum += E
                            T_sum += T
                            V_sum += V
                            nav += particle_number(one_rdm_from_G(w.G))
                        self.estimates[self.names.nav] += w.weight * nav / w.stack_length
                        self.estimates[self.names.enumer] += w.weight*E_sum.real/w.stack_length
                        self.estimates[self.names.ekin:self.names.epot+1] += (
                                w.weight*numpy.array([T_sum,V_sum]).real/w.stack_length
                        )
                    else:
                        w.greens_function(trial)
                        E, T, V = w.local_energy(system)
                        nav = particle_number(one_rdm_from_G(w.G))
                        self.estimates[self.names.nav] += w.weight * nav
                        self.estimates[self.names.enumer] += w.weight*E.real
                        self.estimates[self.names.ekin:self.names.epot+1] += (
                                w.weight*numpy.array([T,V]).real
                        )
                else:
                    w.greens_function(trial)
                    E, T, V = w.local_energy(system)
                    self.estimates[self.names.enumer] += w.weight*E.real
                    self.estimates[self.names.ekin:self.names.epot+1] += (
                            w.weight*numpy.array([T,V]).real
                    )
                self.estimates[self.names.weight] += w.weight
                self.estimates[self.names.edenom] += w.weight
                if self.rdm:
                    self.estimates[self.names.time+1:] += w.weight*w.G.flatten().real

    def print_step(self, comm, nprocs, step, nmeasure, free_projection=False):
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
        if self.thermal:
            if free_projection:
                es[ns.nav] = es[ns.nav]
            else:
                es[ns.nav] = es[ns.nav] / denom
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

    def reset(self, h5f):
        energies = h5f.create_group('mixed_estimates')
        energies.create_dataset('headers',
                                data=numpy.array(self.header[1:], dtype=object),
                                dtype=h5py.special_dtype(vlen=str))
        self.output = H5EstimatorHelper(energies, 'energies',
                                        (self.nmeasure + 1, self.nreg),
                                        self.dtype)
        self.output.reset()

# Energy evaluation routines.

def local_energy(system, G, Ghalf=None, opt=True):
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
    elif system.name == "UEG":
        return local_energy_ueg(system, G)
    else:
        if opt:
            return local_energy_generic_opt(system, G, Ghalf)
        else:
            return local_energy_generic(system, G)

def local_energy_multi_det_full(system, A, B, coeffsA, coeffsB):
    weight = 0
    energies = 0
    denom = 0
    nup = system.nup
    for ix, (Aix, cix) in enumerate(zip(A, coeffsA)):
        for iy, (Biy, ciy) in enumerate(zip(B, coeffsB)):
            # construct "local" green's functions for each component of A
            Gup, inv_O_up = gab_mod_ovlp(Biy[:,:nup], Aix[:,:nup])
            Gdn, inv_O_dn = gab_mod_ovlp(Biy[:,nup:], Aix[:,nup:])
            ovlp = 1.0 / (scipy.linalg.det(inv_O_up)*scipy.linalg.det(inv_O_dn))
            weight = cix*(ciy.conj()) * ovlp
            G = numpy.array([Gup, Gdn])
            e = numpy.array(local_energy(system, G, opt=False))
            energies += weight * e
            denom += weight
    return tuple(energies/denom)

def local_energy_multi_det(system, Gi, weights):
    weight = 0
    energies = 0
    denom = 0
    for w, G in zip(weights, Gi):
        # construct "local" green's functions for each component of A
        energies += w * numpy.array(local_energy(system, G, opt=False))
        denom += w
    return tuple(energies/denom)

class EstimatorEnum(object):
    """Enum structure for help with indexing Mixed estimators.

    python's support for enums doesn't help as it indexes from 1.
    """

    def __init__(self, thermal=False):
        self.weight = 0
        self.enumer = 1
        self.edenom = 2
        self.eproj = 3
        self.ekin = 4
        self.epot = 5
        if thermal:
            self.nav = 6
            self.time = 7
        else:
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
