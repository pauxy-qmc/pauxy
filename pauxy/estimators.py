"""Routines and classes for estimation of observables."""

from __future__ import print_function

import numpy
import time
import copy
import warnings
try:
    from mpi4py import MPI
    mpi_sum = MPI.SUM
except ImportError:
    mpi_sum = None
import scipy.linalg
import os
import h5py
import pauxy.utils
import pauxy.propagation


class Estimators(object):
    """Container for qmc estimates of observables.

    Parameters
    ----------
    estimates : dict
        input options detailing which estimators to calculate. By default only
        mixed estimates will be calculated.
    root : bool
        True if on root/master processor.
    qmc : :class:`pauxy.state.QMCOpts` object.
        Container for qmc input options.
    system : :class:`pauxy.hubbard.Hubbard` / system object in general.
        Container for model input options.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    verbose : bool
        If true we print out additional setup information.

    Attributes
    ----------
    h5f : :class:`h5py.File`
        Output file object.
    estimates : dict
        Dictionary of estimator objects.
    back_propagation : bool
        True if doing back propagation, specified in estimates dict.
    nbp : int
        Number of back propagation steps.
    nprop_tot : int
        Total number of auxiliary field configurations we store / use for back
        propagation and itcf calculation.
    calc_itcf : bool
        True if calculating imaginary time correlation functions (ITCFs).
    """

    def __init__(self, estimates, root, qmc, system, trial, BT2, verbose=False):
        if root:
            index = estimates.get('index', 0)
            h5f_name = estimates.get('filename', None)
            if h5f_name is None:
                overwrite = estimates.get('overwrite', True)
                h5f_name = 'estimates.%s.h5' % index
                while os.path.isfile(h5f_name) and not overwrite:
                    index = int(h5f_name.split('.')[1])
                    index = index + 1
                    h5f_name = 'estimates.%s.h5' % index
            self.h5f = h5py.File(h5f_name, 'w')
        else:
            self.h5f = None
        # Sub-members:
        # 1. Back-propagation
        mixed = estimates.get('mixed', {})
        self.estimators = {}
        dtype = complex
        self.estimators['mixed'] = Mixed(mixed, root, self.h5f,
                                         qmc, trial, dtype)
        bp = estimates.get('back_propagated', None)
        self.back_propagation = bp is not None
        if self.back_propagation:
            self.estimators['back_prop'] = BackPropagation(bp, root, self.h5f,
                                                           qmc, system, trial,
                                                           dtype, BT2)
            self.nprop_tot = self.estimators['back_prop'].nmax
            self.nbp = self.estimators['back_prop'].nmax
        else:
            self.nprop_tot = 1
            self.nbp = 1
        # 2. Imaginary time correlation functions.
        itcf = estimates.get('itcf', None)
        self.calc_itcf = itcf is not None
        if self.calc_itcf:
            self.estimators['itcf'] = ITCF(itcf, qmc, trial, root, self.h5f,
                                           system.nbasis, dtype,
                                           self.nprop_tot, BT2)
            self.nprop_tot = self.estimators['itcf'].nprop_tot

    def print_step(self, comm, nprocs, step, nmeasure):
        """Print QMC estimates.

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
        for k, e in self.estimators.items():
            e.print_step(comm, nprocs, step, nmeasure)
        if comm.Get_rank() == 0:
            self.h5f.flush()

    def update(self, system, qmc, trial, psi, step, free_projection=False):
        """Update estimators

        Parameters
        ----------
        system : system object in general.
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
        for k, e in self.estimators.items():
            e.update(system, qmc, trial, psi, step, free_projection)


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
                                    data=numpy.array(
                                        self.header[1:], dtype=object),
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
                self.estimates[self.names.enumer] += (
                        (w.weight*w.local_energy(system)[0]*w.ot)
                )
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
            print (pauxy.utils.format_fixed_width_floats([step]+
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
        s = pauxy.utils.format_fixed_width_strings(self.header) + eol
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


class BackPropagation(object):
    """Class for computing back propagated estimates.

    Parameters
    ----------
    bp : dict
        Input options for mixed estimates.
    root : bool
        True if on root/master processor.
    h5f : :class:`h5py.File`
        Output file object.
    qmc : :class:`pauxy.state.QMCOpts` object.
        Container for qmc input options.
    system : system object 
        System object.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    dtype : complex or float
        Output type.
    BT2 : :class:`numpy.ndarray`
        One-body propagator for back propagation.

    Attributes
    ----------
    nmax : int
        Max number of measurements.
    header : int
        Output header.
    rdm : bool 
        True if output BP RDM to file.
    nreg : int
        Number of regular estimates (exluding iteration).
    G : :class:`numpy.ndarray`
        One-particle RDM.
    estimates : :class:`numpy.ndarray`
        Store for mixed estimates per processor.
    global_estimates : :class:`numpy.ndarray`
        Store for mixed estimates accross all processors.
    output : :class:`pauxy.estimators.H5EstimatorHelper`
        Class for outputting data to HDF5 group.
    rdm_output : :class:`pauxy.estimators.H5EstimatorHelper`
        Class for outputting rdm data to HDF5 group.
    """

    def __init__(self, bp, root, h5f, qmc, system, trial, dtype, BT2):
        self.nmax = bp.get('nback_prop', 0)
        self.header = ['iteration', 'weight', 'E', 'T', 'V']
        self.rdm = bp.get('rdm', False)
        self.nreg = len(self.header[1:])
        self.G = numpy.zeros(trial.G.shape, dtype=trial.G.dtype)
        self.estimates = numpy.zeros(self.nreg + self.G.size,
                                     dtype=trial.G.dtype)
        self.global_estimates = numpy.zeros(self.nreg + self.G.size,
                                            dtype=trial.G.dtype)
        self.nstblz = qmc.nstblz
        self.BT2 = BT2
        self.restore_weights = bp.get('restore_weights', None)
        self.dt = qmc.dt
        self.key = {
            'iteration': "Simulation iteration when back-propagation "
                         "measurement occured.",
            'E_var': "BP estimate for internal energy.",
            'T': "BP estimate for kinetic energy.",
            'V': "BP estimate for potential energy."
        }
        if root:
            energies = h5f.create_group('back_propagated_estimates')
            header = numpy.array(self.header[1:], dtype=object)
            energies.create_dataset('headers', data=header,
                                    dtype=h5py.special_dtype(vlen=str))
            self.output = H5EstimatorHelper(energies, 'energies',
                                            (qmc.nsteps//self.nmax, self.nreg),
                                            trial.G.dtype)
            if self.rdm:
                self.dm_output = H5EstimatorHelper(energies, 'single_particle_greens_function',
                                                  (qmc.nsteps//self.nmax,)+self.G.shape,
                                                  trial.G.dtype)
        if trial.type == 'GHF':
            self.update = self.update_ghf
            if system.name == "Generic":
                self.back_propagate = pauxy.propagation.back_propagate_generic_uhf
            else:
                self.back_propagate = pauxy.propagation.back_propagate_ghf
        else:
            self.update = self.update_uhf
            if system.name == "Generic":
                self.back_propagate = pauxy.propagation.back_propagate_generic
            else:
                self.back_propagate = pauxy.propagation.back_propagate

    def update_uhf(self, system, qmc, trial, psi, step, free_projection=False):
        """Calculate back-propagated estimates for UHF walkers.

        Parameters
        ----------
        system : system object in general.
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
        if step % self.nmax != 0:
            return
        psi_bp = self.back_propagate(system, psi.walkers, trial,
                                     self.nstblz, self.BT2, qmc.dt)
        nup = system.nup
        denominator = 0
        for i, (wnm, wb) in enumerate(zip(psi.walkers, psi_bp)):
            self.G[0] = gab(wb.phi[:,:nup], wnm.phi_old[:,:nup]).T
            self.G[1] = gab(wb.phi[:,nup:], wnm.phi_old[:,nup:]).T
            energies = numpy.array(list(local_energy(system, self.G)))
            if self.restore_weights is not None:
                weight = wnm.weight * self.calculate_weight_factor(wnm)
            else:
                weight = wnm.weight
            denominator += weight
            self.estimates[1:] = (
                self.estimates[1:] +
                weight*numpy.append(energies,self.G.flatten())
            )
        self.estimates[0] += denominator
        psi.copy_historic_wfn()
        psi.copy_bp_wfn(psi_bp)

    def update_ghf(self, system, qmc, trial, psi, step, free_projection=False):
        """Calculate back-propagated estimates for GHF walkers.

        Parameters
        ----------
        system : system object in general.
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
        if step % self.nmax != 0:
            return
        psi_bp = pauxy.propagation.back_propagate_ghf(system, psi.walkers, trial,
                                                      self.nstblz, self.BT2,
                                                      self.dt)
        denominator = sum(wnm.weight for wnm in psi.walkers)
        nup = system.nup
        for i, (wnm, wb) in enumerate(zip(psi.walkers, psi_bp)):
            construct_multi_ghf_gab(wb.phi, wnm.phi_old, wb.weights, wb.Gi, wb.ots)
            # note that we are abusing the weights variable from the multighf
            # walker to store the reorthogonalisation factors.
            weights = wb.weights * trial.coeffs * wb.ots
            denom = sum(weights)
            energies = numpy.array(list(local_energy_ghf(system, wb.Gi, weights, denom)))
            self.G = numpy.einsum('i,ijk->jk', weights, wb.Gi) / denom
            self.estimates[1:]= (
                self.estimates[1:] + wnm.weight*numpy.append(energies,self.G.flatten())
            )
        self.estimates[0] += denominator
        psi.copy_historic_wfn()
        psi.copy_bp_wfn(psi_bp)

    def calculate_weight_factor(self, walker):
        """Compute reweighting factors for back propagation.

        Used with phaseless aproximation.

        Parameters
        ----------
        walker : walker object
            Current walker.

        Returns
        -------
        factor : complex
            Reweighting factor.
        """
        configs, cos_fac, weight_fac = walker.field_configs.get_block()
        factor = 1.0 + 0j
        for (w, c) in zip(weight_fac, cos_fac):
            factor *= w[0]
            if (self.restore_weights == "full"):
                factor /= c[0]
        return factor

    def print_step(self, comm, nprocs, step, nmeasure=1):
        """Print back-propagated estimates to file.

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
        if step != 0 and step % self.nmax == 0:
            comm.Reduce(self.estimates, self.global_estimates, op=mpi_sum)
            if comm.Get_rank() == 0:
                self.output.push(self.global_estimates[:self.nreg]/(nprocs))
                if self.rdm:
                    rdm = self.global_estimates[self.nreg:].reshape(self.G.shape)/(nprocs)
                    self.dm_output.push(rdm)
            self.zero()

    def zero(self):
        """Zero (in the appropriate sense) various estimator arrays."""
        self.estimates[:] = 0
        self.global_estimates[:] = 0


class ITCF(object):
    """Class for computing ITCF estimates.

    Parameters
    ----------
    itcf : dict
        Input options for ITCF estimates :

            - tmax : float
                Maximum value of imaginary time to calculate ITCF to.
            - stable : bool
                If True use the stabalised algorithm of Feldbacher and Assad.
            - mode : string / list
                How much of the ITCF to save to file:
                    'full' : print full ITCF.
                    'diagonal' : print diagonal elements of ITCF.
                    elements : list : print select elements defined from list.
            - kspace : bool
                If True evaluate correlation functions in momentum space.

    root : bool
        True if on root/master processor.
    h5f : :class:`h5py.File`
        Output file object.
    qmc : :class:`pauxy.state.QMCOpts` object.
        Container for qmc input options.
    system : system object
        System object.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    dtype : complex or float
        Output type.
    BT2 : :class:`numpy.ndarray`
        One-body propagator for back propagation.

    Attributes
    ----------
    nmax : int
        Number of back propagation steps to perform.
    header : list
        Header sfor back propagated estimators.
    key : dict
        Explanation of spgf data structure.
    spgf : :class:`numpy.ndarray`
        Storage for single-particle greens function (SPGF).
    spgf_global : :class:`numpy.ndarray`
        Store for ITCF accross all processors.
    rspace_unit : :class:`pauxy.estimators.H5EstimatorHelper`
        Output dataset for real space itcfs.
    kspace_unit : :class:`pauxy.estimators.H5EstimatorHelper`
        Output dataset for real space itcfs.
    """

    def __init__(self, itcf, qmc, trial, root, h5f, nbasis, dtype, nbp, BT2):
        self.stable = itcf.get('stable', True)
        self.tmax = itcf.get('tmax', 0.0)
        self.mode = itcf.get('mode', 'full')
        self.nmax = int(self.tmax/qmc.dt)
        self.nprop_tot = self.nmax + nbp
        self.nstblz = qmc.nstblz
        self.BT2 = BT2
        self.kspace = itcf.get('kspace', False)
        # self.spgf(i,j,k,l,m) gives the (l,m)th element of the spin-j(=0 for up
        # and 1 for down) k-ordered(0=greater,1=lesser) imaginary time green's
        # function at time i.
        # +1 in the first dimension is for the green's function at time tau = 0.
        self.spgf = numpy.zeros(shape=(self.nmax+1, 2, 2, nbasis, nbasis),
                                dtype=trial.G.dtype)
        self.spgf_global = numpy.zeros(shape=self.spgf.shape,
                                       dtype=trial.G.dtype)
        if trial.type == "GHF":
            self.I = numpy.identity(trial.psi.shape[1], dtype=trial.psi.dtype)
            self.initial_greens_function = self.initial_greens_function_ghf
            self.accumulate = self.accumulate_ghf
            self.back_propagate_single = pauxy.propagation.back_propagate_single_ghf
            self.construct_propagator_matrix = pauxy.propagation.construct_propagator_matrix_ghf
            if self.stable:
                self.increment_tau = self.increment_tau_ghf_stable
            else:
                self.increment_tau = self.increment_tau_ghf_unstable
        else:
            self.I = numpy.identity(trial.psi.shape[0], dtype=trial.psi.dtype)
            self.initial_greens_function = self.initial_greens_function_uhf
            self.accumulate = self.accumulate_uhf
            self.back_propagate_single = pauxy.propagation.back_propagate_single
            self.construct_propagator_matrix = pauxy.propagation.construct_propagator_matrix
            if self.stable:
                self.increment_tau = self.increment_tau_uhf_stable
            else:
                self.increment_tau = self.increment_tau_uhf_unstable
        if self.stable:
            self.calculate_spgf = self.calculate_spgf_stable
        else:
            self.calculate_spgf = self.calculate_spgf_unstable
        self.keys = [['up', 'down'], ['greater', 'lesser']]
        # I don't like list indexing so stick with numpy.
        if root:
            if self.mode == 'full':
                shape = (qmc.nsteps//(self.nmax),) + self.spgf.shape
            elif self.mode == 'diagonal':
                shape = (qmc.nsteps//(self.nmax), self.nmax+1, 2, 2, nbasis)
            else:
                shape = (qmc.nsteps//(self.nmax), self.nmax+1, 2, 2, len(self.mode))
            spgfs = h5f.create_group('single_particle_greens_function')
            self.rspace_unit = H5EstimatorHelper(spgfs, 'real_space', shape,
                                                 self.spgf.dtype)
            if self.kspace:
                self.kspace_unit = H5EstimatorHelper(spgfs, 'k_space', shape,
                                                     self.spgf.dtype)

    def update(self, system, qmc, trial, psi, step, free_projection=False):
        """Update estimators

        Parameters
        ----------
        system : system object in general.
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
        if step % self.nprop_tot == 0:
            self.calculate_spgf(system, psi, trial)

    def calculate_spgf_unstable(self, system, psi, trial):
        r"""Calculate imaginary time single-particle green's function.

        This uses the naive unstable algorithm.

        On return the spgf estimator array will have been updated.

        Parameters
        ----------
        system : system object in general.
            Container for model input options.
        psi : :class:`pauxy.walkers.Walkers` object
            CPMC wavefunction.
        trial : :class:`pauxy.trial_wavefunction.X' object
            Trial wavefunction class.
        """

        nup = system.nup
        denom = sum(w.weight for w in psi.walkers)
        M = system.nbasis
        for ix, w in enumerate(psi.walkers):
            # 1. Construct psi_left for first step in algorithm by back
            # propagating the input back propagated left hand wfn.
            # Note we use the first nmax fields for estimating the ITCF.
            configs = w.field_configs.get_superblock()[0]
            self.back_propagate_single(w.phi_bp, configs, w.weights,
                                       system, self.nstblz, self.BT2)
            (Ggr, Gls) = self.initial_greens_function(w.phi_bp,
                                                      w.phi_init,
                                                      trial, nup,
                                                      w.weights)
            # 2. Calculate G(n,n). This is the equal time Green's function at
            # the step where we began saving auxilary fields (constructed with
            # psi_left back propagated along this path.)
            self.accumulate(0, w.weight, Ggr, Gls, M)
            # 3. Construct ITCF by moving forwards in imaginary time from time
            # slice n along our auxiliary field path.
            for (ic, c) in enumerate(configs):
                # B takes the state from time n to time n+1.
                B = self.construct_propagator_matrix(system, self.BT2, c)
                (Ggr, Gls) = self.increment_tau(Ggr, Gls, B)
                self.accumulate(ic+1, w.weight, Ggr, Gls, M)
        self.spgf = self.spgf / denom
        # copy current walker distribution to initial (right hand) wavefunction
        # for next estimate of ITCF
        psi.copy_init_wfn()

    def calculate_spgf_stable(self, system, psi, trial):
        """Calculate imaginary time single-particle green's function.

        This uses the stable algorithm as outlined in:
        Feldbacher and Assad, Phys. Rev. B 63, 073105.

        On return the spgf estimator array will have been updated.

        Parameters
        ----------
        system : system object in general.
            Container for model input options.
        psi : :class:`pauxy.walkers.Walkers` object
            CPMC wavefunction.
        trial : :class:`pauxy.trial_wavefunction.X' object
            Trial wavefunction class.
        """

        nup = system.nup
        denom = sum(w.weight for w in psi.walkers)
        M = system.nbasis
        for ix, w in enumerate(psi.walkers):
            Ggr = numpy.identity(self.I.shape[0], dtype=self.I.dtype)
            Gls = numpy.identity(self.I.shape[0], dtype=self.I.dtype)
            # 1. Construct psi_L for first step in algorithm by back
            # propagating the input back propagated left hand wfn.
            # Note we use the first itcf_nmax fields for estimating the ITCF.
            # We store for intermediate back propagated left-hand wavefunctions.
            # This leads to more stable equal time green's functions compared to
            # that found by multiplying psi_L^n by B^{-1}(x^(n)) factors.
            configs = w.field_configs.get_superblock()[0]
            psi_Ls = self.back_propagate_single(w.phi_bp, configs, w.weights,
                                                system, self.nstblz, self.BT2,
                                                store=True)
            # 2. Calculate G(n,n). This is the equal time Green's function at
            # the step where we began saving auxilary fields (constructed with
            # psi_L back propagated along this path.)
            (Ggr_nn, Gls_nn) = self.initial_greens_function(w.phi_bp,
                                                            w.phi_init,
                                                            trial, nup,
                                                            w.weights)
            self.accumulate(0, w.weight, Ggr_nn, Gls_nn, M)
            # 3. Construct ITCF by moving forwards in imaginary time from time
            # slice n along our auxiliary field path.
            for (ic, c) in enumerate(configs):
                # B takes the state from time n to time n+1.
                B = self.construct_propagator_matrix(system, self.BT2, c)
                # G is the cumulative product of stabilised short-time ITCFs.
                # The first term in brackets is the G(n+1,n) which should be
                # well conditioned.
                (Ggr, Gls) = self.increment_tau(Ggr, Gls, B, Ggr_nn, Gls_nn)
                self.accumulate(ic+1, w.weight, Ggr, Gls, M)
                # Construct equal-time green's function shifted forwards along
                # the imaginary time interval. We need to update |psi_L> =
                # (B(c)^{dagger})^{-1}|psi_L> and |psi_R> = B(c)|psi_R>, where c
                # is the current configution in this loop. Note that we store
                # |psi_L> along the path, so we don't need to remove the
                # propagator matrices.
                L = psi_Ls[len(psi_Ls)-ic-1]
                pauxy.propagation.propagate_single(w.phi_init, system, B)
                if ic != 0 and ic % self.nstblz == 0:
                    (w.phi_init[:,:nup], R) = pauxy.utils.reortho(w.phi_init[:,:nup])
                    (w.phi_init[:,nup:], R) = pauxy.utils.reortho(w.phi_init[:,nup:])
                (Ggr_nn, Gls_nn) = self.initial_greens_function(L, w.phi_init,
                                                                trial, nup,
                                                                w.weights)
        self.spgf = self.spgf / denom
        # copy current walker distribution to initial (right hand) wavefunction
        # for next estimate of ITCF
        psi.copy_init_wfn()

    def initial_greens_function_uhf(self, A, B, trial, nup, weights):
        """Compute initial green's function at timestep n for UHF wavefunction.

        Here we actually compute the equal-time green's function:

        .. math::

            G_{ij} = \langle c_i c_j^{\dagger} \rangle
        
        Parameters
        ----------
        A : :class:`numpy.ndarray`
            Left hand wavefunction for green's function.
        B : :class:`numpy.ndarray`
            Left hand wavefunction for green's function.
        trial : :class:`pauxy.trial_wavefunction.X' object
            Trial wavefunction class.
        nup : int
            Number of up electrons.
        weight : :class:`numpy.ndarray`
            Any GS orthogonalisation factors which need to be included.

        Returns
        -------
        G_nn : :class:`numpy.ndarray`
            Green's function.
        """
        Ggr_up = self.I - gab(A[:,:nup], B[:,:nup])
        Ggr_down = self.I - gab(A[:,nup:], B[:,nup:])
        Gls_up = self.I - Ggr_up
        Gls_down = self.I - Ggr_down
        return (numpy.array([Ggr_up, Ggr_down]), numpy.array([Gls_up, Gls_down]))

    def initial_greens_function_ghf(self, A, B, trial, nup, weights):
        """Compute initial green's function at timestep n for GHF wavefunction.

        Here we actually compute the equal-time green's function:

        .. math::

            G_{ij} = \langle c_i c_j^{\dagger} \rangle
        
        Parameters
        ----------
        A : :class:`numpy.ndarray`
            Left hand wavefunction for green's function.
        B : :class:`numpy.ndarray`
            Left hand wavefunction for green's function.
        trial : :class:`pauxy.trial_wavefunction.X' object
            Trial wavefunction class.
        nup : int
            Number of up electrons.
        weight : :class:`numpy.ndarray`
            Any GS orthogonalisation factors which need to be included.

        Returns
        -------
        G_nn : :class:`numpy.ndarray`
            Green's function.
        """
        GAB = construct_multi_ghf_gab_back_prop(A, B, trial.coeffs, weights)
        Ggr = self.I - GAB
        Gls = self.I - Ggr
        return (Ggr, Gls)

    def accumulate_uhf(self, idx, weight, Ggr, Gls, nbasis):
        """Accumulate ITCF for UHF wavefunction.

        Parameters
        ----------
        idx : int
            Time index.
        weight : float
            Walker weight.
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Gls : :class:`numpy.ndarray`
            Lesser ITCF.
        nbasis : int
            Number of basis functions.
        """
        self.spgf[idx,0,0] += weight*Ggr[0].real
        self.spgf[idx,1,0] += weight*Ggr[1].real
        self.spgf[idx,0,1] += weight*Gls[0].real
        self.spgf[idx,1,1] += weight*Gls[1].real

    def accumulate_ghf(self, idx, weight, Ggr, Gls, nbasis):
        """Accumulate ITCF for GHF wavefunction.

        Parameters
        ----------
        idx : int
            Time index.
        weight : float
            Walker weight.
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Gls : :class:`numpy.ndarray`
            Lesser ITCF.
        nbasis : int
            Number of basis functions.
        """
        self.spgf[idx,0,0] += weight*Ggr[:nbasis,:nbasis].real
        self.spgf[idx,1,0] += weight*Ggr[nbasis:,nbasis:].real
        self.spgf[idx,0,1] += weight*Gls[:nbasis,:nbasis].real
        self.spgf[idx,1,1] += weight*Gls[nbasis:,nbasis:].real

    def increment_tau_ghf_unstable(self, Ggr, Gls, B, Gnn_gr=None, Gnn_ls=None):
        """Update ITCF to next time slice. Unstable algorithm, GHF format.

        Parameters
        ----------
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Lesser ITCF.
        Gls : :class:`numpy.ndarray`
            Propagator matrix.
        G_nn_gr : :class:`numpy.ndarray`, not used
            Greater equal-time green's function.
        G_nn_ls : :class:`numpy.ndarray`, not used
            Lesser equal-time green's function.

        Returns
        -------
        Ggr : :class:`numpy.ndarray`
            Updated greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Updated lesser ITCF.
        """
        Ggr = B.dot(Ggr)
        Gls = Gls.dot(scipy.linalg.inv(B))
        return Ggr, Gls

    def increment_tau_uhf_unstable(self, Ggr, Gls, B, Gnn_gr=None, Gnn_ls=None):
        """Update ITCF to next time slice. Unstable algorithm, UHF format.

        Parameters
        ----------
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Lesser ITCF.
        Gls : :class:`numpy.ndarray`
            Propagator matrix.
        G_nn_gr : :class:`numpy.ndarray`, not used
            Greater equal-time green's function.
        G_nn_ls : :class:`numpy.ndarray`, not used
            Lesser equal-time green's function.

        Returns
        -------
        Ggr : :class:`numpy.ndarray`
            Updated greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Updated lesser ITCF.
        """
        Ggr[0] = B[0].dot(Ggr[0])
        Ggr[1] = B[1].dot(Ggr[1])
        Gls[0] = Gls[0].dot(scipy.linalg.inv(B[0]))
        Gls[1] = Gls[1].dot(scipy.linalg.inv(B[1]))
        return Ggr, Gls

    def increment_tau_uhf_stable(self, Ggr, Gls, B, Gnn_gr, Gnn_ls):
        """Update ITCF to next time slice. Stable algorithm, UHF format.

        Parameters
        ----------
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Lesser ITCF.
        Gls : :class:`numpy.ndarray`
            Propagator matrix.
        G_nn_gr : :class:`numpy.ndarray`
            Greater equal-time green's function.
        G_nn_ls : :class:`numpy.ndarray`
            Lesser equal-time green's function.

        Returns
        -------
        Ggr : :class:`numpy.ndarray`
            Updated greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Updated lesser ITCF.
        """
        Ggr[0] = (B[0].dot(Gnn_gr[0])).dot(Ggr[0])
        Ggr[1] = (B[1].dot(Gnn_gr[1])).dot(Ggr[1])
        Gls[0] = Gls[0].dot(Gnn_ls[0].dot(scipy.linalg.inv(B[0])))
        Gls[1] = Gls[0].dot(Gnn_ls[1].dot(scipy.linalg.inv(B[1])))
        return Ggr, Gls

    def increment_tau_ghf_stable(self, Ggr, Gls, B, Gnn_gr, Gnn_ls):
        """Update ITCF to next time slice. Stable algorithm, GHF format.

        Parameters
        ----------
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Lesser ITCF.
        Gls : :class:`numpy.ndarray`
            Propagator matrix.
        G_nn_gr : :class:`numpy.ndarray`
            Greater equal-time green's function.
        G_nn_ls : :class:`numpy.ndarray`
            Lesser equal-time green's function.

        Returns
        -------
        Ggr : :class:`numpy.ndarray`
            Updated greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Updated lesser ITCF.
        """
        Ggr = (B.dot(Gnn_gr)).dot(Ggr)
        Gls = (Gnn_ls.dot(scipy.linalg.inv(B))).dot(Gls)
        return Ggr, Gls

    def print_step(self, comm, nprocs, step, nmeasure=1):
        """Print ITCF to file.

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
        if step != 0 and step % self.nprop_tot == 0:
            comm.Reduce(self.spgf, self.spgf_global, op=mpi_sum)
            if comm.Get_rank() == 0:
                self.to_file(self.rspace_unit, self.spgf_global/nprocs)
                if self.kspace:
                    M = self.spgf.shape[-1]
                    # FFT the real space Green's function.
                    # Todo : could just use numpy.fft.fft....
                    # spgf_k = numpy.einsum('ik,rqpkl,lj->rqpij', self.P,
                    # spgf, self.P.conj().T) / M
                    spgf_k = numpy.fft.fft2(self.spgf_global)
                    if self.spgf.dtype == complex:
                        self.to_file(self.kspace_unit, spgf_k/nprocs)
                    else:
                        self.to_file(self.kspace_unit, spgf_k.real/nprocs)
            self.zero()

    def to_file(self, group, spgf):
        """Push ITCF to hdf5 group.

        Parameters
        ----------
        group: string
            HDF5 group name.
        spgf : :class:`numpy.ndarray`
            Single-particle Green's function (SPGF).
        """
        if self.mode == 'full':
            group.push(spgf)
        elif self.mode == 'diagonal':
            group.push(spgf.diagonal(axis1=3, axis2=4))
        else:
            group.push(numpy.array([g[mode] for g in spgf]))

    def zero(self):
        """Zero (in the appropriate sense) various estimator arrays."""
        self.spgf[:] = 0
        self.spgf_global[:] = 0

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

class H5EstimatorHelper(object):
    """Helper class for pushing data to hdf5 dataset of fixed length.

    Parameters
    ----------
    h5f : :class:`h5py.File`
        Output file object.
    name : string
        Dataset name.
    shape : tuple 
        Shape of output data.
    dtype : type 
        Output data type.

    Attributes
    ----------
    store : :class:`h5py.File.DataSet`
        Dataset object.
    index : int
        Counter for incrementing data. 
    """
    def __init__(self, h5f, name, shape, dtype):
        self.store = h5f.create_dataset(name, shape, dtype=dtype)
        self.index = 0

    def push(self, data):
        """Push data to dataset.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            Data to push.
        """
        self.store[self.index] = data
        self.index = self.index + 1

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
