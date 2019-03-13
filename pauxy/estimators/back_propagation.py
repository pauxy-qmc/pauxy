import h5py
import numpy
try:
    from mpi4py import MPI
    mpi_sum = MPI.SUM
except ImportError:
    mpi_sum = None
import sys
from pauxy.estimators.utils import H5EstimatorHelper
from pauxy.estimators.greens_function import gab
from pauxy.estimators.mixed import local_energy
from pauxy.propagation.generic import back_propagate_generic
from pauxy.propagation.planewave import back_propagate_planewave
import pauxy.propagation.hubbard

class BackPropagation(object):
    """Class for computing back propagated estimates.

    Parameters
    ----------
    bp : dict
        Input options for BP estimates.
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
        self.tau_bp = bp.get('tau_bp', 0)
        self.nmax = int(self.tau_bp/qmc.dt)
        self.header = ['iteration', 'Weight', 'E', 'E1b', 'E2b']
        self.calc_one_rdm = bp.get('one_rdm', False)
        self.calc_two_rdm = bp.get('two_rdm', None)
        self.nreg = len(self.header[1:])
        self.eval_energy = bp.get('evaluate_energy', True)
        self.G = numpy.zeros(trial.G.shape, dtype=trial.G.dtype)
        self.nstblz = qmc.nstblz
        self.BT2 = BT2
        self.restore_weights = bp.get('restore_weights', None)
        self.dt = qmc.dt
        dms_size = self.G.size
        # Abuse of language for the moment. Only accumulates S(k) for UEG.
        # TODO: Add functionality to accumulate 2RDM?
        if self.calc_two_rdm is not None:
            if self.calc_two_rdm == "structure_factor":
                two_rdm_shape = (2,2,len(system.qvecs),)
            self.two_rdm = numpy.zeros(two_rdm_shape,
                                       dtype=numpy.complex128)
            dms_size += self.two_rdm.size
        else:
            self.two_rdm = None
        self.estimates = numpy.zeros(self.nreg+dms_size, dtype=dtype)
        self.global_estimates = numpy.zeros(self.nreg+dms_size,
                                            dtype=dtype)
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
            if self.calc_one_rdm:
                one_rdm_shape = (qmc.nsteps//self.nmax+1,)+self.G.shape
                self.one_rdm_output = H5EstimatorHelper(energies, 'one_rdm',
                                                        one_rdm_shape,
                                                        trial.G.dtype)
            if self.calc_two_rdm is not None:
                name = 'two_rdm'
                two_rdm_shape = (qmc.nsteps//self.nmax+1,) + two_rdm_shape
                self.two_rdm_output = H5EstimatorHelper(energies, name,
                                                        two_rdm_shape, dtype)

        if trial.type == 'GHF':
            self.update = self.update_ghf
            self.back_propagate = pauxy.propagation.hubbard.back_propagate_ghf
        else:
            self.update = self.update_uhf
            if system.name == "Generic":
                self.back_propagate = back_propagate_generic
            elif system.name == "UEG":
                self.back_propagate = back_propagate_planewave
            else:
                self.back_propagate = pauxy.propagation.hubbard.back_propagate

    def update_uhf(self, system, qmc, trial, psi, step, free_projection=False):
        """Calculate back-propagated estimates for RHF/UHF walkers.

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
        nup = system.nup
        for i, wnm in enumerate(psi.walkers):
            phi_bp = trial.psi.copy()
            # TODO: Fix for ITCF.
            self.back_propagate(phi_bp, wnm.stack, system,
                                self.nstblz, self.BT2, self.dt)
            self.G[0] = gab(phi_bp[:,:nup], wnm.phi_old[:,:nup])
            self.G[1] = gab(phi_bp[:,nup:], wnm.phi_old[:,nup:])
            if self.eval_energy:
                eloc = local_energy(system, self.G, opt=False,
                                    two_rdm=self.two_rdm)
                energies = numpy.array(list(eloc))
            else:
                energies = numpy.zeros(3)
            if self.restore_weights is not None:
                wfac = wnm.stack.get_wfac()
                if self.restore_weights == "full":
                    wfac = wfac[0]*wfac[1]
                else:
                    wfac  = wfac[1]
                weight = wnm.weight * wfac
            else:
                weight = wnm.weight
            self.estimates[0] += weight
            self.estimates[1:self.nreg] += (
                    self.estimates[1:self.nreg] + weight*energies
                    )
            start = self.nreg
            end = self.nreg + self.G.size
            self.estimates[start:end] += weight*self.G.flatten().real
            if self.calc_two_rdm is not None:
                start = end
                end = end + self.two_rdm.size
                self.estimates[start:end] += weight*self.two_rdm.flatten().real
            wnm.stack.reset()
        psi.copy_historic_wfn()

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
        print(" ***** Back Propagation with GHF is broken.")
        sys.exit()
        psi_bp = self.back_propagate(system, psi.walkers, trial,
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

    def print_step(self, comm, nprocs, step, nmeasure=1, free_projection=False):
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
            if comm.rank == 0:
                self.output.push(self.global_estimates[:self.nreg])
                if self.calc_one_rdm:
                    start = self.nreg
                    end = self.nreg + self.G.size
                    rdm = self.global_estimates[self.nreg:end].reshape(self.G.shape)
                    self.one_rdm_output.push(rdm)
                if self.calc_two_rdm:
                    start = self.nreg + self.G.size
                    rdm = self.global_estimates[start:].reshape(self.two_rdm.shape)
                    self.two_rdm_output.push(rdm)
            self.zero()

    def zero(self):
        """Zero (in the appropriate sense) various estimator arrays."""
        self.estimates[:] = 0
        self.global_estimates[:] = 0
