"""Driver to perform AFQMC calculation"""
import sys
import json
import time
import numpy
import warnings
import uuid
from math import exp
import copy
import h5py
from pauxy.estimators.handler import Estimators
from pauxy.qmc.options import QMCOpts
from pauxy.systems.utils import get_system
from pauxy.thermal.hubbard import Discrete
from pauxy.trial_density_matrices.onebody import OneBody
from pauxy.utils.misc import get_git_revision_hash, serialise
from pauxy.walkers.handler import Walkers


class ThermalAFQMC(object):
    """AFQMC driver.

    Non-zero temperature AFQMC using open ended random walk.

    Parameters
    ----------
    model : dict
        Input parameters for model system.
    qmc_opts : dict
        Input options relating to qmc parameters.
    estimates : dict
        Input options relating to what estimator to calculate.
    trial : dict
        Input options relating to trial wavefunction.
    propagator : dict
        Input options relating to propagator.
    parallel : bool
        If true we are running in parallel.
    verbose : bool
        If true we print out additional setup information.

    Attributes
    ----------
    uuid : string
        Simulation state uuid.
    sha1 : string
        Git hash.
    seed : int
        RNG seed. This is set during initialisation in calc.
    root : bool
        If true we are on the root / master processor.
    nprocs : int
        Number of processors.
    rank : int
        Processor id.
    cplx : bool
        If true then most numpy arrays are complex valued.
    init_time : float
        Calculation initialisation (cpu) time.
    init_time : float
        Human readable initialisation time.
    system : system object.
        Container for model input options.
    qmc : :class:`pauxy.state.QMCOpts` object.
        Container for qmc input options.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    propagators : :class:`pauxy.propagation.Projectors` object
        Container for system specific propagation routines.
    estimators : :class:`pauxy.estimators.Estimators` object
        Estimator handler.
    psi : :class:`pauxy.walkers.Walkers` object
        Stores walkers which sample the partition function.
    """

    def __init__(self, model, qmc_opts, estimates,
                 trial, propagator, parallel=False,
                 verbose=False):
        # 1. Environment attributes
        self.uuid = str(uuid.uuid1())
        self.sha1 = get_git_revision_hash()
        self.seed = qmc_opts['rng_seed']
        # Hack - this is modified later if running in parallel on
        # initialisation.
        self.root = True
        self.nprocs = 1
        self.rank = 1
        self.init_time = time.time()
        self.run_time = time.asctime(),
        # 2. Calculation objects.
        self.system = get_system(model, qmc_opts['dt'], verbose)
        self.qmc = QMCOpts(qmc_opts, self.system, verbose)
        self.qmc.ntime_slices = int(qmc.beta/qmc.dt)
        self.cplx = self.determine_dtype(propagator, self.system)
        self.trial = OneBody(system, qmc.beta, qmc.dt, verbose)
        self.propagators = get_propagator(propagator, self.qmc, self.system,
                                          self.trial, verbose)
        if not parallel:
            self.estimators = (
                Estimators(estimates, self.root, self.qmc, self.system,
                           self.trial, self.propagators.BT_BP, verbose)
            )
            self.psi = Walkers(self.system, self.trial, self.qmc.nwalkers,
                               self.estimators.nprop_tot,
                               self.estimators.nbp, verbose)
            json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')
            json_string = json.dumps(serialise(self, verbose=1),
                                     sort_keys=False, indent=4)
            self.estimators.h5f.create_dataset('metadata',
                                               data=numpy.array([json_string],
                                                                dtype=object),
                                               dtype=h5py.special_dtype(vlen=str))
            self.estimators.estimators['mixed'].print_key()
            self.estimators.estimators['mixed'].print_header()

    def run(self, psi=None, comm=None):
        """Perform AFQMC simulation on state object using open-ended random walk.

        Parameters
        ----------
        state : :class:`pauxy.state.State` object
            Model and qmc parameters.
        psi : :class:`pauxy.walker.Walkers` object
            Initial wavefunction / distribution of walkers.
        comm : MPI communicator
        """
        if psi is not None:
            self.psi = psi
        (E_T, ke, pe) = self.psi.walkers[0].local_energy(self.system)
        # Calculate estimates for initial distribution of walkers.
        self.estimators.estimators['mixed'].update(self.system, self.qmc,
                                                   self.trial, self.psi, 0,
                                                   self.propagators.free_projection)
        # Print out zeroth step for convenience.
        if self.root:
            self.estimators.estimators['mixed'].print_key()
            self.estimators.estimators['mixed'].print_header()
        self.estimators.estimators['mixed'].print_step(comm, self.nprocs, 0, 1)

        for step in range(1, self.qmc.nsteps + 1):
            for ts in range(0, self.ntime_slices):
                for w in self.psi.walkers:
                    if abs(w.weight) > 1e-8:
                        self.propagators.propagate_walker(w, self.system, self.trial)
                if ts % self.qmc.npop_control == 0:
                    self.psi.pop_control(comm)
                if ts % self.qmc.nstblz == 0:
                    self.psi.recompute_greens_function(ts)
            self.estimators.update(self.system, self.qmc,
                                   self.trial, self.psi, step,
                                   self.propagators.free_projection)
            self.estimators.print_step(comm, self.nprocs, step,
                                       self.qmc.nmeasure)

    def finalise(self, verbose):
        """Tidy up.

        Parameters
        ----------
        verbose : bool
            If true print out some information to stdout.
        """
        if self.root:
            if self.estimators.back_propagation:
                self.estimators.h5f.close()
            if verbose:
                print("# End Time: %s" % time.asctime())
                print("# Running time : %.6f seconds" %
                      (time.time() - self.init_time))


    def determine_dtype(self, propagator, system):
        """Determine dtype for trial wavefunction and walkers.

        Parameters
        ----------
        propagator : dict
            Propagator input options.
        system : object
            System object.
        """
        hs_type = propagator.get('hubbard_stratonovich', 'discrete')
        continuous = 'continuous' in hs_type
        twist = system.ktwist.all() is not None
        return continuous or twist
