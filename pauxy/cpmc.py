"""Driver to perform CPMC calculation"""
import sys
import json
import time
import numpy
import warnings
import uuid
from math import exp
import copy
import h5py
import pauxy.qmc
import pauxy.walker
import pauxy.estimators
import pauxy.utils
import pauxy.systems


class CPMC(object):
    """CPMC driver.

    This object contains all the instances of the classes which parse input
    options.

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
    system : :class:`pauxy.hubbard.Hubbard` / system object in general.
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
        Walker handler. Stores the CPMC wavefunction.
    """

    def __init__(self, model, qmc_opts, estimates,
                 trial, propagator, parallel=False,
                 verbose=False):
        # 1. Environment attributes
        self.uuid = str(uuid.uuid1())
        self.sha1 = pauxy.utils.get_git_revision_hash()
        self.seed = qmc_opts['rng_seed']
        # Hack - this is modified later if running in parallel on
        # initialisation.
        self.root = True
        self.nprocs = 1
        self.rank = 1
        self.init_time = time.time()
        self.run_time = time.asctime(),
        # 2. Calculation objects.
        self.system = pauxy.systems.get_system(model, qmc_opts['dt'], verbose)
        self.qmc = pauxy.qmc.QMCOpts(qmc_opts, self.system, verbose)
        self.cplx = self.determine_dtype(propagator, self.system)
        self.trial = (
            pauxy.trial_wavefunction.get_trial_wavefunction(trial, self.system,
                                                            self.cplx,
                                                            parallel, verbose)
        )
        self.propagators = pauxy.propagation.get_propagator(propagator,
                                                            self.qmc,
                                                            self.system,
                                                            self.trial,
                                                            verbose)
        if not parallel:
            self.estimators = (
                pauxy.estimators.Estimators(estimates,
                                            self.root,
                                            self.qmc,
                                            self.system,
                                            self.trial,
                                            self.propagators.BT_BP,
                                            verbose)
            )
            self.psi = pauxy.walker.Walkers(self.system, self.trial,
                                            self.qmc.nwalkers,
                                            self.estimators.nprop_tot,
                                            self.estimators.nbp,
                                            verbose)
            json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')
            json_string = json.dumps(pauxy.utils.serialise(self, verbose=1),
                                     sort_keys=False, indent=4)
            self.estimators.h5f.create_dataset('metadata',
                                               data=numpy.array([json_string],
                                                                dtype=object),
                                               dtype=h5py.special_dtype(vlen=str))
            self.estimators.estimators['mixed'].print_key()
            self.estimators.estimators['mixed'].print_header()

    def run(self, psi=None, comm=None):
        """Perform CPMC simulation on state object.

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
        self.propagators.mean_local_energy = E_T.real
        # Calculate estimates for initial distribution of walkers.
        self.estimators.estimators['mixed'].update(self.system, self.qmc,
                                                   self.trial, self.psi, 0,
                                                   self.propagators.free_projection)
        # Print out zeroth step for convenience.
        self.estimators.estimators['mixed'].print_key()
        self.estimators.estimators['mixed'].print_header()
        self.estimators.estimators['mixed'].print_step(comm, self.nprocs, 0, 1)

        for step in range(1, self.qmc.nsteps + 1):
            for w in self.psi.walkers:
                # Want to possibly allow for walkers with negative / complex weights
                # when not using a constraint. I'm not so sure about the criteria
                # for complex weighted walkers.
                if abs(w.weight) > 1e-8 and w.alive:
                    self.propagators.propagate_walker(
                        w, self.system, self.trial)
                # Constant factors
                w.weight = w.weight * exp(self.qmc.dt * E_T.real)
            # calculate estimators
            self.estimators.update(self.system, self.qmc,
                                   self.trial, self.psi, step,
                                   self.propagators.free_projection)
            if step % self.qmc.nstblz == 0:
                self.psi.orthogonalise(self.trial,
                                       self.propagators.free_projection)
            if step % self.qmc.nupdate_shift == 0:
                E_T = self.estimators.estimators['mixed'].projected_energy()
            if step % self.qmc.nmeasure == 0:
                self.estimators.print_step(comm, self.nprocs, step,
                                           self.qmc.nmeasure)
            if step < self.qmc.nequilibrate:
                # Update local energy bound.
                self.propagators.mean_local_energy = E_T
            if step % self.qmc.npop_control == 0:
                self.psi.pop_control(comm, self.rank, self.nprocs)

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
