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
import pauxy.pop_control
import pauxy.systems

class CPMC:
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

    Attributes
    ----------
    system : :class:`pauxy.hubbard.Hubbard` / system object in general.
        Container for model input options.
    qmc : :class:`pauxy.state.QMCOpts` object.
        Container for qmc input options.
    uuid : string
        Simulation state uuid.
    seed : int
        RNG seed. This is set during initialisation but is useful to output in
        json_string.
    root : bool
        If True we are on the root / master processor.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    propagators : :class:`pauxy.propagation.Projectors` object
        Container for system specific propagation routines.
    json_string : string
        String containing all input options and certain derived options.
    """
    def __init__(self, model, qmc_opts, estimates, trial, parallel=False):
        self.system = pauxy.systems.get_system(model, qmc_opts['dt'])
        self.qmc = pauxy.qmc.QMCOpts(qmc_opts, self.system)
        # Store input dictionaries for the moment.
        self.uuid = str(uuid.uuid1())
        self.sha1 =  pauxy.utils.get_git_revision_hash()
        self.seed = qmc_opts['rng_seed']
        # Hack - this is modified on initialisation.
        self.root = True
        self.nprocs = 1
        self.init_time = time.time()
        self.trial = (
            pauxy.trial_wavefunction.get_trial_wavefunction(trial, self.system,
                                                            self.qmc.cplx,
                                                            parallel)
        )
        if self.qmc.hubbard_stratonovich == 'discrete':
            self.propagators = pauxy.propagation.DiscreteHubbard(self.qmc,
                                                                   self.system,
                                                                   self.trial)
        elif self.qmc.hubbard_stratonovich == "continuous":
            self.propagators = pauxy.propagation.ContinuousHubbard(self.qmc,
                                                                     self.system,
                                                                     self.trial)
        elif self.qmc.hubbard_stratonovich  == "generic_continuous":
            self.propagators = pauxy.propagation.GenericContinuous(self.qmc,
                                                                     self.system,
                                                                     self.trial)
        # Handy to keep original dicts so they can be printed at run time.
        if not parallel:
            self.estimators = (
                pauxy.estimators.Estimators(estimates,
                                            self.root,
                                            self.qmc,
                                            self.system,
                                            self.trial,
                                            self.propagators.BT_BP)
            )
            self.psi = pauxy.walker.Walkers(self.system, self.trial,
                                            self.qmc.nwalkers,
                                            self.estimators.nprop_tot,
                                            self.estimators.nbp)
            json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')
            json_string = json.dumps(pauxy.utils.serialise(self, verbose=1),
                                     sort_keys=False, indent=4)
            self.estimators.h5f.create_dataset('metadata',
                                               data=numpy.array([json_string],
                                               dtype=object),
                                               dtype=h5py.special_dtype(vlen=str))
            self.estimators.estimators['mixed'].print_key()
            self.estimators.estimators['mixed'].print_header()

    # Remove - each class should have a serialiser
    def write_json(self, model, qmc_opts, estimates):
        r"""Print out state object information to string.

        Parameters
        ----------
        print_function : method, optional
            How to print state information, e.g. to std out or file. Default : print.
        eol : string, optional
            String to append to output, e.g., '\n', Default : ''.
        verbose : bool, optional
            How much information to print. Default : True.
        """

        # Combine some metadata in dicts so it can be easily printed/read.
        calc_info =  {
            'sha1': pauxy.utils.get_git_revision_hash(),
            'Run time': time.asctime(),
            'uuid': self.uuid
        }
        trial_wavefunction = {
            'name': self.trial.__class__.__name__,
            'sp_eigv': self.trial.eigs.round(6).tolist(),
            'initialisation_time': round(self.trial.initialisation_time, 5),
            'trial_energy': self.trial.etrial,
        }
        # http://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module
        # ugh
        json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')
        info = {
            'calculation': calc_info,
            'model': model,
            'qmc_options': qmc_opts,
            'trial_wavefunction': trial_wavefunction,
            'estimates': estimates,
        }
        # Note that we require python 3.6 to print dict in ordered fashion.
        md = json.dumps(info, sort_keys=False, indent=4)
        return (md)

    def run(self, psi=None, comm=None):
        """Perform CPMC simulation on state object.

        Parameters
        ----------
        state : :class:`pauxy.state.State` object
            Model and qmc parameters.
        psi : list of :class:`pauxy.walker.Walker` objects
            Initial wavefunction / distribution of walkers.
        comm : MPI communicator
        """
        if psi is not None:
            self.psi = psi
        (E_T, ke, pe) = self.psi.walkers[0].local_energy(self.system)
        self.propagators.mean_local_energy = E_T.real
        # Calculate estimates for initial distribution of walkers.
        self.estimators.estimators['mixed'].update(self.system, self.qmc,
                                                   self.trial, self.psi, 0)
        # Print out zeroth step for convenience.
        self.estimators.estimators['mixed'].print_step(comm, self.nprocs, 0, 1)

        for step in range(1, self.qmc.nsteps+1):
            for w in self.psi.walkers:
                # Want to possibly allow for walkers with negative / complex weights
                # when not using a constraint. I'm not so sure about the criteria
                # for complex weighted walkers.
                if abs(w.weight) > 1e-8:
                    self.propagators.propagate_walker(w, self.system, self.trial)
                # Constant factors
                w.weight = w.weight*exp(self.qmc.dt*E_T.real)
                # Add current (propagated) walkers contribution to estimates.
            # calculate estimators
            self.estimators.update(self.system, self.qmc,
                                   self.trial, self.psi, step)
            if step%self.qmc.nstblz == 0:
                self.psi.orthogonalise(self.trial, self.qmc.importance_sampling)
            if step%self.qmc.nupdate_shift == 0:
                # Todo: proj energy function
                E_T = self.estimators.estimators['mixed'].projected_energy()
            if step%self.qmc.nmeasure == 0:
                self.estimators.print_step(comm, self.nprocs, step,
                                           self.qmc.nmeasure)
            if step < self.qmc.nequilibrate:
                # Update local energy bound.
                self.propagators.mean_local_energy = E_T
            if step%self.qmc.npop_control == 0:
                pauxy.pop_control.comb(self.psi, self.qmc.nwalkers)

    def finalise(self):
        if self.root:
            print ("# End Time: %s"%time.asctime())
            print ("# Running time : %.6f seconds"%(time.time()-self.init_time))
            if self.estimators.back_propagation:
                self.estimators.h5f.close()
