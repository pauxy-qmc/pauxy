
from __future__ import print_function

import subprocess
import sys
import time
import json
import numpy
import uuid
import afqmcpy.hubbard as hubbard
import afqmcpy.trial_wave_function as trial_wave_function
import afqmcpy.propagation
import afqmcpy.hs_transform

class State:

    def __init__(self, model, qmc_opts, estimates, trial):

        if model['name'] == 'Hubbard':
            # sytem packages all generic information + model specific information.
            self.system = hubbard.Hubbard(model, qmc_opts['dt'])
        self.qmc = QMCOpts(qmc_opts, self.system)
        # Store input dictionaries for the moment.
        # Todo : output constructed derived objects.
        self.back_propagation = qmc_opts.get('back_propagation', False)
        self.uuid = str(uuid.uuid1())
        self.seed = qmc_opts['rng_seed']
        # Hack - this is modified on initialisation.
        self.root = True
        self.propagators = afqmcpy.propagation.Projectors(model['name'],
                                                          self.qmc.hubbard_stratonovich,
                                                          self.qmc.dt, self.system.T,
                                                          self.qmc.importance_sampling,
                                                          self.system.eks,
                                                          self.qmc.ffts)
        # effective hubbard U for UHF trial wavefunction.
        if trial['name'] == 'free_electron':
            self.trial = trial_wave_function.FreeElectron(self.system,
                                                           self.qmc.cplx,
                                                           trial)
        if trial['name'] == 'UHF':
            self.trial = trial_wave_function.UHF(self.system,
                                                 self.qmc.cplx,
                                                 trial)
        elif trial['name'] == 'multi_determinant':
            self.trial = trial_wave_function.MultiDeterminant(self.system,
                                                              self.cplx,
                                                              trial)
        self.propagators = afqmcpy.propagation.Projectors(model['name'],
                                                          self.hubbard_stratonovich,
                                                          self.qmc.dt, self.system.T,
                                                          self.qmc.importance_sampling,
                                                          self.system.eks,
                                                          self.qmc.ffts,
                                                          self.trial.name)
        self.local_energy_bound = (2.0/self.dt)**0.5
        self.mean_local_energy = 0
        # Handy to keep original dicts so they can be printed at run time.
        self.json_string = self.write_json(model, qmc_opts, estimates)
        print (self.json_string)

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
            'sha1': get_git_revision_hash(),
            'Run time': time.asctime(),
            'uuid': self.uuid
        }
        trial_wavefunction = {
            'name': self.trial.__class__.__name__,
            'sp_eigv': self.trial.eigs.round(6).tolist(),
            'initialisation_time': round(self.trial.initialisation_time, 5),
            'trial_energy': self.trial.emin,
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
        first = '# Input options:\n'
        last =  '\n# End of input options.'
        md = json.dumps(info, sort_keys=False, indent=4)
        return (first + md + last)

class QMCOpts:
    """Input options and certain constants / parameters derived from them.

    Initialised from a dict containing the following options, not all of which
    are required.

    Parameters/Attributes
    ---------------------
    method : string
        Which auxiliary field method are we using? Currently only CPMC is
        implemented.
    nwalkers : int
        Number of walkers to propagate in a simulation.
    dt : float
        Timestep.
    nsteps : int
        Total number of Monte Carlo steps to perform.
    nmeasure : int
        Frequency of energy measurements.
    nstblz : int
        Frequency of Gram-Schmidt orthogonalisation steps.
    npop_control : int
        Frequency of population control.
    temp : float
        Temperature. Currently not used.
    nequilibrate : int
        Number of steps used for equilibration phase. Only used to fix local
        energy bound when using phaseless approximation.
    importance_sampling : boolean
        Are we using importance sampling. Default True.
    hubbard_statonovich : string
        Which hubbard stratonovich transformation are we using. Currently the
        options are:
            discrete : Use the discrete Hirsch spin transformation.
            opt_continuous : Use the continuous transformation for the Hubbard
                model.
            generic : Use the generic transformation. Not currently implemented.
    ffts : boolean
        Use FFTS to diagonalise the kinetic energy propagator? Default False.
        This may speed things up for larger lattices.

    Derived Attributes
    ------------------
    cplx : boolean
        Do we require complex wavefunctions?
    mf_shift : float
        Mean field shift for continuous Hubbard-Stratonovich transformation.
    iut_fac : complex float
        Stores i*(U*dt)**0.5 for continuous Hubbard-Stratonovich transformation.
    ut_fac : float
        Stores (U*dt) for continuous Hubbard-Stratonovich transformation.
    mf_nsq : float
        Stores M * mf_shift for continuous Hubbard-Stratonovich transformation.
    local_energy_bound : float
        Energy pound for continuous Hubbard-Stratonovich transformation.
    mean_local_energy : float
        Estimate for mean energy for continuous Hubbard-Stratonovich transformation.
    """

    def __init__(self, inputs, system):
        self.method = inputs.get('method', 'CPMC')
        self.nwalkers = inputs.get('nwalkers', None)
        self.dt = inputs.get('dt', None)
        self.nsteps = inputs.get('nsteps', None)
        self.nmeasure = inputs.get('nmeasure', 10)
        self.nstblz = inputs.get('nstabilise', 10)
        self.npop_control = inputs.get('npop_control', 10)
        self.temp = inputs.get('temperature', None)
        self.nequilibrate = inputs.get('nequilibrate', int(1.0/self.dt))
        self.importance_sampling = inputs.get('importance_sampling', True)
        self.hubbard_stratonovich = inputs.get('hubbard_stratonovich',
                                                'discrete')
        self.ffts = inputs.get('kinetic_kspace', False)
        self.cplx = ('continuous' in self.hubbard_stratonovich
                     or system.ktwist.all() != 0)
        if self.hubbard_stratonovich == 'opt_continuous':
            # optimal mean-field shift for the hubbard model
            self.mf_shift = (system.nup+system.ndown) / float(system.nbasis)
            self.iut_fac = 1j*numpy.sqrt((system.U*self.dt))
            self.ut_fac = self.dt*system.U
            # Include factor of M! bad name
            self.mf_nsq = system.nbasis * self.mf_shift**2.0
        self.local_energy_bound = (2.0/self.dt)**0.5
        self.mean_local_energy = 0


def get_git_revision_hash():
    '''Return git revision.

    Adapted from: http://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script

Returns
-------
sha1 : string
    git hash with -dirty appended if uncommitted changes.
'''

    src = [s for s in sys.path if 'afqmcpy' in s][-1]

    sha1 = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=src).strip()
    suffix = subprocess.check_output(['git', 'status',
                                     '--porcelain',
                                     './afqmcpy'],
                                     cwd=src).strip()
    if suffix:
        return sha1.decode('utf-8') + '-dirty'
    else:
        return sha1.decode('utf-8')
