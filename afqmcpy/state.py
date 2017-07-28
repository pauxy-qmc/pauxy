
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

    def __init__(self, model, qmc_opts, trial):

        if model['name'] == 'Hubbard':
            # sytem packages all generic information + model specific information.
            self.system = hubbard.Hubbard(model, qmc_opts['dt'])
        self.qmc = QMCOpts(qmc_opts, self.system)
        self.back_propagation = qmc_opts.get('back_propagation', False)
        self.nback_prop = qmc_opts.get('nback_prop', 0)
        itcf_opts = qmc_opts.get('itcf', None)
        # repackage these options into class/dict
        self.itcf_nmax = 0
        if itcf_opts is not None:
            self.itcf = True
            self.itcf_stable = itcf_opts.get('stable', True)
            self.itcf_tmax = itcf_opts.get('tmax', 0.0)
            self.itcf_mode = itcf_opts.get('mode', 'full')
            self.itcf_nmax = int(self.itcf_tmax/self.qmc.dt)
            self.itcf_kspace = itcf_opts.get('kspace', False)
        else:
            self.itcf = False
        self.nprop_tot = max(1, self.itcf_nmax+self.nback_prop)
        self.uuid = str(uuid.uuid1())
        self.seed = qmc_opts['rng_seed']

        self.propagators = afqmcpy.propagation.Projectors(model['name'],
                                                          self.qmc.hubbard_stratonovich,
                                                          self.qmc.dt, self.system.T,
                                                          self.qmc.importance_sampling,
                                                          self.system.eks,
                                                          self.qmc.ffts)
        # effective hubbard U for UHF trial wavefunction.
        if trial['name'] == 'free_electron':
            self.trial = trial_wave_function.Free_Electron(self.system,
                                                           self.qmc.cplx,
                                                           trial)
        elif trial['name'] == 'UHF':
            self.trial = trial_wave_function.UHF(self.system,
                                                 self.qmc.cplx,
                                                 trial)
        elif trial['name'] == 'multi_determinant':
            self.trial = trial_wave_function.multi_det(self.system, self.cplx)
        # Handy to keep original dicts so they can be printed at run time.
        self.model = model
        self.qmc_opts = qmc_opts


    def write_json(self, print_function=print, eol='', eoll='\n',
                   verbose=True, encode=False):
        r"""Print out state object information.

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
        if verbose:
            info = {
                'calculation': calc_info,
                'model': self.model,
                'qmc_options': self.qmc_opts,
                'trial_wavefunction': trial_wavefunction,
            }
        else:
            info = {'calculation': calc_info,}
        # Note that we require python 3.6 to print dict in ordered fashion.
        first = '# Input options:' + eol
        last = eol + '# End of input options' + eoll
        md = json.dumps(info, sort_keys=False, indent=4)
        output_string = first + md + last
        if encode == True:
            output_string = output_string.encode('utf-8')
        print_function(output_string)

class QMCOpts:

    def __init__(self, inputs, system):
        self.method = inputs.get('method', None)
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
