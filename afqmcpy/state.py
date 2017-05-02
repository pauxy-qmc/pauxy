import random
import json
import subprocess
import sys
import time
import json
import afqmcpy.hubbard as hubbard
import afqmcpy.trial_wave_function as trial_wave_function

class State:

    def __init__(self, model, qmc_opts):

        # Generic method option
        self.method = qmc_opts['method']
        self.nwalkers = qmc_opts['nwalkers']
        self.dt = qmc_opts['dt']
        self.nsteps = qmc_opts['nsteps']
        self.nmeasure = qmc_opts['nmeasure']
        self.temp = qmc_opts['temperature']
        if model['name'] == 'Hubbard':
            # sytem packages all generic information + model specific information.
            self.system = hubbard.Hubbard(model)
            self.gamma = np.arccosh(np.exp(0.5*self.dt*self.system.U))
            self.auxf = np.array([[np.exp(self.gamma), np.exp(-self.gamma)],
                                  [np.exp(-self.gamma), np.exp(self.gamma)]])
            # self.auxf = self.auxf * np.exp(-0.5*dt*self.system.U*self.system.ne)
            # Constant energy factor emerging from HS transformation.
            self.cfac = 0.5*self.system.U*self.system.ne
            if self.method ==  'CPMC':
                self.projectors = hubbard.Projectors(self.system, self.dt)

        (self.psi_trial, self.sp_eigs) = trial_wave_function.free_electron(self.system)
        random.seed(qmc_opts['rng_seed'])
        self.pack_metadata
        # Handy to keep original dicts so they can be printed at run time.
        self.model = model
        self.qmc_opts = qmc_opts


    def write_json(self):

        # Combine some metadata in dicts so it can be easily printed/read.
        calc_info =  {'sha1': get_git_revision_hash(),
                      'Run time': time.asctime()}
        derived = {'sp_eigv': self.sp_eigs}
        print ("# Input options: ")
        print (json.dumps({'calculation': calc_info,
                           'derived': derived,
                           'model': self.model,
                           'qmc_options': self.qmc_opts}, indent=4))
        print ("# End of input options ")


def get_git_revision_hash():
    '''Return git revision.

    Adapted from: http://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script

Returns
-------
sha1 : string
    git hash with -dirty appended if uncommitted changes.
'''

    src = [s for s in sys.path if 'afqmcpy' in s][0]

    sha1 = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=src).strip()
    suffix = subprocess.check_output(['git', 'status',
                                     '--porcelain',
                                     'afqmcpy/'],
                                     cwd=src).strip()
    if suffix:
        return sha1.decode('utf-8') + '-dirty'
    else:
        return sha1.decode('utf-8')
