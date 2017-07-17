import json
from json import encoder
import subprocess
import sys
import time
import json
import numpy
import uuid as uuid
import afqmcpy.hubbard as hubbard
import afqmcpy.trial_wave_function as trial_wave_function
import afqmcpy.propagation
import afqmcpy.hs_transform

class State:

    def __init__(self, model, qmc_opts):

        # Generic method option
        self.method = qmc_opts['method']
        self.nwalkers = qmc_opts['nwalkers']
        self.dt = qmc_opts['dt']
        self.nsteps = qmc_opts['nsteps']
        self.nmeasure = qmc_opts['nmeasure']
        self.nstblz = qmc_opts.get('nstabilise', 10)
        self.npop_control = qmc_opts.get('npop_control')
        self.temp = qmc_opts['temperature']
        # number of steps to equilibrate simulation, default to tau = 1.
        self.nequilibrate = qmc_opts.get('nequilibrate', int(1.0/self.dt))
        self.importance_sampling = qmc_opts['importance_sampling']
        self.hubbard_stratonovich = qmc_opts.get('hubbard_stratonovich')
        self.ffts = qmc_opts.get('kinetic_kspace', False)
        self.back_propagation = qmc_opts.get('back_propagation', False)
        self.nback_prop = qmc_opts.get('nback_prop', 0)
        itcf_opts = qmc_opts.get('itcf', None)
        self.itcf_nmax = 0
        if itcf_opts is not None:
            self.itcf = True
            self.itcf_stable = itcf_opts.get('stable', True)
            self.itcf_tmax = itcf_opts.get('tmax', 0.0)
            self.itcf_mode = itcf_opts.get('mode', 'full')
            self.itcf_nmax = int(self.itcf_tmax/self.dt)
        self.nprop_tot = max(1, self.itcf_nmax+self.nback_prop)
        self.uuid = str(uuid.uuid1())
        self.seed = qmc_opts['rng_seed']
        if model['name'] == 'Hubbard':
            # sytem packages all generic information + model specific information.
            self.system = hubbard.Hubbard(model)
            self.gamma = numpy.arccosh(numpy.exp(0.5*self.dt*self.system.U))
            self.auxf = numpy.array([[numpy.exp(self.gamma), numpy.exp(-self.gamma)],
                                    [numpy.exp(-self.gamma), numpy.exp(self.gamma)]])
            self.auxf = self.auxf * numpy.exp(-0.5*self.dt*self.system.U)
            if qmc_opts['hubbard_stratonovich'] == 'continuous':
                self.two_body = hs_transform.construct_generic_one_body(system.Hubbard.gamma)

        self.propagators = afqmcpy.propagation.Projectors(model['name'],
                                                         self.hubbard_stratonovich,
                                                         self.dt, self.system.T,
                                                         self.importance_sampling,
                                                         self.system.eks,
                                                         self.ffts)
        self.cplx = 'continuous' in self.hubbard_stratonovich
        # effective hubbard U for UHF trial wavefunction.
        self.ueff = qmc_opts.get('ueff', 0.4)
        if self.cplx:
            # optimal mean-field shift for the hubbard model
            self.mf_shift = (self.system.nup + self.system.ndown) / float(self.system.nbasis)
            self.iut_fac = 1j*numpy.sqrt((self.system.U*self.dt))
            self.ut_fac = self.dt*self.system.U
            # Include factor of M! bad name
            self.mf_nsq = self.system.nbasis * self.mf_shift**2.0
        if qmc_opts['trial_wavefunction'] == 'free_electron':
            self.trial = trial_wave_function.Free_Electron(self.system, self.cplx)
        elif qmc_opts['trial_wavefunction'] == 'UHF':
            self.trial = trial_wave_function.UHF(self.system, self.cplx, self.ueff)
        elif qmc_opts['trial_wavefunction'] == 'multi_determinant':
            self.trial = trial_wave_function.multi_det(self.system, self.cplx)
        self.local_energy_bound = (2.0/self.dt)**0.5
        self.mean_local_energy = 0
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
        encoder.FLOAT_REPR = lambda o: format(o, '.6f')
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
