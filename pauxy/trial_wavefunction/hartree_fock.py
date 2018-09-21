import numpy
import time
from pauxy.estimators.mixed import local_energy
from pauxy.estimators.greens_function import gab, gab_mod

class HartreeFock(object):

    def __init__(self, system, cplx, trial, parallel=False, verbose=False):
        if verbose:
            print ("# Parsing Hartree--Fock trial wavefunction input options.")
        init_time = time.time()
        self.name = "hartree_fock"
        self.type = "hartree_fock"
        self.initial_wavefunction = trial.get('initial_wavefunction',
                                              'hartree_fock')
        self.trial_type = numpy.complex128
        self.psi = numpy.zeros(shape=(system.nbasis, system.nup+system.ndown),
                               dtype=self.trial_type)
        self.excite_ia = trial.get('excitation', None)
        if system.mo_coeff is not None:
            if len(system.mo_coeff.shape) == 3:
                self.psi[:,:system.nup] = numpy.copy(system.mo_coeff[0][:,:system.nup])
                self.psi[:,system.nup:] = numpy.copy(system.mo_coeff[1][:,:system.ndown])
            else:
                self.psi[:,:system.nup] = numpy.copy(system.mo_coeff[:,:system.nup])
                self.psi[:,system.nup:] = numpy.copy(system.mo_coeff[:,:system.nup])
        else:
            occup = numpy.arange(system.nup)
            occdown = numpy.arange(system.ndown)
            if self.excite_ia is not None:
                occup[self.excite_ia[0]] = self.excite_ia[1]
                self.full_mo = numpy.eye(system.nbasis)
            self.psi[occup, numpy.arange(system.nup)] = 1
            self.psi[occdown, numpy.arange(system.ndown)+system.nup] = 1
        gup, gup_half = gab_mod(self.psi[:,:system.nup],
                                self.psi[:,:system.nup])
        gdown = numpy.zeros(gup.shape)
        gdown_half  = numpy.zeros(gup_half.shape)
        if system.ndown > 0:
            gdown, gdown_half = gab_mod(self.psi[:,system.nup:], self.psi[:,system.nup:])

        self.G = numpy.array([gup,gdown],dtype=self.trial_type)
        (self.energy, self.e1b, self.e2b) = local_energy(system, self.G,
                                                         Ghalf=[gup_half, gdown_half],
                                                         opt=False)
        self.coeffs = 1.0
        self.bp_wfn = trial.get('bp_wfn', None)
        self.error = False
        self.initialisation_time = time.time() - init_time
        if verbose:
            print ("# Finished setting up trial wavefunction.")
