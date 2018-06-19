import numpy
import time
from pauxy.estimators.mixed import gab, local_energy

class HartreeFock(object):

    def __init__(self, system, cplx, trial, parallel=False, verbose=False):
        if verbose:
            print ("# Parsing Hartree--Fock trial wavefunction input options.")
        init_time = time.time()
        self.name = "hartree_fock"
        self.type = "hartree_fock"
        self.initial_wavefunction = trial.get('initial_wavefunction',
                                              'hartree_fock')
        self.trial_type = complex
        self.psi = numpy.zeros(shape=(system.nbasis, system.nup+system.ndown),
                               dtype=self.trial_type)
        occup = numpy.identity(system.nup)
        occdown = numpy.identity(system.ndown)
        self.psi[:system.nup,:system.nup] = occup
        self.psi[:system.ndown,system.nup:] = occdown
        gup = gab(self.psi[:,:system.nup],
                                   self.psi[:,:system.nup])
        gdown = numpy.zeros(gup.shape)
        if (system.ndown >0):
            gdown = gab(self.psi[:,system.nup:], self.psi[:,system.nup:])

        self.G = numpy.array([gup,gdown])
        (self.energy, self.e1b, self.e2b) = local_energy(system, self.G)
        self.coeffs = 1.0
        self.bp_wfn = trial.get('bp_wfn', None)
        self.error = False
        self.initialisation_time = time.time() - init_time
        if verbose:
            print ("# Finished setting up trial wavefunction.")
