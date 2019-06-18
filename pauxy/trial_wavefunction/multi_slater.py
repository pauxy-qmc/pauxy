import numpy
import time
from pauxy.estimators.mixed import local_energy
from pauxy.estimators.greens_function import gab, gab_mod

class MultiSlater(object):

    def __init__(self, system, wfn, coeffs, nbasis=None, options={},
                 init=None, parallel=False, verbose=False):
        self.verbose = verbose
        if verbose:
            print ("# Parsing Hartree--Fock trial wavefunction input options.")
        init_time = time.time()
        self.name = "MultiSlater"
        # TODO : Fix for MSD.
        self.psi = wfn
        if init is not None:
            self.init = init
        else:
            if len(self.psi.shape) == 3:
                self.init = self.psi[0].copy()
            else:
                self.init = self.psi.copy()
        self.coeffs = coeffs
        self.error = False
        na, nb = system.nelec
        Ga, Gha = gab_mod(wfn[0,:,:na], wfn[0,:,:na])
        Gb, Ghb = gab_mod(wfn[0,:,na:], wfn[0,:,na:])
        self.G = numpy.array([Ga, Gb])
        self.Ghalf = numpy.array([Gha, Ghb])
        self.initialisation_time = time.time() - init_time
        if verbose:
            print ("# Finished setting up trial wavefunction.")

    def calculate_energy(self, system):
        if self.verbose:
            print ("# Computing trial wavefunction energy.")
        start = time.time()
        # Write separate energy routine based on wavefunction.
        (self.energy, self.e1b, self.e2b) = local_energy(system, self.G,
                                                         Ghalf=self.Ghalf,
                                                         opt=True)
        if self.verbose:
            print("# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                   %(self.energy.real, self.e1b.real, self.e2b.real))
            print("# Time to evaluate local energy: %f s"%(time.time()-start))
