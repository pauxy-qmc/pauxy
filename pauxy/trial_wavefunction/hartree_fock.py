import numpy
import time
from pauxy.estimators.mixed import local_energy
from pauxy.estimators.greens_function import gab, gab_mod
from pauxy.utils.io import read_qmcpack_wfn

class HartreeFock(object):

    def __init__(self, system, cplx, trial, parallel=False, verbose=False):
        self.verbose = verbose
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
        self.wfn_file = trial.get('filename', None)
        if self.wfn_file is not None:
            if verbose:
                print ("# Reading trial wavefunction from %s."%self.wfn_file)
            mo_matrix = read_qmcpack_wfn(self.wfn_file)
            if verbose:
                print ("# Finished reading wavefunction.")
            msq = system.nbasis**2
            if len(mo_matrix) == msq:
                mo_matrix = mo_matrix.reshape((system.nbasis, system.nbasis))
            else:
                mo_alpha = mo_matrix[:msq].reshape((system.nbasis, system.nbasis))
                mo_beta = mo_matrix[msq:].reshape((system.nbasis, system.nbasis))
                mo_matrix = numpy.array([mo_alpha, mo_beta])
        elif system.mo_coeff is not None:
            mo_matrix = system.mo_coeff
        else:
            # Assuming we're in the MO basis.
            mo_matrix = numpy.eye(system.nbasis)
        # Assuming energy ordered basis set.
        self.full_mo = mo_matrix
        occ_a = numpy.arange(system.nup)
        occ_b = numpy.arange(system.ndown)
        if len(mo_matrix.shape) == 2:
            # RHF
            self.psi[:,:system.nup] = mo_matrix[:,occ_a]
            self.psi[:,system.nup:] = mo_matrix[:,occ_b]
            if self.excite_ia is not None:
                # Only deal with alpha spin excitation for the moment.
                i = self.excite_ia[0]
                a = self.excite_ia[1]
                self.psi[:,i] = mo_matrix[:,a]
        else:
            # UHF
            self.psi[:,:system.nup] = mo_matrix[0][:,occ_a]
            self.psi[:,system.nup:] = mo_matrix[1][:,occ_b]
            if self.excite_ia is not None:
                # "Promotion energy" calculation.
                # Only deal with alpha spin excitation for the moment.
                i = self.excite_ia[0]
                a = self.excite_ia[1]
                self.psi[:,i] = mo_matrix[:,a]
        gup, self.gup_half = gab_mod(self.psi[:,:system.nup],
                                self.psi[:,:system.nup])
        gdown = numpy.zeros(gup.shape)
        self.gdown_half  = numpy.zeros(self.gup_half.shape)
        if system.ndown > 0:
            gdown, self.gdown_half = gab_mod(self.psi[:,system.nup:], self.psi[:,system.nup:])

        self.G = numpy.array([gup,gdown],dtype=self.trial_type)
        self.coeffs = 1.0
        self.bp_wfn = trial.get('bp_wfn', None)
        self.error = False
        self.initialisation_time = time.time() - init_time
        if verbose:
            print ("# Finished setting up trial wavefunction.")

    def energy(self, system):
        if self.verbose:
            print ("# Computing trial energy.")
        (self.energy, self.e1b, self.e2b) = local_energy(system, self.G,
                                                         Ghalf=[self.gup_half,
                                                            self.gdown_half],
                                                         opt=True)
        if self.verbose:
            print ("# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                   %(self.energy.real, self.e1b.real, self.e2b.real))
