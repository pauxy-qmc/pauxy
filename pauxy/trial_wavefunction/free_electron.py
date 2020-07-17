import numpy
import time
from pauxy.utils.io import read_fortran_complex_numbers
from pauxy.utils.linalg import diagonalise_sorted
from pauxy.estimators.mixed import local_energy
from pauxy.estimators.greens_function import gab

class FreeElectron(object):

    def __init__(self, system, trial, verbose=False):
        self.verbose = verbose
        if verbose:
            print ("# Parsing free electron input options.")
        init_time = time.time()
        self.name = "free_electron"
        self.type = "free_electron"
        self.initial_wavefunction = trial.get('initial_wavefunction',
                                              'free_electron')
        if verbose:
            print ("# Diagonalising one-body Hamiltonian.")
        (self.eigs_up, self.eigv_up) = diagonalise_sorted(system.T[0])
        (self.eigs_dn, self.eigv_dn) = diagonalise_sorted(system.T[1])
        self.reference = trial.get('reference', None)
        self.trial_type = complex
        self.read_in = trial.get('read_in', None)
        self.psi = numpy.zeros(shape=(system.nbasis, system.nup+system.ndown),
                               dtype=self.trial_type)
        if self.read_in is not None:
            if verbose:
                print ("# Reading trial wavefunction from %s"%(self.read_in))
            try:
                self.psi = numpy.load(self.read_in)
                self.psi = self.psi.astype(self.trial_type)
            except OSError:
                if verbose:
                    print("# Trial wavefunction is not in native numpy form.")
                    print("# Assuming Fortran GHF format.")
                orbitals = read_fortran_complex_numbers(self.read_in)
                tmp = orbitals.reshape((2*system.nbasis, system.ne),
                                       order='F')
                ups = []
                downs = []
                # deal with potential inconsistency in ghf format...
                for (i, c) in enumerate(tmp.T):
                    if all(abs(c[:system.nbasis]) > 1e-10):
                        ups.append(i)
                    else:
                        downs.append(i)
                self.psi[:, :system.nup] = tmp[:system.nbasis, ups]
                self.psi[:, system.nup:] = tmp[system.nbasis:, downs]
        else:
            # I think this is slightly cleaner than using two separate
            # matrices.
            if self.reference is not None:
                self.psi[:, :system.nup] = self.eigv_up[:, self.reference]
                self.psi[:, system.nup:] = self.eigv_dn[:, self.reference]
            else:
                self.psi[:, :system.nup] = self.eigv_up[:, :system.nup]
                self.psi[:, system.nup:] = self.eigv_dn[:, :system.ndown]
                
                nocca = system.nup
                noccb = system.ndown
                nvira = system.nbasis-system.nup
                nvirb = system.nbasis-system.ndown
                self.virt = numpy.zeros((system.nbasis, nvira+nvirb))

                self.virt[:, :nvira] = self.eigv_up[:,nocca:nocca+nvira]
                self.virt[:, nvira:nvira+nvirb] = self.eigv_dn[:,noccb:noccb+nvirb]

        gup = gdown = numpy.zeros((system.nbasis, system.nbasis))

        if (system.nup>0):
            gup = gab(self.psi[:, :system.nup],
                                         self.psi[:, :system.nup]).T
        if (system.ndown>0):
            gdown = gab(self.psi[:, system.nup:],
                                           self.psi[:, system.nup:]).T
        self.G = numpy.array([gup, gdown])
        # For interface compatability
        self.init = self.psi
        self.coeffs = 1.0
        self.ndets = 1
        self.bp_wfn = trial.get('bp_wfn', None)
        self.error = False
        self.eigs = numpy.append(self.eigs_up, self.eigs_dn)
        self.eigs.sort()
        self.initialisation_time = time.time() - init_time
        self.init = self.psi

        if (system.name == "HubbardHolstein"):
            rho = [self.G[0].diagonal(), self.G[1].diagonal()]
            self.shift = numpy.sqrt(system.m*system.w0*2.0) * system.g *(rho[0] + rho[1]) / (system.m * system.w0**2)
            print("# Shift = {}".format(self.shift[:3]))
            nX = numpy.array([numpy.diag(self.shift), numpy.diag(self.shift)])
            V = - system.g * numpy.sqrt(system.w0 * 2.0) * nX
            self.update_wfn(system, V, verbose=0) # trial update
            if verbose:
                print ("# Updated free_electron.")

        self._mem_required = 0.0
        self._rchol = None
        if verbose:
            print ("# Finished initialising free electron trial wavefunction.")

    def update_greens_function(self, system, verbose=0):
        gup = gab(self.psi[:, :system.nup],
                                         self.psi[:, :system.nup]).T
        gdown = gab(self.psi[:, system.nup:],
                                           self.psi[:, system.nup:]).T
        self.G = numpy.array([gup, gdown])

    def update_wfn(self, system, V, verbose=0):
        (self.eigs_up, self.eigv_up) = diagonalise_sorted(system.T[0]+V[0])
        (self.eigs_dn, self.eigv_dn) = diagonalise_sorted(system.T[1]+V[1])

        # I think this is slightly cleaner than using two separate
        # matrices.
        if self.reference is not None:
            self.psi[:, :system.nup] = self.eigv_up[:, self.reference]
            self.psi[:, system.nup:] = self.eigv_dn[:, self.reference]
        else:
            self.psi[:, :system.nup] = self.eigv_up[:, :system.nup]
            self.psi[:, system.nup:] = self.eigv_dn[:, :system.ndown]
            nocca = system.nup
            noccb = system.ndown
            nvira = system.nbasis-system.nup
            nvirb = system.nbasis-system.ndown

            self.virt[:, :nvira] = numpy.real(self.eigv_up[:,nocca:nocca+nvira])
            self.virt[:, nvira:nvira+nvirb] = numpy.real(self.eigv_dn[:,noccb:noccb+nvirb])
        
        self.eigs = numpy.append(self.eigs_up, self.eigs_dn)
        self.eigs.sort()

        gup = gdown = numpy.zeros((system.nbasis, system.nbasis))

        if (system.nup>0):
            gup = gab(self.psi[:, :system.nup],
                                         self.psi[:, :system.nup]).T
        if (system.ndown>0):
            gdown = gab(self.psi[:, system.nup:],
                                           self.psi[:, system.nup:]).T

        self.G = numpy.array([gup, gdown])

    def calculate_energy(self, system):
        if self.verbose:
            print ("# Computing trial energy.")
        (self.energy, self.e1b, self.e2b) = local_energy(system, self.G)
        if self.verbose:
            print ("# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                   %(self.energy.real, self.e1b.real, self.e2b.real))
