import copy
import numpy
from pauxy.estimators.mixed import local_energy_multi_det
from pauxy.trial_wavefunction.free_electron import FreeElectron
from pauxy.utils.io import read_fortran_complex_numbers

class MultiDetWalker(object):
    """Multi-Det style walker.

    Parameters
    ----------
    weight : int
        Walker weight.
    system : object
        System object.
    trial : object
        Trial wavefunction object.
    index : int
        Element of trial wavefunction to initalise walker to.
    weights : string
        Initialise weights to zeros or ones.
    wfn0 : string
        Initial wavefunction.
    """

    def __init__(self, walker_opts, system, trial, index=0,
                 weights='zeros', wfn0='init'):
        self.weight = walker_opts.get('weight', 1)
        self.alive = 1
        # Initialise to a particular free electron slater determinant rather
        # than GHF. Can actually initialise to GHF by passing single GHF with
        # initial_wavefunction. The distinction is really for back propagation
        # when we may want to use the full expansion.
        self.nup = system.nup
        self.phi = copy.deepcopy(trial.psi[0])
        # This stores an array of overlap matrices with the various elements of
        # the trial wavefunction.
        self.inv_ovlp = numpy.zeros(shape=(trial.ndets, system.ne, system.ne),
                                    dtype=self.phi.dtype)
        if weights == 'zeros':
            self.weights = numpy.zeros(trial.ndets, dtype=trial.psi.dtype)
        else:
            self.weights = numpy.ones(trial.ndets, dtype=trial.psi.dtype)
        if wfn0 != 'GHF':
            self.inverse_overlap(trial.psi)
        # Green's functions for various elements of the trial wavefunction.
        self.Gi = numpy.zeros(shape=(trial.ndets, 2, system.nbasis,
                                     system.nbasis), dtype=self.phi.dtype)
        # Actual green's function contracted over determinant index in Gi above.
        # i.e., <psi_T|c_i^d c_j|phi>
        self.G = numpy.zeros(shape=(2, system.nbasis, system.nbasis),
                             dtype=self.phi.dtype)
        self.ots = numpy.zeros(trial.ndets, dtype=self.phi.dtype)
        # Contains overlaps of the current walker with the trial wavefunction.
        if wfn0 != 'GHF':
            self.ot = self.calc_otrial(trial)
            self.greens_function(trial)
            self.E_L = pauxy.estimators.local_energy_multi_det(system, self.Gi,
                                                               self.weights)
        self.nb = system.nbasis
        # Historic wavefunction for back propagation.
        self.phi_old = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        self.phi_init = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        self.phi_bp = copy.deepcopy(trial.psi)

    def inverse_overlap(self, trial):
        """Compute inverse overlap matrix from scratch.

        Parameters
        ----------
        trial : :class:`numpy.ndarray`
            Trial wavefunction.
        """
        nup = self.nup
        for (indx, t) in enumerate(trial):
            self.inv_ovlp[indx,0,:,:] = (
                    scipy.linalg.inv((t.conj()).T.dot(self.phi[:,:nup]))
            )
            self.inv_ovlp[indx,1,:,:] = (
                    scipy.linalg.inv((t.conj()).T.dot(self.phi[:,nup:]))
            )

    def calc_otrial(self, trial):
        """Caculate overlap with trial wavefunction.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.

        Returns
        -------
        ot : float / complex
            Overlap.
        """
        # The trial wavefunctions coefficients should be complex conjugated
        # on initialisation!
        for (ix, inv) in enumerate(self.inv_ovlp):
            self.ots[ix] = 1.0 / scipy.linalg.det(inv[0])*scipy.linalg.det(inv[1])
            self.weights[ix] = trial.coeffs[ix].conj() * self.ots[ix]
        return sum(self.weights)

    def reortho(self, trial):
        """reorthogonalise walker.

        parameters
        ----------
        trial : object
            trial wavefunction object. for interface consistency.
        """
        nup = self.nup
        ndown = self.ndown
        (self.phi[:,:nup], Rup) = scipy.linalg.qr(self.phi[:,:nup],
                                                  mode='economic')
        Rdown = numpy.zeros(Rup.shape)
        if (ndown > 0):
            (self.phi[:,nup:], Rdown) = scipy.linalg.qr(self.phi[:,nup:],
                                                        mode='economic')
        signs_up = numpy.diag(numpy.sign(numpy.diag(Rup)))
        if (ndown > 0):
            signs_down = numpy.diag(numpy.sign(numpy.diag(Rdown)))
        self.phi[:,:nup] = self.phi[:,:nup].dot(signs_up)
        if (ndown > 0):
            self.phi[:,nup:] = self.phi[:,nup:].dot(signs_down)
        drup = scipy.linalg.det(signs_up.dot(Rup))
        drdn = 1.0
        if (ndown > 0):
            drdn = scipy.linalg.det(signs_down.dot(Rdown))
        detR = drup * drdn
        self.ot = self.ot / detR
        return detR

    def greens_function(self, trial):
        """Compute walker's green's function.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        """
        nup = self.nup
        for (ix, t) in enumerate(trial.psi):
            # construct "local" green's functions for each component of psi_T
            self.Gi[ix,0,:,:] = (
                    (self.phi[:,:nup].dot(self.inv_ovlp[ix]).dot(t.conj().T)).T
            )
            self.Gi[ix,0,:,:] = (
                    (self.phi[:,nup:].dot(self.inv_ovlp[ix]).dot(t.conj().T)).T
            )

    def local_energy(self, system):
        """Compute walkers local energy

        Parameters
        ----------
        system : object
            System object.

        Returns
        -------
        (E, T, V) : tuple
            Mixed estimates for walker's energy components.
        """
        return local_energy_multi_det(system, self.Gi, self.weights)
