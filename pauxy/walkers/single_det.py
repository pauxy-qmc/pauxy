import copy
import numpy
import scipy.linalg
from pauxy.estimators.mixed import local_energy
from pauxy.trial_wavefunction.free_electron import FreeElectron
from pauxy.utils.linalg import sherman_morrison

class SingleDetWalker(object):
    """UHF style walker.

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
    """

    def __init__(self, walker_opts, system, trial, index=0):
        self.weight = walker_opts.get('weight', 1)
        self.alive = 1
        if trial.initial_wavefunction == 'free_electron':
            self.phi = numpy.zeros(shape=(system.nbasis,system.ne),
                                   dtype=trial.psi.dtype)
            tmp = FreeElectron(system, system.ktwist.all() != None, {})
            self.phi[:,:system.nup] = tmp.psi[:,:system.nup]
            self.phi[:,system.nup:] = tmp.psi[:,system.nup:]
        else:
            self.phi = copy.deepcopy(trial.psi)
            # JOONHO randomizing the guess
            # self.phi = numpy.random.rand([system.nbasis,system.ne])
        self.inv_ovlp = [0, 0]
        self.nup = system.nup
        self.ndown = system.ndown
        self.inverse_overlap(trial.psi)
        self.G = numpy.zeros(shape=(2, system.nbasis, system.nbasis),
                             dtype=trial.psi.dtype)
        self.Gmod = numpy.zeros(shape=(2, system.nbasis, system.nup),
                                dtype=trial.psi.dtype)
        self.greens_function(trial)
        self.ot = 1.0
        # interface consistency
        self.ots = numpy.zeros(1)
        self.E_L = local_energy(system, self.G)[0].real
        # walkers overlap at time tau before backpropagation occurs
        self.ot_bp = 1.0
        # walkers weight at time tau before backpropagation occurs
        self.weight_bp = self.weight
        # Historic wavefunction for back propagation.
        self.phi_old = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        self.phi_init = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        self.phi_bp = copy.deepcopy(self.phi)
        self.weights = numpy.array([1])

    def inverse_overlap(self, trial):
        """Compute inverse overlap matrix from scratch.

        Parameters
        ----------
        trial : :class:`numpy.ndarray`
            Trial wavefunction.
        """
        nup = self.nup
        ndown = self.ndown
        
        self.inv_ovlp[0] = (
            scipy.linalg.inv((trial[:,:nup].conj()).T.dot(self.phi[:,:nup]))
        )
        
        self.inv_ovlp[1] = numpy.zeros(self.inv_ovlp[0].shape)
        if (ndown>0):
            self.inv_ovlp[1] = (
                scipy.linalg.inv((trial[:,nup:].conj()).T.dot(self.phi[:,nup:]))
            )

    def update_inverse_overlap(self, trial, vtup, vtdown, i):
        """Update inverse overlap matrix given a single row update of walker.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        vtup : :class:`numpy.ndarray`
            Update vector for spin up sector.
        vtdown : :class:`numpy.ndarray`
            Update vector for spin down sector.
        i : int
            Basis index.
        """
        nup = self.nup
        ndown = self.ndown

        self.inv_ovlp[0] = (
            sherman_morrison(self.inv_ovlp[0], trial.psi[i,:nup].conj(), vtup)
        )
        self.inv_ovlp[1] = (
            sherman_morrison(self.inv_ovlp[1], trial.psi[i,nup:].conj(), vtdown)
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
        dup = scipy.linalg.det(self.inv_ovlp[0])
        ndown = self.ndown
        ddn = 1.0
        if (ndown >0):
            ddn = scipy.linalg.det(self.inv_ovlp[1])
        return 1.0 / (dup*ddn)

    def update_overlap(self, probs, xi, coeffs):
        """Update overlap.

        Parameters
        ----------
        probs : :class:`numpy.ndarray`
            Probabilities for chosing particular field configuration.
        xi : int
            Chosen field configuration.
        coeffs : :class:`numpy.ndarray`
            Trial wavefunction coefficients. For interface consistency.
        """
        self.ot = 2 * self.ot * probs[xi]

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
        if (ndown >0):
            (self.phi[:,nup:], Rdown) = scipy.linalg.qr(self.phi[:,nup:],
                                                        mode='economic')
        signs_up = numpy.diag(numpy.sign(numpy.diag(Rup)))
        if (ndown >0):
            signs_down = numpy.diag(numpy.sign(numpy.diag(Rdown)))
        self.phi[:,:nup] = self.phi[:,:nup].dot(signs_up)
        if (ndown >0):
            self.phi[:,nup:] = self.phi[:,nup:].dot(signs_down)
        drup = scipy.linalg.det(signs_up.dot(Rup))
        drdn = 1.0
        if (ndown >0):
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
        ndown = self.ndown

        t = trial.psi
        self.G[0] = (
            (self.phi[:,:nup].dot(self.inv_ovlp[0]).dot(t[:,:nup].conj().T)).T
        )
        self.G[1] = numpy.zeros(self.G[0].shape)
        if (ndown >0 ):
            self.G[1] = (
                (self.phi[:,nup:].dot(self.inv_ovlp[1]).dot(t[:,nup:].conj().T)).T
            )

    def rotated_greens_function(self):
        """Compute "rotated" walker's green's function.

        Green's function without trial wavefunction multiplication.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        """
        nup = self.nup
        ndown = self.ndown
        self.Gmod[0] = self.phi[:,:nup].dot(self.inv_ovlp[0])
        self.Gmod[1] = numpy.zeros(self.Gmod[0].shape)
        if (ndown>0):
            self.Gmod[1] = self.phi[:,nup:].dot(self.inv_ovlp[1])

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
        return local_energy(system, self.G)

    def get_buffer(self):
        """Get walker buffer for MPI communication

        Returns
        -------
        buff : dict
            Relevant walker information for population control.
        """
        buff = {
            'phi': self.phi,
            'phi_old': self.phi_old,
            'phi_init': self.phi_init,
            'phi_bp': self.phi_bp,
            'weight': self.weight,
            'inv_ovlp': self.inv_ovlp,
            'G': self.G,
            'overlap': self.ot,
            'overlaps': self.ots,
            'fields': self.field_configs.configs,
            'cfacs': self.field_configs.cos_fac,
            'E_L': self.E_L,
            'weight_fac': self.field_configs.weight_fac
        }
        return buff

    def set_buffer(self, buff):
        """Set walker buffer following MPI communication

        Parameters
        -------
        buff : dict
            Relevant walker information for population control.
        """
        self.phi = numpy.copy(buff['phi'])
        self.phi_old = numpy.copy(buff['phi_old'])
        self.phi_init = numpy.copy(buff['phi_init'])
        self.phi_bp = numpy.copy(buff['phi_bp'])
        self.inv_ovlp = numpy.copy(buff['inv_ovlp'])
        self.G = numpy.copy(buff['G'])
        self.weight = buff['weight']
        self.ot = buff['overlap']
        self.E_L = buff['E_L']
        self.ots = numpy.copy(buff['overlaps'])
        self.field_configs.configs = numpy.copy(buff['fields'])
        self.field_configs.cos_fac = numpy.copy(buff['cfacs'])
        self.field_configs.weight_fac = numpy.copy(buff['weight_fac'])
