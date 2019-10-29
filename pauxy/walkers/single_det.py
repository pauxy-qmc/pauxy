import copy
import numpy
import scipy.linalg
from pauxy.estimators.mixed import local_energy
from pauxy.trial_wavefunction.free_electron import FreeElectron
from pauxy.utils.linalg import sherman_morrison
from pauxy.walkers.stack import PropagatorStack, FieldConfig
from pauxy.utils.misc import get_numeric_names

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

    def __init__(self, walker_opts, system, trial, index=0, nprop_tot=None, nbp=None):
        self.weight = walker_opts.get('weight', 1)
        self.unscaled_weight = self.weight
        self.phase = 1 + 0j
        self.alive = 1
        self.phi = trial.init.copy()
        # JOONHO randomizing the guess
        # self.phi = numpy.random.rand([system.nbasis,system.ne])
        self.inv_ovlp = [0, 0]
        self.nup = system.nup
        self.ndown = system.ndown
        self.inverse_overlap(trial)
        self.G = numpy.zeros(shape=(2, system.nbasis, system.nbasis),
                             dtype=trial.psi.dtype)
        self.Gmod = [numpy.zeros(shape=(system.nup, system.nbasis),
                                 dtype=trial.psi.dtype),
                     numpy.zeros(shape=(system.ndown, system.nbasis),
                                 dtype=trial.psi.dtype)]
        self.greens_function(trial)
        self.total_weight = 0
        self.ot = 1.0
        # interface consistency
        self.ots = numpy.zeros(1)
        self.E_L = local_energy(system, self.G, self.Gmod)[0].real
        # walkers overlap at time tau before backpropagation occurs
        self.ot_bp = 1.0
        # walkers weight at time tau before backpropagation occurs
        self.weight_bp = self.weight
        # Historic wavefunction for back propagation.
        self.phi_old = copy.deepcopy(self.phi)
        self.hybrid_energy = 0.0
        # Historic wavefunction for ITCF.
        self.phi_right = copy.deepcopy(self.phi)
        self.weights = numpy.array([1])
        # Number of propagators to store for back propagation / ITCF.
        num_propg = walker_opts.get('num_propg', 1)
        # if system.name == "Generic":
            # self.stack = PropagatorStack(self.stack_size, num_propg,
                                         # system.nbasis, trial.psi.dtype,
                                         # BT=None, BTinv=None,
                                         # diagonal=False)
        try:
            excite = trial.excite_ia
        except AttributeError:
            excite = None
        if excite is not None:
            self.ia = trial.excite_ia
            self.reortho = self.reortho_excite
            self.trial_buff =  numpy.copy(trial.full_orbs[:,:self.ia[1]+1])
        if nbp is not None:
            self.field_configs = FieldConfig(system.nfields,
                                             nprop_tot, nbp,
                                             numpy.complex128)
        else:
            self.field_configs = None
        self.buff_names, self.buff_size = get_numeric_names(self.__dict__)

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
            scipy.linalg.inv((trial.psi[:,:nup].conj()).T.dot(self.phi[:,:nup]))
        )

        self.inv_ovlp[1] = numpy.zeros(self.inv_ovlp[0].shape)
        if (ndown>0):
            self.inv_ovlp[1] = (
                scipy.linalg.inv((trial.psi[:,nup:].conj()).T.dot(self.phi[:,nup:]))
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
        if ndown > 0:
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

    def reortho_excite(self, trial):
        """reorthogonalise walker.

        parameters
        ----------
        trial : object
            trial wavefunction object. for interface consistency.
        """
        nup = self.nup
        ndown = self.ndown
        # print (self.phi[:,:self.nup])
        buff = numpy.copy(self.trial_buff).astype(trial.psi.dtype)
        buff[:,:self.ia[0]] = self.phi[:,:self.ia[0]]
        buff[:,self.ia[0]+1:self.nup] = self.phi[:,self.ia[0]:self.nup-1]
        buff[:,-1] = self.phi[:,self.nup-1]
        (buff, Rup) = scipy.linalg.qr(buff, mode='economic')
        Rdown = numpy.zeros(Rup.shape)
        if (ndown > 0):
            (self.phi[:,nup:], Rdown) = scipy.linalg.qr(self.phi[:,nup:],
                                                        mode='economic')
        signs_up = numpy.diag(numpy.sign(numpy.diag(Rup)))
        if ndown > 0:
            signs_down = numpy.diag(numpy.sign(numpy.diag(Rdown)))
        buff = buff.dot(signs_up)
        self.phi[:,:self.ia[0]] = numpy.copy(buff[:,:self.ia[0]])
        self.phi[:,self.ia[0]:self.nup-1] = numpy.copy(buff[:,self.ia[0]+1:self.nup])
        self.phi[:,self.nup-1] = numpy.copy(buff[:,-1])
        if ndown > 0:
            self.phi[:,nup:] = self.phi[:,nup:].dot(signs_down)
        drup = scipy.linalg.det(signs_up.dot(Rup))
        drdn = 1.0
        if ndown > 0:
            drdn = scipy.linalg.det(signs_down.dot(Rdown))
        detR = drup * drdn
        # This only affects free projection
        self.ot = self.ot / detR
        return detR

    def greens_function(self, trial):
        """Compute walker's green's function.

        Also updates walker's inverse overlap.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        """
        nup = self.nup
        ndown = self.ndown

        ovlp = numpy.dot(self.phi[:,:nup].T, trial.psi[:,:nup].conj())
        # self.inv_ovlp[0] = scipy.linalg.inv(ovlp)
        self.Gmod[0] = numpy.dot(scipy.linalg.inv(ovlp), self.phi[:,:nup].T)
        self.G[0] = numpy.dot(trial.psi[:,:nup].conj(), self.Gmod[0])
        if ndown > 0:
            # self.inv_ovlp[1] = scipy.linalg.inv(ovlp)
            ovlp = numpy.dot(self.phi[:,nup:].T, trial.psi[:,nup:].conj())
            self.Gmod[1] = numpy.dot(scipy.linalg.inv(ovlp), self.phi[:,nup:].T)
            self.G[1] = numpy.dot(trial.psi[:,nup:].conj(), self.Gmod[1])

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

    def local_energy(self, system, two_rdm=None):
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
        return local_energy(system, self.G, Ghalf=self.Gmod, two_rdm=two_rdm)

    def get_buffer(self):
        """Get walker buffer for MPI communication

        Returns
        -------
        buff : dict
            Relevant walker information for population control.
        """
        s = 0
        buff = numpy.zeros(self.buff_size, dtype=numpy.complex128)
        for d in self.buff_names:
            data = self.__dict__[d]
            if isinstance(data, (numpy.ndarray)):
                buff[s:s+data.size] = data.ravel()
                s += data.size
            else:
                buff[s:s+1] = data
                s += 1
        if self.field_configs is not None:
            stack_buff = self.field_configs.get_buffer()
            return numpy.concatenate((buff,stack_buff))
        else:
            return buff

    def set_buffer(self, buff):
        """Set walker buffer following MPI communication

        Parameters
        -------
        buff : dict
            Relevant walker information for population control.
        """
        s = 0
        for d in self.buff_names:
            data = self.__dict__[d]
            if isinstance(data, numpy.ndarray):
                self.__dict__[d] = buff[s:s+data.size].reshape(data.shape).copy()
                dsize = data.size
            else:
                self.__dict__[d] = buff[s]
                dsize = 1
            s += dsize
