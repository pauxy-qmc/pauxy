import numpy as np
import scipy.linalg
import copy
import afqmcpy.estimators

# Worthwhile overloading / having real and complex walker classes (Hermitian
# conjugate)?
class Walker:

    def __init__(self, nw, system, trial, index):
        self.weight = nw
        self.phi = copy.deepcopy(trial)
        self.inv_ovlp = [0, 0]
        self.inverse_overlap(trial)
        self.G = [0, 0]
        self.greens_function(trial)
        self.ot = 1.0
        self.E_L = afqmcpy.estimators.local_energy(system, self.G)[0].real
        # walkers overlap at time tau before backpropagation occurs
        self.ot_bp = 1.0
        # walkers weight at time tau before backpropagation occurs
        self.weight_bp = nw
        # walkers auxiliary field configuration in back propagation interval
        self.field_config = np.zeros(shape=(system.nbasis), dtype=int)

    def inverse_overlap(self, trial, nup):
        self.inv_ovlp[0] = scipy.linalg.inv((trial[:,:nup].conj()).T.dot(self.phi[:,:nup]))
        self.inv_ovlp[1] = scipy.linalg.inv((trial[:,nup:].conj()).T.dot(self.phi[:,nup:]))

    def calc_otrial(self, trial):
        # The importance function, i.e. <phi_T|phi>. We do 1 over this because
        # inv_ovlp stores the inverse overlap matrix for ease when updating the
        # green's function.
        return 1.0/(scipy.linalg.det(self.inv_ovlp[0])*scipy.linalg.det(self.inv_ovlp[1]))

    def reortho(self, nup):
        (self.phi[:,:nup], Rup) = scipy.linalg.qr(self.phi[:,:nup], mode='economic')
        (self.phi[:,ndown:], Rdown) = scipy.linalg.qr(self.phi[:,nup:], mode='economic')
        signs_up = np.diag(np.sign(np.diag(Rup)))
        signs_down = np.diag(np.sign(np.diag(Rdown)))
        self.phi[:,:nup] = self.phi[:,:nup].dot(signs_up)
        self.phi[:,ndown:] = self.phi[:,nup:].dot(signs_down)
        detR = (scipy.linalg.det(signs_up.dot(Rup))*scipy.linalg.det(signs_down.dot(Rdown)))
        self.ot = self.ot / detR
        return detR

    def greens_function(self, trial, nup):
        self.G[0] = np.dot(np.dot(self.phi[:,:nup], self.inv_ovlp[0]), (trial[:,:nup].conj()).T).T
        self.G[1] = np.dot(np.dot(self.phi[:,nup:], self.inv_ovlp[1]), (trial[:,nup:].conj()).T).T
