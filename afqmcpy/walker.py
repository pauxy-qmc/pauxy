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

    def inverse_overlap(self, trial):
        self.inv_ovlp[0] = scipy.linalg.inv((trial[0].conj()).T.dot(self.phi[0]))
        self.inv_ovlp[1] = scipy.linalg.inv((trial[1].conj()).T.dot(self.phi[1]))

    def calc_otrial(self, trial):
        # The importance function, i.e. <phi_T|phi>. We do 1 over this because
        # inv_ovlp stores the inverse overlap matrix for ease when updating the
        # green's function.
        return 1.0/(scipy.linalg.det(self.inv_ovlp[0])*scipy.linalg.det(self.inv_ovlp[1]))

    def reortho(self):
        (self.phi, R) = [list(t)
                         for t in zip(*[scipy.linalg.qr(p, mode='economic')
                         for p in self.phi])]
        signs_up = np.diag(np.sign(np.diag(R[0])))
        signs_down = np.diag(np.sign(np.diag(R[1])))
        self.phi[0] = self.phi[0].dot(signs_up)
        self.phi[1] = self.phi[1].dot(signs_down)
        detR = (scipy.linalg.det(signs_up.dot(R[0]))*scipy.linalg.det(signs_down.dot(R[1])))
        self.ot = self.ot / detR

    def reortho_free(self):
        (self.phi, R) = [list(t)
                         for t in zip(*[scipy.linalg.qr(p, mode='economic')
                         for p in self.phi])]
        signs_up = np.diag(np.sign(np.diag(R[0])))
        signs_down = np.diag(np.sign(np.diag(R[1])))
        self.phi[0] = self.phi[0].dot(signs_up)
        self.phi[1] = self.phi[1].dot(signs_down)
        detR = (scipy.linalg.det(signs_up.dot(R[0]))*scipy.linalg.det(signs_down.dot(R[1])))
        self.ot = self.ot / detR
        self.weight = self.weight * detR

    def greens_function(self, trial):
        self.G[0] = np.dot(np.dot(self.phi[0], self.inv_ovlp[0]), (trial[0].conj()).T).T
        self.G[1] = np.dot(np.dot(self.phi[1], self.inv_ovlp[1]), (trial[1].conj()).T).T
