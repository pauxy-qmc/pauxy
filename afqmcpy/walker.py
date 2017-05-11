import numpy as np
import random
import scipy.linalg
import copy
import afqmcpy.estimators

class Walker:

    def __init__(self, nw, system, trial):
        self.weight = nw
        self.phi = copy.deepcopy(trial)
        self.inv_ovlp = [0, 0]
        self.inverse_overlap(trial)
        self.G = [0, 0]
        self.greens_function(trial)
        self.ot = 1.0
        (self.E_L, self.vbar) = afqmcpy.estimators.local_energy(system, self.G)

    def inverse_overlap(self, trial):
        self.inv_ovlp[0] = scipy.linalg.inv(trial[0].T.dot(self.phi[0]))
        self.inv_ovlp[1] = scipy.linalg.inv(trial[1].T.dot(self.phi[1]))

    def calc_otrial(self, trial):
        # The importance function, i.e. <phi_T|phi>. We do 1 over this because
        # inv_ovlp stores the inverse overlap matrix for ease when updating the
        # green's function.
        return 1.0/(scipy.linalg.det(self.inv_ovlp[0])*scipy.linalg.det(self.inv_ovlp[1]))

    def reortho(self):
        (self.phi, R) = [list(t)
                         for t in zip(*[scipy.linalg.qr(p, mode='economic')
                         for p in self.phi])]
        self.ot = self.ot / (scipy.linalg.det(R[0])*scipy.linalg.det(R[1]))

    def greens_function(self, trial):
        self.G[0] = np.dot(np.dot(self.phi[0], self.inv_ovlp[0]), np.transpose(trial[0]))
        self.G[1] = np.dot(np.dot(self.phi[1], self.inv_ovlp[1]), np.transpose(trial[1]))
