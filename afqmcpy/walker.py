import numpy as np
import random
import scipy.linalg
import copy

class Walker:

    def __init__(self, nw, system, trial):
        self.weight = nw
        self.phi = copy.deepcopy(trial)
        self.inv_ovlp = [0, 0]
        self.inverse_overlap(trial)
        self.G = [0, 0]
        self.greens_function(trial)
        self.ot = 1.0

    def prop_t2(self, bt2, trial):
        self.phi[0] = bt2.dot(self.phi[0])
        self.phi[1] = bt2.dot(self.phi[1])
        # Update inverse overlap
        self.inverse_overlap(trial)
        # Update walker weight
        ot_new = self.calc_otrial(trial)
        self.greens_function(trial)
        if abs(ot_new) > 1e-16:
            self.weight = self.weight * (ot_new/self.ot)
            self.ot = ot_new
        else:
            self.weight = 0.0


    def prop_v(self, auxf, nbasis, trial):
        # Construct random auxilliary field.
        delta = auxf - 1
        for i in range(0, nbasis):
            # Ratio of determinants for the two choices of auxilliary fields
            probs = 0.5 * np.array([(1+delta[0][0]*self.G[0][i,i])*(1+delta[0][1]*self.G[1][i,i]),
                                    (1+delta[1][0]*self.G[0][i,i])*(1+delta[1][1]*self.G[1][i,i])])
            norm = sum(probs)
            self.weight = self.weight * norm
            r = random.random()
            if self.weight > 0:
                if r < probs[0]/norm:
                    vtup = self.phi[0][i,:] * delta[0, 0]
                    vtdown = self.phi[1][i,:] * delta[0, 1]
                    self.phi[0][i,:] = self.phi[0][i,:] + vtup
                    self.phi[1][i,:] = self.phi[1][i,:] + vtdown
                    self.ot = 2 * self.ot * probs[0]
                else:
                    vtup = self.phi[0][i,:] * delta[1, 0]
                    vtdown = self.phi[1][i,:] * delta[1, 1]
                    self.phi[0][i,:] = self.phi[0][i,:] + vtup
                    self.phi[1][i,:] = self.phi[1][i,:] + vtdown
                    self.ot = 2 * self.ot * probs[1]
            self.inv_ovlp[0] = sherman_morrison(self.inv_ovlp[0], trial[0].T[:,i],
                                                vtup)
            self.inv_ovlp[1] = sherman_morrison(self.inv_ovlp[1], trial[1].T[:,i],
                                                vtdown)
            self.greens_function(trial)

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


def sherman_morrison(Ainv, u, vt):

    return Ainv - (Ainv.dot(np.outer(u,vt)).dot(Ainv))/(1.0+vt.dot(Ainv).dot(u))
