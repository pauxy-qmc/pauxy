import numpy as np
import random
import scipy.linalg

class Walker:

    def __init__(self, nw, system, eigv):
        self.weight = nw
        self.phi = np.array([eigv[:,:system.nup], eigv[:,:system.ndown]])
        self.ovlp = ([np.zeros(np.shape(self.phi[0])), 
                      np.zeros(np.shape(self.phi[0]))])


    def prop_t2(self, bt2):
        self.phi[0] = np.dot(bt2, self.phi[0])
        self.phi[1] = np.dot(bt2, self.phi[1])

    def prop_v(self, auxf, nbasis):
        # Construct random auxilliary field.
        x = np.array([auxf[random.randint(0, 1)] for i in range(0, nbasis)])
        self.phi[0] = np.dot(np.diag(x), self.phi[0])
        self.phi[1] = np.dot(np.diag(1.0/x), self.phi[1])

    def overlap(self, trial):
        self.ovlp[0] = scipy.linalg.inv(np.dot(np.transpose(trial[0]), self.phi[0]))
        self.ovlp[1] = scipy.linalg.inv(np.dot(np.transpose(trial[1]), self.phi[1]))

    def reortho(self):
        for (phi, ovlp) in zip(self.phi, self.ovlp):
            (Q, R) = scipy.linalg.qr(phi, mode='economic')
            phi = Q
            ovlp = ovlp / scipy.linalg.det(R)
