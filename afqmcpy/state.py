import hubbard
import numpy as np
import random

class State:

    def __init__(self, model, dt, nsteps, method='CPMC',
                 constrained=False, temp=0.0, nmeasure=10, seed=7,
                 nwalkers=1):

        if model['name'] == 'Hubbard':
            # sytem packages all generic information + model specific information.
            self.system = hubbard.Hubbard(model)
            self.nwalkers = nwalkers
            self.gamma = np.arccosh(np.exp(0.5*dt*self.system.U))
            self.auxf = np.array([[np.exp(self.gamma), np.exp(-self.gamma)],
                                  [np.exp(-self.gamma), np.exp(self.gamma)]])
            self.auxf = self.auxf * np.exp(-0.5*dt*self.system.U*self.system.ne)
            if method ==  'CPMC':
                self.projectors = hubbard.Projectors(self.system, dt)

        self.dt = dt
        self.method = method
        self.constrainted = constrained
        self.temp = temp
        self.nsteps = nsteps
        self.nmeasure = nmeasure
