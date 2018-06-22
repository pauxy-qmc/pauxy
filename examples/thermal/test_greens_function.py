#!/usr/bin/env python

import numpy
import matplotlib.pyplot as pl
from pauxy.estimators.thermal import greens_function, greens_function_stable
from pauxy.walkers.thermal import ThermalWalker
from pauxy.systems.hubbard import Hubbard
from pauxy.trial_density_matrices.onebody import OneBody
from pauxy.propagation.hubbard import ThermalDiscrete

sys_dict = {'name': 'Hubbard', 'nx': 4, 'ny': 4,
            'nup': 7, 'ndown': 7, 'U': 4, 't': 1}

dt = 0.1

system = Hubbard(sys_dict, dt)

beta = 0.4

trial = OneBody({}, system, beta, dt, verbose=True)

num_slices = int(beta/dt)

walker = ThermalWalker(1, system, trial, 1)

xis = numpy.random.randint(2, size=system.nbasis)

BV = numpy.zeros((2,system.nbasis))

class QMC:
    def __init__(self, dt):
        self.dt = 0.01
        self.nstblz = 10

qmc = QMC(dt)
propg = ThermalDiscrete({}, qmc, system, trial)

G0 = numpy.copy(walker.G)
print (walker.local_energy(system))
for j in range(0,num_slices):
    for (i, xi) in enumerate(xis):
        BV[0,i] = propg.auxf[xi,0]
        BV[1,i] = propg.auxf[xi,1]
        propg.update_greens_function(walker, i, xi)
    B = numpy.einsum('ki,kij->kij', BV, propg.BH1)
    walker.stack.update(B)
    propg.propagate_greens_function(walker)

G1 = numpy.copy(walker.G)
walker.construct_greens_function_stable(num_slices)
G2 = numpy.copy(walker.G)
A = numpy.linalg.matrix_power(propg.BH1[0], num_slices-1)
# A1 = B[0].dot(A)
# A2 = B[1].dot(A)
# A = numpy.array([A1,A2])
# G3 = greens_function_stable(A)
print (numpy.max(numpy.abs(G1-G2)))
