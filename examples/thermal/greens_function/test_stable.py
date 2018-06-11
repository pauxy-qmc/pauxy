#!/usr/bin/env python

from pauxy.estimators.thermal import greens_function
from pauxy.walkers.thermal import ThermalWalker
from pauxy.systems.hubbard import Hubbard
from pauxy.trial_density_matrices.onebody import OneBody

import matplotlib.pyplot as pl
import numpy

sys_dict = {'name': 'Hubbard', 'nx': 4, 'ny': 4,
            'nup': 8, 'ndown': 8, 'U': 4, 't': 1}

dt = 0.01

system = Hubbard(sys_dict, dt)

chem_pot = 1.0
beta = 1.0

trial = OneBody(system, beta, dt, verbose=True)

# mu = numpy.linspace(-10,10,100)

# nav = [14-trial.nav(beta, m) for m in mu]
# pl.plot(mu, nav)
# pl.show()

# walker = ThermalWalker(1, system, trial, 100, 10)
