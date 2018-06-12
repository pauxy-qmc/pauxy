#!/usr/bin/env python

from pauxy.estimators.thermal import greens_function
from pauxy.walkers.thermal import ThermalWalker
from pauxy.systems.hubbard import Hubbard
from pauxy.trial_density_matrices.onebody import OneBody

import matplotlib.pyplot as pl
import numpy

sys_dict = {'name': 'Hubbard', 'nx': 4, 'ny': 4,
            'nup': 7, 'ndown': 7, 'U': 4, 't': 1}

dt = 0.01

system = Hubbard(sys_dict, dt)

beta = 2

trial = OneBody(system, beta, dt, verbose=True)

num_slices = int(beta/dt)

walker = ThermalWalker(1, system, trial, num_slices, bin_size=2)
walker.construct_greens_function_unstable(0)
print ("Unstable trace: ", walker.G[0].trace())
walker.construct_greens_function_stable(0)
print ("Stable trace: ", walker.G[0].trace())
