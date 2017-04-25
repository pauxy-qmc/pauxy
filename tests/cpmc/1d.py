#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import afqmcpy as af

qs = af.state.State({'name': 'Hubbard', 't': 1.0, 'U': 1, 'nx': 4, 'ny': 1,
                     'nup': 1, 'ndown': 1}, dt=0.01, nsteps=1000, nmeasure=10, nwalkers=100)

af.qmc.do_qmc(qs)
