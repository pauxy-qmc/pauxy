#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import afqmcpy as af

qs = af.state.State({'name': 'Hubbard', 't': 1.0, 'U': 1, 'nx': 4, 'ny': 1,
                     'nup': 2, 'ndown': 2}, 0.01, 1, nmeasure=1, nwalkers=1)

af.qmc.do_qmc(qs)
