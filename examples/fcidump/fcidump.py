#!/usr/bin/env python

import afqmcpy.hubbard
import scipy.linalg

system = {
    "t": 1.0,
    "U": 4,
    "nx": 3,
    "ny": 3,
    "nup": 1,
    "ktwist": [0.01, -0.01],
    "ndown": 1,
}

sys = afqmcpy.hubbard.Hubbard(system, 0.1)
print (scipy.linalg.eigh(sys.T[0])[0])

f = open('FCIDUMP', 'w')
fcidump = sys.fcidump(False)
f.write(fcidump)
f.close()
