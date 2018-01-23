#!/usr/bin/env python

import afqmcpy.hubbard
import scipy.linalg
import sys

n = int(sys.argv[1])
U = float(sys.argv[2])
if (sys.argv[3] == 'None'):
    kx = None
    ky = None
else:
    kx = float(sys.argv[3])
    ky = float(sys.argv[4])

system = {
    "t": 1.0,
    "U": U,
    "nx": 3,
    "ny": 3,
    "nup": n,
    "ndown": n,
    "ktwist": [kx, ky]
}

sys = afqmcpy.hubbard.Hubbard(system, 0.1)
print (scipy.linalg.eigh(sys.T[0])[0])

if kx is None:
    f = open('FCIDUMP', 'w')
else:
    f = open('FCIDUMP_CPLX', 'w')
fcidump = sys.fcidump(False)
f.write(fcidump)
f.close()
