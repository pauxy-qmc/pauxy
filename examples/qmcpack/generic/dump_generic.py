#!/usr/bin/env python
# Dump Cholesky factorisation and trial wavefunction to file for QMCPACK.
import numpy
from pauxy.systems.generic import Generic
from pauxy.utils.io import dump_qmcpack_cholesky, dump_qmcpack_trial_wfn

options = {'nup': 5, 'ndown': 5, 'integrals': 'fcidump.h5'}

atom = Generic(options, verbose=True)

# Factor of 4 is down to conventions in factorisations.
dump_qmcpack_cholesky(atom.h1e_mod, atom.schol_vecs, (atom.nup, atom.ndown),
                      atom.nbasis, atom.ecore, filename='neon.hamiltonian.h5')
dump_qmcpack_trial_wfn(numpy.eye(atom.nbasis), atom.ne, filename='neon.wfn.dat')
