#!/usr/bin/env python
# Dump Cholesky factorisation and trial wavefunction to file for QMCPACK.
import numpy
from pauxy.systems.generic import Generic
from pauxy.trial_wavefunction.hartree_fock import HartreeFock
from pauxy.utils.io import dump_qmcpack_cholesky, dump_qmcpack_trial_wfn

options = {'nup': 5, 'ndown': 5, 'integrals': 'fcidump.h5'}

atom = Generic(options, verbose=True)
trial = HartreeFock(atom, True, {}, verbose=True)

dump_qmcpack_cholesky(atom.T, atom.schol_vecs, (atom.nup, atom.ndown),
                      atom.nbasis, atom.ecore, filename='neon.hamiltonian.h5')
dump_qmcpack_trial_wfn(atom.mo_coeff, atom.ne, filename='neon.wfn.dat')
