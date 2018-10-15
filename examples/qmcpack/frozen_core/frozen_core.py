#!/usr/bin/env python
# Dump Cholesky factorisation and trial wavefunction to file for QMCPACK.
import numpy
from pauxy.systems.generic import Generic
from pauxy.trial_wavefunction.hartree_fock import HartreeFock
from pauxy.utils.io import dump_qmcpack_cholesky, dump_qmcpack_trial_wfn

options = {
    'nup': 34,
    'ndown': 34,
    'nfrozen_virt': 19,
    'nfrozen_core': 29,
    'integrals': 'naph.fcidump.h5'
}

atom = Generic(options, verbose=True)
trial = HartreeFock(atom, True, {}, verbose=True)
atom.frozen_core_hamiltonian(trial)
atom.construct_integral_tensors(trial)

dump_qmcpack_cholesky(atom.T, atom.schol_vecs, (atom.nup, atom.ndown),
                      atom.nbasis, atom.ecore, filename='naph.hamiltonian.h5')
dump_qmcpack_trial_wfn(atom.orbs, atom.ne, filename='naph.wfn.dat')
