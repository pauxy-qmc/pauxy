# Dump Cholesky factorisation and trial wavefunction to file for QMCPACK.
import numpy
from pauxy.systems.generic import Generic
from pauxy.trial_wavefunction.hartree_fock import HartreeFock
from pauxy.utils.io import dump_qmcpack_cholesky, dump_qmcpack_trial_wfn
from pyscf.tools.fcidump import from_integrals

options = {
    'nup': 34,
    'ndown': 34,
    'nfrozen_virt': 0,
    'nfrozen_core': 10,
    'integrals': 'naph.fcidump.h5'
}

atom = Generic(options, verbose=True)
trial = HartreeFock(atom, True, {}, verbose=True)
atom.frozen_core_hamiltonian(trial)
from_integrals('fcidump.ascii', atom.T[0], atom.h2e, atom.nbasis, atom.ne,
               nuc=atom.ecore)
