#!/usr/bin/env python

import afqmcpy.generic
import afqmcpy.trial_wavefunction
import afqmcpy.estimators
import pyscf.tools.fcidump
import numpy
import scipy.linalg

atom = {'nup': 5, 'ndown': 5, 'integrals': 'fcidump.ascii'}

system = afqmcpy.generic.Generic(atom, 0.05)
dump = pyscf.tools.fcidump.read('fcidump.ascii')
h1e = dump['H1']
diag = h1e.diagonal()
tmp = h1e - numpy.diag(diag)
tmp = tmp + tmp.conj().T
h1e = tmp + numpy.diag(diag)
(e, ev) = scipy.linalg.eigh(h1e)
(e1, ev1) = scipy.linalg.eigh(system.T[0])
trial = {'name': "hartree_fock"}
trial = afqmcpy.trial_wavefunction.HartreeFock(system, False, trial)
print (trial.etrial)
print (afqmcpy.estimators.local_energy_generic_cholesky(system, trial.G))
qmc_opt = {
    'dt': 0.05,
    'expansion_order': 6,
    'hubbard_stratonovich': 'generic_continuous'
}
qmc = afqmcpy.qmc.QMCOpts(qmc_opt, system)
propagator = afqmcpy.propagation.ContinuousGeneric(qmc, system, trial)
