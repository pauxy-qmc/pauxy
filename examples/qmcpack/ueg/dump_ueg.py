#!/usr/bin/env python
# Dump Cholesky factorisation and RHF trial wavefunction to file for QMCPACK.
import numpy
from pauxy.systems.ueg import UEG
from pauxy.utils.io import dump_qmcpack_cholesky, dump_qmcpack_trial_wfn

options = {'rs': 1, 'nup': 7, 'ndown': 7, 'ecut': 1}

ueg = UEG(options, verbose=True)

# Factor of 4 is down to conventions in factorisations.
dump_qmcpack_cholesky(ueg.T, 4*ueg.chol_vecs, (ueg.nup, ueg.ndown),
                      ueg.nbasis, ueg.ecore)
dump_qmcpack_trial_wfn(numpy.eye(ueg.nbasis), ueg.ne)
