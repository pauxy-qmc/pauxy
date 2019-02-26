#!/usr/bin/env python

from pyscf import gto, scf
import numpy
import h5py
from pauxy.utils.from_pyscf import dump_pauxy

mol = gto.M(atom=[['Ne', (0,0,0)]],
            basis='cc-pVDZ',
            unit='Angstrom')
mf = scf.RHF(mol)
mf.chkfile = 'neon.chk.h5'
mf.kernel()
hcore = mf.get_hcore()
with h5py.File(mf.chkfile) as fh5:
    fh5['/scf/hcore'] = hcore
    fh5['/scf/orthoAORot'] = mf.mo_coeff
    fh5['/scf/orbs'] = numpy.eye(hcore.shape[-1])

# Dump integrals and freeze the 1s orbitals.
dump_pauxy(chkfile=mf.chkfile, outfile='neon.h5', qmcpack=True,
           cholesky=True, cas=[8,13])
