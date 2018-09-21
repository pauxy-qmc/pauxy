#!/usr/bin/env python
# Simple example on how to run PYSCF calculation and dump integrals for PAUXY.

from pyscf import gto, scf
from pauxy.utils.linalg import get_orthoAO
from pauxy.utils.from_pyscf import dump_pauxy
import h5py

mol = gto.Mole()
mol.basis = 'cc-pvdz',
mol.atom = (('Ne', 0.0000000, 0.0000000, 0.00000000),)
mol.build()

mf = scf.RHF(mol)
mf.chkfile = 'scf.neon.dump'
mf.kernel()

# Save some additional information required for AFQMC
hcore = mf.get_hcore()
fock = hcore + mf.get_veff()
s1e = mol.intor('int1e_ovlp_sph')
orthoAO = get_orthoAO(s1e)
with h5py.File(mf.chkfile) as fh5:
  fh5['scf/hcore'] = hcore
  fh5['scf/fock'] = fock
  fh5['scf/orthoAORot'] = orthoAO

# Dump necessary data using pyscf checkpoint file.
dump_pauxy(chkfile=mf.chkfile)

# Dump necessary data by directly passing necessary information.
dump_pauxy(mol=mol, mf=mf, outfile='from_mol.h5')
