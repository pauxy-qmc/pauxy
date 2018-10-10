#!/usr/bin/env python
# Dump MO integrals for PAUXY.

from pyscf import gto, scf
from pauxy.utils.linalg import get_orthoAO
from pauxy.utils.from_pyscf import dump_pauxy
import numpy as np
import h5py
mol = gto.M(
    atom = [['C',  -2.433661,   0.708302,   0.000000],
['C',  -2.433661,  -0.708302,   0.000000],
['H',  -3.378045,  -1.245972,   0.000000],
['H',  -3.378045,   1.245972,   0.000000],
['C',  -1.244629,   1.402481,   0.000000],
['C',  -1.244629,  -1.402481,   0.000000],
['C',  -0.000077,   0.717168,   0.000000],
['C',  -0.000077,  -0.717168,   0.000000],
['H',  -1.242734,   2.490258,   0.000000],
['H',  -1.242734,  -2.490258,   0.000000],
['C',   1.244779,   1.402533,   0.000000],
['C',   1.244779,  -1.402533,   0.000000],
['C',   2.433606,   0.708405,   0.000000],
['C',   2.433606,  -0.708405,   0.000000],
['H',   1.242448,   2.490302,   0.000000],
['H',   1.242448,  -2.490302,   0.000000],
['H',   3.378224,   1.245662,   0.000000],
['H',   3.378224,  -1.245662,   0.000000]],
    basis = 'sto-3g',
    symmetry = 1,
    symmetry_subgroup = 'C1',
    verbose = 5,
    charge = 0,
    spin = 0
)

mf = scf.RHF(mol)
mf.chkfile = 'scf.naph.dump'
mf.kernel()
#1:26 28:30 27 31:39
reorder_mo = [i for i in range (26)]
reorder_mo += [i+27 for i in range(3)]
reorder_mo += [26]
reorder_mo += [i+30 for i in range(9)]
reorder_mo += [i+39 for i in range(19)]
mo_ordered = np.zeros(mf.mo_coeff.shape)
for inew, imo in enumerate(reorder_mo):
  mo_ordered[:,inew] = mf.mo_coeff[:,imo].copy()
mf.mo_coeff = mo_ordered.copy()
nao = mo_ordered.shape[0]
dip = mol.intor('int1e_r').reshape(3,nao,nao)
delta = 0.001
Efield = np.array([0.0, 0.0, delta ])
hefield = -np.einsum("i,ijk->jk",Efield,dip)
hcore = mf.get_hcore() + hefield
fock = hcore + mf.get_veff()
s1e = mol.intor('int1e_ovlp_sph')
orthoAO = get_orthoAO(s1e)
with h5py.File(mf.chkfile) as fh5:
  fh5['scf/orbs'] = np.eye(mf.mo_coeff.shape[-1])
  fh5['scf/hcore'] = hcore
  fh5['scf/fock'] = fock
  fh5['scf/orthoAORot'] = mo_ordered

# Dump necessary data using pyscf checkpoint file.
dump_pauxy(chkfile=mf.chkfile, outfile='naph.dip.fcidump.h5')
