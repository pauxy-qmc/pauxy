import numpy as np
from pyscf import scf
from dfmp2 import *
from pyscf import gto
import h5py
from pauxy.utils.from_pyscf import dump_pauxy

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

nfc = 10
nfv = 5
pt = DFMP2(mf, frozen=nfc, nfv = nfv)
pt.with_df.auxbasis = 'cc-pvdz-jkfit'
emp2, t2 = pt.kernel(with_t2=True)

t2 = np.array(t2)

nocc = pt.nocc
nvir = pt.nmo - nocc - nfv

slaters = []
coefs = []

mohf = mf.mo_coeff.copy()

tthresh = 2.e-2

for i in range(nocc):
    for j in range (nocc):
        for a in range (nvir):
            for b in range (nvir):
                if (abs(t2[i,j,a,b]) > tthresh):
                    mo_ordered = np.zeros(mf.mo_coeff.shape)
                    reorder_mo = [p for p in range(nfc+nocc+nvir)]
                    reorder_mo[nfc+i] = a
                    reorder_mo[nfc+j] = b
                    reorder_mo[a] = nfc+i
                    reorder_mo[b] = nfc+j
                    for inew, imo in enumerate(reorder_mo):
                      mo_ordered[:,inew] = mf.mo_coeff[:,imo].copy()
                    slaters += [mo_ordered.copy()]
                    coefs += [t2[i,j,a,b]]

coefs = np.array(coefs)
sortidx = np.argsort(np.abs(coefs))

slaters_ordered = []
coefs_ordered = []

print("Total {} determinants selcted above tthresh of {}".format(len(coefs),tthresh))

for idx in reversed(sortidx):
    slaters_ordered += [slaters[idx]]
    coefs_ordered += [coefs[idx]]

reorder_mo = [i for i in range (26)]
reorder_mo += [i+27 for i in range(3)]
reorder_mo += [26]
reorder_mo += [i+30 for i in range(9)]
reorder_mo += [i+39 for i in range(19)]
mo_ordered = np.zeros(mf.mo_coeff.shape)
for inew, imo in enumerate(reorder_mo):
  mo_ordered[:,inew] = mf.mo_coeff[:,imo].copy()
mf.mo_coeff = mo_ordered.copy()
hcore = mf.get_hcore()
fock = hcore + mf.get_veff()
with h5py.File(mf.chkfile) as fh5:
  fh5['scf/orbs'] = numpy.array(slaters_ordered)
  fh5['scf/coeffs'] = numpy.array(coefs_ordered)
  fh5['scf/hcore'] = hcore
  fh5['scf/fock'] = fock
  fh5['scf/orthoAORot'] = mo_ordered

# Dump necessary data using pyscf checkpoint file.
dump_pauxy(chkfile=mf.chkfile, outfile='naph.fcidump.h5')
