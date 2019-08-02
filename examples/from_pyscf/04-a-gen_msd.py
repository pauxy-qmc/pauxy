import numpy
import h5py
from pyscf import gto, scf, mcscf, fci, ao2mo, lib
from pauxy.systems.generic import Generic
from pauxy.utils.from_pyscf import integrals_from_scf
from pauxy.utils.io import write_qmcpack_wfn
from pauxy.estimators.ci import get_hmatel, simple_fci
from pauxy.trial_wavefunction.multi_slater import MultiSlater

mol = gto.M(atom=[('Be', 0, 0, 0)], basis='sto-3g', verbose=0)
mf = scf.RHF(mol)
mf.chkfile = 'be.scf.chk'
ehf = mf.kernel()
h1e, chol, ecore, oao = integrals_from_scf(mf, verbose=0,
                                           chol_cut=1e-5,
                                           ortho_ao=False)
mc = mcscf.CASSCF(mf, 5, 4)
e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
coeff, oa, ob = zip(*fci.addons.large_ci(fcivec, mo.shape[0], (2,2),
                                         tol=1e-5, return_strs=False))
oa = [numpy.array([x for x in o.tolist()]) for o in oa]
ob = [numpy.array([x for x in o.tolist()]) for o in ob]
coeff = numpy.array(coeff,dtype=numpy.complex128)
write_qmcpack_wfn('wfn.h5', (coeff,oa,ob), 'uhf', (2,2), 5)
with h5py.File(mf.chkfile) as fh5:
    fh5['scf/orthoAORot'] = mo
