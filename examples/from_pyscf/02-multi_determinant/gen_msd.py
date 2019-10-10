import numpy
import h5py
import scipy.sparse
from pyscf import gto, scf, mcscf, fci, ao2mo, lib
from pauxy.systems.generic import Generic
from pauxy.utils.from_pyscf import generate_integrals
from pauxy.utils.io import (
        write_qmcpack_wfn,
        dump_qmcpack_cholesky,
        write_input
        )

mol = gto.M(atom=[('Be', 0, 0, 0)], basis='sto-3g', verbose=0)
mf = scf.RHF(mol)
ehf = mf.kernel()
mc = mcscf.CASSCF(mf, 5, 4)
e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
# Rotate by casscf mo coeffs.
h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mo,
                                            chol_cut=1e-5)
dump_qmcpack_cholesky(numpy.array([h1e,h1e]), scipy.sparse.csr_matrix(chol), nelec,
                      h1e.shape[-1], e0=enuc, filename='afqmc.h5')
coeff, oa, ob = zip(*fci.addons.large_ci(fcivec, mo.shape[0], (2,2),
                                         tol=1e-5, return_strs=False))
oa = [numpy.array([x for x in o.tolist()]) for o in oa]
ob = [numpy.array([x for x in o.tolist()]) for o in ob]
coeff = numpy.array(coeff,dtype=numpy.complex128)
write_qmcpack_wfn('afqmc.h5', (coeff,oa,ob), 'uhf', (2,2), 5, mode='a')
write_input('input.json', 'afqmc.h5', 'afqmc.h5',
            options={'system': {'sparse': False}})
