import os
import pytest
from pyscf import gto, ao2mo, scf
from pauxy.utils.from_pyscf import (
        integrals_from_scf,
        integrals_from_chkfile,
        get_pyscf_wfn
        )

def test_from_pyscf():
    atom = gto.M(atom='Ne 0 0 0', basis='sto-3g', verbose=0)
    mf = scf.RHF(atom)
    mf.kernel()
    h1e, chol, ecore, oao = integrals_from_scf(mf, verbose=0, chol_cut=1e-5)
    nb = h1e.shape[0]

def test_from_chkfile():
    atom = gto.M(atom='Ne 0 0 0', basis='sto-3g', verbose=0)
    mf = scf.RHF(atom)
    mf.chkfile = 'scf.chk'
    mf.kernel()
    h1e, chol, ecore, oao, mol = integrals_from_chkfile('scf.chk', verbose=0, chol_cut=1e-5)
    nb = h1e.shape[0]

def tearDown(self):
    cwd = os.getcwd()
    files = ['scf.chk']
    for f in files:
        try:
            os.remove(cwd+'/'+f)
        except OSError:
            pass
