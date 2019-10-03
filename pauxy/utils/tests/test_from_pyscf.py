import os
import unittest
try:
    from pyscf import gto, ao2mo, scf
    from pauxy.utils.from_pyscf import integrals_from_scf, integrals_from_chkfile
    no_pyscf = False
except ImportError:
    no_pyscf = True

class TestGeneric(unittest.TestCase):

    @unittest.skipIF(no_pyscf, "PYSCF not found.")
    def test_from_pyscf(self):
        atom = gto.M(atom='Ne 0 0 0', basis='sto-3g', verbose=0)
        mf = scf.RHF(atom)
        mf.kernel()
        h1e, chol, ecore, oao = integrals_from_scf(mf, verbose=0, chol_cut=1e-5)
        nb = h1e.shape[0]

    @unittest.skipIF(no_pyscf, "PYSCF not found.")
    def test_from_chkfile(self):
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

if __name__ == '__main__':
    unittest.main()
