import unittest
from pyscf import gto, ao2mo, scf
from pauxy.systems.generic import Generic
from pauxy.utils.from_pyscf import integrals_from_scf
from pauxy.trial_wavefunction.utils import get_trial_wavefunction

class TestGeneric(unittest.TestCase):

    def test_from_pyscf(self):
        atom = gto.M(atom='Ne 0 0 0', basis='sto-3g', verbose=0)
        mf = scf.RHF(atom)
        ehf = mf.kernel()
        h1e, chol, ecore, oao = integrals_from_scf(mf, verbose=0, chol_cut=1e-5)
        nb = h1e.shape[0]
        system = Generic(nelec=mf.mol.nelec, h1e=h1e,
                         chol=chol.reshape((-1,nb,nb)),
                         ecore=ecore, verbose=0)
        system.oao = oao
        trial = get_trial_wavefunction(system,mf=mf) 
        system.construct_integral_tensors_real(trial)
        trial.calculate_energy(system)
        self.assertAlmostEqual(trial.energy, ehf)
