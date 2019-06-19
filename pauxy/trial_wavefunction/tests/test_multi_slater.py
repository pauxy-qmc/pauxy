import numpy
import scipy.linalg
import unittest
from pyscf import gto, ao2mo, scf
from pauxy.estimators.mixed import variational_energy_multi_det
from pauxy.systems.generic import Generic
from pauxy.systems.ueg import UEG
from pauxy.utils.from_pyscf import integrals_from_scf
from pauxy.utils.io import read_qmcpack_wfn_hdf
from pauxy.utils.misc import dotdict
from pauxy.trial_wavefunction.utils import get_trial_wavefunction
from pauxy.trial_wavefunction.multi_slater import MultiSlater

class TestMultiSlater(unittest.TestCase):

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

    def test_nomsd(self):
        system = UEG({'nup': 7, 'ndown': 7, 'rs': 5, 'ecut': 4,
                      'thermal': True})
        wfn, coeffs, psi0 = read_qmcpack_wfn_hdf('wfn.h5')
        trial = MultiSlater(system, wfn, coeffs, init=psi0)
        trial.calculate_energy(system)
        ndets = len(coeffs)
        H = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
        S = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
        variational_energy_multi_det(system, wfn, coeffs, H=H, S=S)
        e, ev = scipy.linalg.eigh(H,S)
        evar = variational_energy_multi_det(system, wfn, ev[:,0])
        self.assertAlmostEqual(e[0],0.15400990069739182)
        self.assertAlmostEqual(e[0],evar[0])
