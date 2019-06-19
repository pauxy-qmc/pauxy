import numpy
import scipy.linalg
import unittest
from pyscf import gto, ao2mo, scf, fci
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

    def test_ci(self):
        mol = gto.M(atom=[('Be', 0, 0, 0)], basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        ehf = mf.kernel()
        # h1e, chol, ecore, oao = integrals_from_scf(mf, verbose=0, chol_cut=1e-5)
        # nb = h1e.shape[0]
        h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
        eri = ao2mo.kernel(mol, mf.mo_coeff)
        cisolver = fci.direct_spin1.FCI(mol)
        e, ci = cisolver.kernel(h1, eri, h1.shape[1], mol.nelec,
                                ecore=mol.energy_nuc())
        coeff, oa, ob = zip(*fci.addons.large_ci(ci, mf.mo_coeff.shape[0], mol.nelec,
                                   tol=0, return_strs=False))
        oa = [o.tolist() for o in oa]
        ob = [o.tolist() for o in ob]
        ndets = len(oa)
        H = numpy.zeros((ndets,ndets))
        print(eri.shape)
        class Excit:
            def __init__(self):
                i = None
                j = None
                a = None
                b = None
                sign = 1.0

        def get_excit(di, dj):
            ex = Excit()
            na = len(di[0])
            nb = len(di[1])
            nelec = (na,nb)
            exa = set(di[0])-set(dj[0])
            exb = set(di[1])-set(dj[1])
            nexa = len(exa)
            nexb = len(exb)
            return ex

        def slater_condon0(system, occs):
            pass
        def slater_condon1(system, occs):
            pass
        def slater_condon2(system, occs):
            pass

        for i in range(ndets):
            for j in range(ndets):
                excit = get_excit((oa[i],ob[i]), (oa[j],ob[j]))
