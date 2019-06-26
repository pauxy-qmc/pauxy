import numpy
import scipy.linalg
import unittest
import sys
from pyscf import gto, ao2mo, scf, fci, tools
from pauxy.estimators.mixed import variational_energy_multi_det, local_energy
from pauxy.estimators.greens_function import gab
from pauxy.estimators.ci import get_hmatel, simple_fci
from pauxy.systems.generic import Generic
from pauxy.systems.ueg import UEG
from pauxy.utils.from_pyscf import integrals_from_scf
from pauxy.utils.io import (
        read_qmcpack_wfn_hdf,
        write_qmcpack_wfn,
        dump_qmcpack_cholesky
        )
from pauxy.utils.misc import dotdict
from pauxy.utils.testing import get_random_wavefunction
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
        wfn, psi0 = read_qmcpack_wfn_hdf('wfn.h5')
        trial = MultiSlater(system, wfn, init=psi0)
        trial.calculate_energy(system)
        ndets = trial.ndets
        H = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
        S = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
        variational_energy_multi_det(system, trial.psi, trial.coeffs, H=H, S=S)
        e, ev = scipy.linalg.eigh(H,S)
        evar = variational_energy_multi_det(system, trial.psi, ev[:,0])
        self.assertAlmostEqual(e[0],0.15400990069739182)
        self.assertAlmostEqual(e[0],evar[0])

    # Todo: move to estimator tests.
    def test_slater_condon(self):
        mol = gto.M(atom=[('C', 0, 0, 0)], basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        ehf = mf.kernel()
        h1e, chol, ecore, oao = integrals_from_scf(mf, verbose=0, chol_cut=1e-5,
                                                   ortho_ao=False)
        nb = h1e.shape[0]
        system = Generic(nelec=mf.mol.nelec, h1e=h1e,
                         chol=chol.reshape((-1,nb,nb)),
                         ecore=ecore, verbose=0)
        eri = ao2mo.kernel(mol, mf.mo_coeff, aosym=1)
        system.oao = mf.mo_coeff
        cisolver = fci.direct_spin1.FCI(mol)
        # H_fci = fci.direct_spin1.pspace(h1e, eri, nb, mol.nelec)[1]
        # e_all, v_all = numpy.linalg.eigh(H_fci)

        e_fci, ci_fci = cisolver.kernel(h1e, eri, h1e.shape[1], mol.nelec,
                                        ecore=mol.energy_nuc())
        coeff, oa, ob = zip(*fci.addons.large_ci(ci_fci, mf.mo_coeff.shape[0],
                                                 mol.nelec, tol=0,
                                                 return_strs=False))
        # Unpack determinants into spin orbital basis.
        soa = [[2*x for x in o.tolist()] for o in oa]
        sob = [a+[2*x+1 for x in o.tolist()] for (a,o) in zip(soa,ob)]
        dets = [numpy.sort(numpy.array(x)) for x in sob]
        ndets = len(dets)

        H = numpy.zeros((ndets,ndets))
        for i in range(ndets):
            for j in range(i,ndets):
                hmatel = get_hmatel(system, dets[i],dets[j])
                H[i,j] = hmatel
        e_direct, ev_direct = scipy.linalg.eigh(H,lower=False)
        self.assertAlmostEqual(e_direct[0], e_fci)

    def test_phmsd(self):
        mol = gto.M(atom=[('Be', 0, 0, 0)], basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        ehf = mf.kernel()
        h1e, chol, ecore, oao = integrals_from_scf(mf, verbose=0, chol_cut=1e-8,
                                                   ortho_ao=False)
        nb = h1e.shape[0]
        system = Generic(nelec=mf.mol.nelec, h1e=h1e,
                         chol=chol.reshape((-1,nb,nb)),
                         ecore=ecore, verbose=0)
        (ee, eev), (dets, oa, ob) = simple_fci(system, dets=True)
        coeff = numpy.array(eev[:,0], dtype=numpy.complex128)
        # Test rediagonalisation
        options = {'rediag': True}
        wfn = (coeff,oa,ob)
        trial = MultiSlater(system,  wfn, verbose=False,
                            options=options)
        trial.calculate_energy(system)
        self.assertAlmostEqual(trial.energy, -14.403655108067667)
        numpy.random.seed(7)
        init = get_random_wavefunction(system.nelec, system.nbasis)
        na = system.nup
        write_qmcpack_wfn('wfn.phmsd.h5', wfn, 'uhf', system.nelec,
                          system.nbasis, init=[init[:,:na].copy(),
                              init[:,na:].copy()])
        write_qmcpack_wfn('wfn.nomsd.h5', (trial.coeffs, trial.psi), 'uhf', system.nelec,
                          system.nbasis, init=[init[:,:na].copy(),
                              init[:,na:].copy()])
        system.write_integrals()
        nume = 0
        deno = 0
        for i in range(trial.ndets):
            psia = trial.psi[i,:,:na]
            psib = trial.psi[i,:,na:]
            oa = numpy.dot(psia.conj().T, init[:,:na])
            ob = numpy.dot(psib.conj().T, init[:,na:])
            isa = numpy.linalg.inv(oa)
            isb = numpy.linalg.inv(ob)
            ovlp = numpy.linalg.det(oa)*numpy.linalg.det(ob)
            ga = numpy.dot(init[:,:system.nup], numpy.dot(isa, psia.conj().T)).T
            gb = numpy.dot(init[:,system.nup:], numpy.dot(isb, psib.conj().T)).T
            e = local_energy(system, numpy.array([ga,gb]), opt=False)[0]
            nume += trial.coeffs[i].conj()*ovlp*e
            deno += trial.coeffs[i].conj()*ovlp
        print(nume/deno,nume,deno)
        # TODO : Move to simple read / write test.
        # wfn, psi0 = read_qmcpack_wfn_hdf('wfn.phmsd.h5')
        # trial = MultiSlater(system, wfn, init=psi0, options={'rediag': True})
        # trial.calculate_energy(system)
