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
        read_phfmol,
        dump_qmcpack_cholesky
        )
from pauxy.utils.misc import dotdict
from pauxy.utils.testing import get_random_wavefunction
from pauxy.trial_wavefunction.utils import get_trial_wavefunction
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.walkers.multi_det import MultiDetWalker

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
        import os
        path = os.path.dirname(os.path.abspath(__file__))
        wfn, psi0 = read_qmcpack_wfn_hdf(path+'/wfn.h5')
        trial = MultiSlater(system, wfn, init=psi0)
        trial.calculate_energy(system)
        ndets = trial.ndets
        H = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
        S = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
        variational_energy_multi_det(system, trial.psi, trial.coeffs, H=H, S=S)
        e, ev = scipy.linalg.eigh(H,S)
        evar = variational_energy_multi_det(system, trial.psi, ev[:,0])
        # self.assertAlmostEqual(e[0],0.15400990069739182)
        # self.assertAlmostEqual(e[0],evar[0])

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
        H_fci = fci.direct_spin1.pspace(h1e, eri, nb, mol.nelec)[1]
        e_all, v_all = numpy.linalg.eigh(H_fci)

        # e_fci, ci_fci = cisolver.kernel(h1e, eri, h1e.shape[1], mol.nelec,
                                        # ecore=mol.energy_nuc())
        # coeff, oa, ob = zip(*fci.addons.large_ci(ci_fci, mf.mo_coeff.shape[0],
                                                 # mol.nelec, tol=0,
                                                 # return_strs=False))
        (ee, eev), (dets, oa, ob) = simple_fci(system, gen_dets=True)
        self.assertTrue(numpy.allclose(ee, e_all))

    def test_phmsd(self):
        mol = gto.M(atom=[('Be', 0, 0, 0)], basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        ehf = mf.kernel()
        # print(ehf)
        h1e, chol, ecore, oao = integrals_from_scf(mf, verbose=0, chol_cut=1e-8,
                                                   ortho_ao=False)
        nb = h1e.shape[0]
        chol=chol.reshape((-1,nb,nb))
        eri = numpy.einsum('nik,njl->ikjl', chol, chol).reshape(nb*nb,nb*nb)
        tools.fcidump.from_integrals('FCIDUMP', h1e, eri, nb,
                                     mol.nelectron, ms=0)
        system = Generic(nelec=mf.mol.nelec, h1e=h1e,
                         chol=chol,
                         ecore=ecore, verbose=0,
                         inputs={'sparse': False})
        system.write_integrals()
        HH = numpy.zeros((100,100))
        (ee, eev), (dets, oa, ob) = simple_fci(system, gen_dets=True)
        cisolver = fci.direct_spin1.FCI(mol)
        H_fci = fci.direct_spin1.pspace(h1e, eri, nb, mol.nelec)[1]
        e_all, v_all = numpy.linalg.eigh(H_fci)

        # e_fci, ci_fci = cisolver.kernel(h1e, eri, h1e.shape[1], mol.nelec,
                                        # ecore=mol.energy_nuc())
        coeff = numpy.array(eev[:,0], dtype=numpy.complex128)
        # keep = numpy.where(numpy.abs(coeff)>1e-8)
        # dets = numpy.array(dets)[keep]
        # eev[:,0]
        # ndet = len(oa)
        # output = open('wfn.dat', 'w')
        # nkeep = sum(abs(eev[:,0])>1e-12)
        # namelist = "&FCI\n UHF = 1\n NCI = %d\n TYPE = occ\n&END" % nkeep
        # output.write(namelist+'\n')
        # output.write("Configurations:"+'\n')
        # for idet in range(ndet):
            # if abs(eev[idet,0])>1e-12:
                # coeff = '%.13f'%eev[idet,0]
                # oup = ' '.join('{:d}'.format(x+1) for x in oa[idet])
                # odown = ' '.join('{:d}'.format(x+nb+1) for x in ob[idet])
                # output.write(coeff+' '+oup+' '+odown+'\n')
        # numer = 0
        # denom = 0
        # ix = numpy.where(numpy.abs(coeff)<1e-8)
        # coeff[ix] = 0.0
        # pco, pdet = read_phfmol('determinants1.det', 5, 2, 2)
        # trial = MultiSlater(system, (pco,pdet))
        # trial.psi = pdet
        # trial.coeff = pco
        # trial.recompute_ci_coeffs(system)
        # trial.init = numpy.zeros((system.nbasis,system.nup+system.ndown),dtype=numpy.complex128)
        # I = numpy.eye(system.nbasis, dtype=numpy.complex128)
        # trial.init[:,:system.nup] = I[:,:system.nup].copy()
        # trial.init[:,system.nup:] = I[:,:system.nup].copy()
        # # print(trial.coeff)
        # trial.calculate_energy(system)
        # # print("NOMSD: ", trial.energy)
        # walker = MultiDetWalker({}, system, trial)
        # # print("walker energy: ", walker.local_energy(system))
        # for i, di in enumerate(dets):
            # denom += coeff[i].conj()*coeff[i]
            # for j, dj in enumerate(dets):
                # hm = get_hmatel(system, di, dj)
                # HH[i,j] = hm
                # numer += coeff[i].conj() * coeff[j] * HH[i,j]
                # # if abs(hm) > 0:
                # # print(" HAM: {} {} {:13.12e} {:13.12e} {:13.12e} {:13.12e} ".format(i, j, hm.real, numer.real, denom.real, (coeff[i].conj()*coeff[j]).real))
        # numer = 0
        # denom = 0
        # for i in range(len(dets)):
            # numer += HH[i,0]*coeff[i].conj()
        # print(numer/coeff[0])
        # print(numpy.dot(coeff,numpy.dot(HH,coeff)),ee[0])
        # print(numer / denom)
        # Test rediagonalisation
        options = {'rediag': False}
        # e_fci, ci_fci = cisolver.kernel(h1e, eri, h1e.shape[1], mol.nelec,
                                        # ecore=mol.energy_nuc())
        # coeff, oa, ob = zip(*fci.addons.large_ci(ci_fci, mf.mo_coeff.shape[0],
                                                 # mol.nelec, tol=0,
                                                 # return_strs=False))
        wfn = (numpy.array(coeff,dtype=numpy.complex128),numpy.array(oa),numpy.array(ob))
        numpy.random.seed(7)
        init = get_random_wavefunction(system.nelec, system.nbasis)
        na = system.nup
        # print(wfn[0])
        trial = MultiSlater(system,  wfn, verbose=False,
                            options=options, init=init)
        trial.calculate_energy(system)
        # print(trial.coeffs, wfn[0])
        init_write = [init[:,:na].copy(), init[:,na:].copy()]
        write_qmcpack_wfn('wfn.phmsd.h5', wfn, 'uhf', system.nelec,
                          system.nbasis, init=None)
        # print(trial.energy, ee[0], e_fci)
        # # self.assertAlmostEqual(trial.energy, -14.403655108067667)
        # # write_qmcpack_wfn('wfn.nomsd.h5', (trial.coeffs, trial.psi), 'uhf', system.nelec,
                          # # system.nbasis, init=[init[:,:na].copy(),
                              # # init[:,na:].copy()])
        # # system.write_integrals()
        # walker = MultiDetWalker({}, system, trial)
        # mf_shift = [trial.contract_one_body(Vpq) for Vpq in system.hs_pot]
        # print(mf_shift)
        # print(walker.local_energy(system))
        # nume = 0
        # deno = 0
        # tot_ovlp = 0
        # # # print(init[:,:na])
        # for i in range(trial.ndets):
            # psia = trial.psi[i,:,:na]
            # psib = trial.psi[i,:,na:]
            # # oa = numpy.dot(psia.conj().T, init[:,:na])
            # # ob = numpy.dot(psib.conj().T, init[:,na:])
            # oa = numpy.dot(init[:,:na].conj().T, psia)
            # ob = numpy.dot(init[:,na:].conj().T, psib)
            # isa = numpy.linalg.inv(oa)
            # isb = numpy.linalg.inv(ob)
            # ovlp = numpy.linalg.det(oa)*numpy.linalg.det(ob)
            # # print("ovlp: ", tot_ovlp, trial.coeffs[i].conj(),
                  # # numpy.linalg.det(oa), numpy.linalg.det(ob),
                  # # trial.coeffs[i].conj()*ovlp)
            # tot_ovlp += trial.coeffs[i]*ovlp
            # # ga = numpy.dot(init[:,:system.nup], numpy.dot(isa, psia.conj().T)).T
            # # gb = numpy.dot(init[:,system.nup:], numpy.dot(isb, psib.conj().T)).T
            # ga = numpy.dot(psia, numpy.dot(isa, init[:,:na].conj().T)).T
            # gb = numpy.dot(psib, numpy.dot(isb, init[:,na:].conj().T)).T
            # e = local_energy(system, numpy.array([ga,gb]), opt=False)[0]
            # nume += trial.coeffs[i]*ovlp*e
            # deno += trial.coeffs[i]*ovlp
        # print(tot_ovlp)
        # print(nume/deno,nume,deno,trial.energy)
        # TODO : Move to simple read / write test.
        # wfn, psi0 = read_qmcpack_wfn_hdf('wfn.phmsd.h5')
        # trial = MultiSlater(system, wfn, init=psi0, options={'rediag': True})
        # trial.calculate_energy(system)

if __name__ == '__main__':
    unittest.main()
