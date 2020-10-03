import numpy
import pytest
from pauxy.systems.ueg import UEG
from pauxy.estimators.ueg import fock_ueg, local_energy_ueg
from pauxy.estimators.greens_function import gab
from pauxy.utils.testing import get_random_wavefunction

@pytest.mark.unit
def test_fock_build():
    sys = UEG({'rs': 2.0, 'ecut': 2, 'nup': 7, 'ndown': 7, 'thermal': True})
    numpy.random.seed(7)
    psi = get_random_wavefunction(sys.nelec, sys.nbasis).real
    trial = numpy.eye(sys.nbasis, sys.nelec[0])
    G = numpy.array([gab(psi[:,:sys.nup], psi[:,:sys.nup]),
                     gab(psi[:,sys.nup:], psi[:,sys.nup:])])
    nb = sys.nbasis
    # from pyscf import gto, scf, ao2mo
    # mol = gto.M()
    # mol.nelec = sys.nelec
    # mf = scf.UHF(mol)
    # U = sys.compute_real_transformation()
    # h1_8 = numpy.dot(U.conj().T, numpy.dot(sys.H1[0], U))
    # mf.get_hcore = lambda *args: h1_8
    # mf.get_ovlp = lambda *args: numpy.eye(nb)
    # mf._eri = sys.eri_8()
    # mf._eri = ao2mo.restore(8, eri_8, nb)
    # veff = mf.get_veff(dm=dm)
    eris = sys.eri_4()
    F, J, K = fock_ueg(sys, G)
    vj = numpy.einsum('pqrs,xqp->xrs', eris, G)
    vk = numpy.einsum('pqrs,xqr->xps', eris, G)
    fock = numpy.zeros((2,33,33))
    fock[0] = sys.H1[0] + vj[0] + vj[1] - vk[0]
    fock[1] = sys.H1[1] + vj[0] + vj[1] - vk[1]
    assert numpy.linalg.norm(fock - F) == pytest.approx(0.0)
