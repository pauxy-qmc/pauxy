import numpy
import os
import unittest
from pyscf import gto, ao2mo, scf
from pauxy.systems.generic import Generic
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.utils.from_pyscf import integrals_from_scf, integrals_from_chkfile
from pauxy.utils.misc import dotdict
from pauxy.walkers.multi_det import MultiDetWalker

class TestMultiDetWalker(unittest.TestCase):

    def test_nomsd_walker(self):
        system = dotdict({'nup': 5, 'ndown': 5, 'nbasis': 10,
                          'nelec': (5,5), 'ne': 10})
        numpy.random.seed(7)
        a = numpy.random.rand(3*system.nbasis*(system.nup+system.ndown))
        b = numpy.random.rand(3*system.nbasis*(system.nup+system.ndown))
        wfn = (a + 1j*b).reshape((3,system.nbasis,system.nup+system.ndown))
        coeffs = numpy.array([0.5+0j,0.3+0j,0.1+0j])
        trial = MultiSlater(system, (coeffs, wfn))
        walker = MultiDetWalker({}, system, trial)
        def calc_ovlp(a,b):
            return numpy.linalg.det(numpy.dot(a.conj().T, b))
        ovlp = 0.0+0j
        na = system.nup
        pa = trial.psi[0,:,:na]
        pb = trial.psi[0,:,na:]
        for i, d in enumerate(trial.psi):
            ovlp += coeffs[i].conj()*calc_ovlp(d[:,:na],pa)*calc_ovlp(d[:,na:],pb)
        self.assertAlmostEqual(ovlp.real,walker.ovlp.real)
        self.assertAlmostEqual(ovlp.imag,walker.ovlp.imag)
