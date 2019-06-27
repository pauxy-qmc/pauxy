import itertools
import numpy
import os
import unittest
from pyscf import gto, ao2mo, scf
from pauxy.systems.generic import Generic
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.propagation.generic import GenericContinuous
from pauxy.utils.from_pyscf import (
        integrals_from_scf, integrals_from_chkfile
        )
from pauxy.utils.misc import dotdict
from pauxy.utils.linalg import modified_cholesky
from pauxy.walkers.multi_det import MultiDetWalker

class TestGenericPropagator(unittest.TestCase):

    def test_multi_det(self):
        nb = 10
        numpy.random.seed(5)
        h1e = numpy.random.rand(nb*nb).reshape(nb,nb)
        h1e = 0.5*(h1e + h1e.T)
        eri = numpy.random.rand(nb**4).reshape(nb*nb,nb*nb)
        eri = numpy.dot(eri, eri.T)
        chol = modified_cholesky(eri, 1e-5, verbose=False, cmax=40)
        chol = chol.reshape((-1,nb,nb))
        options = {'sparse': False}
        system = Generic(nelec=(5,5), h1e=h1e, chol=chol,
                         ecore=0, inputs=options)
        a = numpy.random.rand(3*system.nbasis*(system.nup+system.ndown))
        b = numpy.random.rand(3*system.nbasis*(system.nup+system.ndown))
        wfn = (a + 1j*b).reshape((3,system.nbasis,system.nup+system.ndown))
        coeffs = numpy.array([0.5+0j,0.3+0j,0.1+0j])
        trial = MultiSlater(system, (coeffs, wfn))
        # walker = MultiDetWalker({}, system, trial)
        qmc = dotdict({'dt': 0.005, 'nstblz': 5})
        prop = GenericContinuous(system, trial, qmc)
        # Test PH type wavefunction.
        orbs = numpy.arange(system.nbasis)
        oa = [c for c in itertools.combinations(orbs, system.nup)]
        ob = [c for c in itertools.combinations(orbs, system.ndown)]
        oa, ob = zip(*itertools.product(oa,ob))
        oa = oa[:5]
        ob = ob[:5]
        coeffs = numpy.array([0.9, 0.01, 0.01, 0.02, 0.04],
                             dtype=numpy.complex128)
        wfn = (coeffs,oa,ob)
        a = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
        b = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
        init = (a + 1j*b).reshape((system.nbasis,system.nup+system.ndown))
        trial = MultiSlater(system, wfn, init=init)
        prop = GenericContinuous(system, trial, qmc)
        walker = MultiDetWalker({}, system, trial)
        fb = prop.construct_force_bias(system, walker, trial)
        vhs = prop.construct_VHS(system, fb)

if __name__ == '__main__':
    unittest.main()
