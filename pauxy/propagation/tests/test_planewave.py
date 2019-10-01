import numpy
import unittest
from pauxy.systems.ueg import UEG
from pauxy.systems.generic import Generic
from pauxy.propagation.planewave import PlaneWave
from pauxy.walkers.single_det import SingleDetWalker
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.utils.misc import dotdict


class TestPlanewavePropagator(unittest.TestCase):

    def test_pw(self):
        options = {'rs': 2, 'nup': 7, 'ndown': 7, 'ecut': 2,
                   'write_integrals': True}
        system = UEG(inputs=options)
        occ = numpy.eye(system.nbasis)[:,:system.nup]
        wfn = numpy.zeros((1,system.nbasis,system.nup+system.ndown),
                          dtype=numpy.complex128)
        wfn[0,:,:system.nup] = occ
        wfn[0,:,system.nup:] = occ
        coeffs = numpy.array([1+0j])
        trial = MultiSlater(system, (coeffs, wfn))
        trial.psi = trial.psi[0]
        qmc = dotdict({'dt': 0.005, 'nstblz': 5})
        prop = PlaneWave(system, trial, qmc)
        walker = SingleDetWalker({}, system, trial)
        numpy.random.seed(7)
        a = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
        b = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
        wfn = (a + 1j*b).reshape((system.nbasis,system.nup+system.ndown))
        walker.phi = wfn.copy()
        walker.greens_function(trial)
        # fb = prop.construct_force_bias_slow(system, walker, trial)
        fb = prop.construct_force_bias(system, walker, trial)
        self.assertAlmostEqual(numpy.linalg.norm(fb), 0.16660828645573392)
        xi = numpy.random.rand(system.nfields)
        vhs = prop.construct_VHS(system, xi-fb)
        self.assertAlmostEqual(numpy.linalg.norm(vhs), 0.1467322554815581)

if __name__ == '__main__':
    unittest.main()
