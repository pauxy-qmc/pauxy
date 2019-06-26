import copy
import numpy
import unittest
from pauxy.estimators.ci import simple_fci
from pauxy.propagation.continuous import Continuous
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.utils.testing import (
        get_random_generic,
        get_random_wavefunction,
        get_random_phmsd,
        get_test_mol
        )
from pauxy.utils.misc import dotdict
from pauxy.utils.io import write_qmcpack_wfn
from pauxy.walkers.multi_det import MultiDetWalker

class TestContinuous(unittest.TestCase):

    def test_multi_det_generic(self):
        numpy.random.seed(7)
        system = get_test_mol()
        system.write_integrals()
        (e0, ev), (d,oa,ob) = simple_fci(system, dets=True)
        qmc = dotdict({'dt': 0.005})
        init = get_random_wavefunction((2,2), 5)
        na = system.nup
        init[:,na:] = init[:,:na].copy()
        options = {'rediag': True}
        trial_md = MultiSlater(system, (ev[:100,0],oa[:100],ob[:100]),
                               init=init, options=options)
        trial_md.calculate_energy(system)
        # TODO: Move this test to trial wavefunction
        # sys_T = copy.deepcopy(system)
        # sys_T.chol_vecs[:,:,:] = 0.0
        # trial_md.calculate_energy(sys_T)
        # print(trial_md.energy)
        # print(trial_md.contract_one_body(sys_T.H1[0]))
        trial_md.write_wavefunction(filename='wfn.md.h5',
                                    init=[init[:,:na].copy(),
                                          init[:,:na].copy()],
                                    occs=True)
        trial_sd = MultiSlater(system, (ev[:1,0],oa[:1],ob[:1]), init=init)
        trial_sd.write_wavefunction(filename='wfn.sd.h5',
                                    init=[init[:,:na].copy(),
                                          init[:,:na].copy()])
        trial_sd.calculate_energy(system)
        propg_md = Continuous(system, trial_md, qmc)
        propg_sd = Continuous(system, trial_sd, qmc)
        walker_md = MultiDetWalker({}, system, trial_md)
        walker_sd = MultiDetWalker({}, system, trial_sd)
        propg_md.propagate_walker(walker_md, system, trial_md, 0.0)
        propg_sd.propagate_walker(walker_sd, system, trial_sd, 0.0)
        # print(walker_md.ot, walker_sd.ot,walker_md.weight,walker_sd.weight)
        # print(numpy.max(numpy.abs(propg_md.propagator.mf_shift)),
        # numpy.max(numpy.abs(propg_sd.propagator.mf_shift)))
