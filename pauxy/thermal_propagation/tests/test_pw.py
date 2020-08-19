import numpy
import pytest
from pauxy.systems.ueg import UEG
from pauxy.trial_density_matrices.onebody import OneBody
from pauxy.thermal_propagation.planewave import PlaneWave
from pauxy.walkers.thermal import ThermalWalker
from pauxy.utils.misc import dotdict

options = {'rs': 2.0, 'nup': 7, 'ndown': 7, 'mu': 1.0, 'ecut': 2.0}
system = UEG(options, verbose=False)
beta = 4.0
dt = 0.05
nslice = int(round(beta/dt))
trial = OneBody(system, beta, dt)
qmc = dotdict({'dt': dt, 'nstblz': 10})

@pytest.mark.unit
def test_full_rank():
    prop = PlaneWave(system, trial, qmc, verbose=False)
    numpy.random.seed(7)
    walker_a = ThermalWalker(system, trial,
                             walker_opts={'stack_size': 1, 'low_rank': False},
                             verbose=False)
    prop.propagate_walker_phaseless(system, walker_a, 0, 0)
    numpy.random.seed(7)
    assert walker_a.weight-1.7149706389545953 == pytest.approx(0.0)
    walker_b = ThermalWalker(system, trial,
                             walker_opts={'stack_size': 1, 'low_rank': False},
                             verbose=False)
    numpy.random.seed(7)
    prop.propagate_walker_free(system, walker_b, 0, 0)
    assert walker_b.weight-1.71497087042060 == pytest.approx(0.0)

@pytest.mark.unit
def test_low_rank():
    prop = PlaneWave(system, trial, qmc, verbose=False, lowrank=True)
    numpy.random.seed(7)
    walker_a = ThermalWalker(system, trial,
                             walker_opts={'stack_size': 1, 'low_rank': True},
                             verbose=False)
    prop.propagate_walker_phaseless(system, walker_a, 0, 0)
    numpy.random.seed(7)
    assert walker_a.weight-1.7149706389545953 == pytest.approx(0.0, abs=1e-6)
    walker_b = ThermalWalker(system, trial,
                             walker_opts={'stack_size': 1, 'low_rank': True},
                             verbose=False)
    numpy.random.seed(7)
    prop.propagate_walker_free(system, walker_b, 0, 0)
    assert walker_b.weight-1.71497087042060 == pytest.approx(0.0, abs=1e-6)
