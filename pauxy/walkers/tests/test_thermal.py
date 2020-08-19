import numpy
import pytest
from pauxy.systems.hubbard import Hubbard
from pauxy.trial_density_matrices.onebody import OneBody
from pauxy.walkers.thermal import ThermalWalker
from pauxy.utils.misc import dotdict
from pauxy.estimators.thermal import (
        greens_function,
        one_rdm_from_G
        )

@pytest.mark.unit
def test_greens_function():
    numpy.random.seed(7)
    options = {'nx': 4, 'ny': 4, 'U': 1, 'mu': 2.0, 'nup': 8, 'ndown': 8}
    system = Hubbard(options, verbose=False)
    beta = 4.0
    dt = 0.05
    nslice = int(round(beta/dt))
    trial = OneBody(system, beta, dt, verbose=False)
    ss = 1
    walker_a = ThermalWalker(system, trial, verbose=True,
                             walker_opts={'stack_size': ss})
    walker_b = ThermalWalker(system, trial,
                             walker_opts={'stack_size': ss},
                             verbose=True)
    for i in range(nslice):
        B = numpy.random.random(2*system.nbasis*system.nbasis).reshape(2,system.nbasis,system.nbasis)
        walker_b.stack.update_new(B)
        walker_a.stack.update(B)
    G1 = walker_a.greens_function_svd(trial, inplace=False)
    G2 = walker_a.greens_function_qr(trial, inplace=False)
    G3 = walker_b.greens_function_qr_strat(trial, inplace=False)
    assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0, abs=1e-8)
    assert numpy.linalg.norm(G1-G3) == pytest.approx(0.0, abs=1e-8)
    assert numpy.linalg.norm(G2-G3) == pytest.approx(0.0, abs=1e-8)

@pytest.mark.unit
def test_greens_function_complex():
    options = {'nx': 4, 'ny': 4, 'U': 1, 'mu': 2.0, 'nup': 8, 'ndown': 8}
    system = Hubbard(options, verbose=False)
    beta = 4
    dt = 0.05
    nslice = int(round(beta/dt))
    trial = OneBody(system, beta, dt, verbose=False)
    ss = 1
    walker_a = ThermalWalker(system, trial, verbose=True,
                             walker_opts={'stack_size': ss})
    walker_b = ThermalWalker(system, trial,
                             walker_opts={'stack_size': ss},
                             verbose=True)
    numpy.random.seed(7)
    for i in range(nslice):
        X = numpy.random.random(2*system.nbasis*system.nbasis).reshape(2,system.nbasis,system.nbasis)
        Y = numpy.random.random(2*system.nbasis*system.nbasis).reshape(2,system.nbasis,system.nbasis)
        B = X + 1j*Y
        walker_b.stack.update_new(B)
        walker_a.stack.update(B)
    numpy.random.seed(7)
    G1 = walker_a.greens_function_svd(trial, inplace=False)
    G2 = walker_a.greens_function_qr(trial, inplace=False)
    G3 = walker_b.greens_function_qr_strat(trial, inplace=False)
    assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0, abs=1e-8)
    assert numpy.linalg.norm(G1-G3) == pytest.approx(0.0, abs=1e-8)
    assert numpy.linalg.norm(G2-G3) == pytest.approx(0.0, abs=1e-8)
