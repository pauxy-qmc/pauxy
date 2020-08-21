import numpy
import pytest
from pauxy.estimators.thermal import greens_function_qr_strat
from pauxy.systems.hubbard import Hubbard
from pauxy.walkers.stack import PropagatorStack
from pauxy.dqmc_updates.hubbard import DiscreteHubbard

@pytest.mark.unit
def test_update_spin():
    system = Hubbard({'nx': 6, 'ny': 1, 'U': 4.0, 'mu': 2, 'nup': 3, 'ndown': 3})
    numpy.random.seed(7)
    beta = 0.5
    dt = 0.05
    nslice = 10
    prop = DiscreteHubbard(system, dt, nslice)
    G1 = greens_function_qr_strat(prop.stack, slice_ix=0)
    for i in range(system.nbasis):
        if prop.fields[0,i] == 0:
            prop.fields[0,i] = 1
            prop.update_greens_function(G1, i, 1)
        else:
            prop.fields[0,i] = 0
            prop.update_greens_function(G1, i, 0)
    B = prop.construct_bmatrix(prop.fields[0])
    prop.stack.update(B)
    G2 = greens_function_qr_strat(prop.stack, slice_ix=0)
    assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0)

@pytest.mark.unit
def test_update_charge():
    system = Hubbard({'nx': 6, 'ny': 1, 'U': 4.0, 'mu': 2, 'nup': 3, 'ndown': 3})
    beta = 2.0
    dt = 0.05
    nslice = int(round(beta/dt))
    prop = DiscreteHubbard(system, dt, nslice, charge_decomp=True)
    G1 = greens_function_qr_strat(prop.stack, slice_ix=0)
    for i in range(system.nbasis):
        if prop.fields[0,i] == 0:
            prop.fields[0,i] = 1
            prop.update_greens_function(G1, i, 1)
        else:
            prop.fields[0,i] = 0
            prop.update_greens_function(G1, i, 0)
    B = prop.construct_bmatrix(prop.fields[0])
    prop.stack.update(B)
    G2 = greens_function_qr_strat(prop.stack, slice_ix=0)
    assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0)

@pytest.mark.unit
def test_update_stack_size():
    system = Hubbard({'nx': 6, 'ny': 1, 'U': 4.0, 'mu': 2, 'nup': 3, 'ndown': 3})
    beta = 2.0
    dt = 0.01
    nslice = 10
    numpy.random.seed(7)
    prop_a = DiscreteHubbard(system, dt, nslice, stack_size=1)
    numpy.random.seed(7)
    prop_b = DiscreteHubbard(system, dt, nslice, stack_size=10)
    G1 = greens_function_qr_strat(prop_a.stack, slice_ix=nslice)
    G2 = greens_function_qr_strat(prop_b.stack, slice_ix=nslice)
    for islice in range(nslice):
        G1 = prop_a.propagate_greens_function(G1, (islice+1)%nslice)
        G2 = prop_b.propagate_greens_function(G2, (islice+1)%nslice)
        for i in range(system.nbasis):
            if prop_a.fields[0,i] == 0:
                prop_a.fields[0,i] = 1
                prop_a.update_greens_function(G1, i, 1)
                prop_b.fields[0,i] = 1
                prop_b.update_greens_function(G2, i, 1)
            else:
                prop_a.fields[0,i] = 0
                prop_a.update_greens_function(G1, i, 0)
                prop_b.fields[0,i] = 0
                prop_b.update_greens_function(G2, i, 0)
        assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0)
        if islice % 10 == 0 and islice != 0:
            G1 = greens_function_qr_strat(prop_a.stack, slice_ix=islice)
            G2 = greens_function_qr_strat(prop_b.stack, slice_ix=islice)
        else:
            Gx = G1
            Gy = G2
        assert numpy.linalg.norm(G1-Gx) == pytest.approx(0.0)
        assert numpy.linalg.norm(G2-Gy) == pytest.approx(0.0)
        assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0)

@pytest.mark.unit
def test_propagate():
    system = Hubbard({'nx': 6, 'ny': 1, 'U': 4.0, 'mu': 2, 'nup': 3, 'ndown': 3})
    beta = 2.0
    dt = 0.05
    nslice = int(round(beta/dt))
    numpy.random.seed(7)
    prop = DiscreteHubbard(system, dt, nslice, charge_decomp=True)
    for islice in range(nslice):
        G1 = greens_function_qr_strat(prop.stack, slice_ix=islice)
        G1 = prop.propagate_greens_function(G1, (islice+1)%nslice)
        G2 = greens_function_qr_strat(prop.stack, slice_ix=(islice+1)%nslice)
        assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0, abs=1e-10)

@pytest.mark.unit
def test_stack_size():
    system = Hubbard({'nx': 6, 'ny': 1, 'U': 4.0, 'mu': 2, 'nup': 3, 'ndown': 3})
    beta = 2.0
    dt = 0.05
    nslice = int(round(beta/dt))
    numpy.random.seed(7)
    prop_a = DiscreteHubbard(system, dt, nslice,
                             charge_decomp=True,
                             stack_size=1)
    numpy.random.seed(7)
    prop_b = DiscreteHubbard(system, dt, nslice,
                             charge_decomp=True,
                             stack_size=5)
    numpy.random.seed(7)
    prop_c = DiscreteHubbard(system, dt, nslice,
                             charge_decomp=True,
                             stack_size=10)
    G1 = greens_function_qr_strat(prop_a.stack, slice_ix=nslice)
    G2 = greens_function_qr_strat(prop_b.stack, slice_ix=nslice)
    G3 = greens_function_qr_strat(prop_c.stack, slice_ix=nslice)
    assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0)
    assert numpy.linalg.norm(G1-G3) == pytest.approx(0.0)
