from pauxy.systems.hubbard import Hubbard
from greens import (
        update_greens_function,
        recompute_greens_function,
        propagate_greens_function
        )
from utils import get_aux_fields
from utils import get_one_body
from pauxy.walkers.stack import PropagatorStack
import numpy
import pytest

@pytest.mark.unit
def test_update_spin():
    system = Hubbard({'nx': 6, 'ny': 1, 'U': 4.0, 'mu': 2, 'nup': 3, 'ndown': 3})
    numpy.random.seed(7)
    beta = 0.5
    dt = 0.05
    nslice = 10
    stack = PropagatorStack(1, nslice, system.nbasis, numpy.complex128, lowrank=False)
    fields = numpy.random.randint(0, 2, nslice*system.nbasis).reshape(nslice, system.nbasis)
    charge_decomp = False
    gamma, auxf, delta, aux_wfac = get_aux_fields(system, dt, charge_decomp)
    BH1, BH1inv = get_one_body(system, dt)
    G1 = recompute_greens_function(fields, stack, auxf, BH1, time_slice=0,
                                    from_scratch=True)
    for i in range(system.nbasis):
        if fields[0,i] == 0:
            fields[0,i] = 1
            update_greens_function(G1, i, 1, delta)
        else:
            fields[0,i] = 0
            update_greens_function(G1, i, 0, delta)
    G2 = recompute_greens_function(fields, stack, auxf, BH1, time_slice=0,
                                   from_scratch=True)
    G3 = recompute_greens_function(fields, stack, auxf, BH1, time_slice=0,
                                   from_scratch=False)
    print(G1[0,0,0], G2[0,0,0])
    assert False
    assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0)
    assert numpy.linalg.norm(G1-G3) == pytest.approx(0.0)

@pytest.mark.unit
def test_update_charge():
    system = Hubbard({'nx': 6, 'ny': 1, 'U': 4.0, 'mu': 2, 'nup': 3, 'ndown': 3})
    beta = 2.0
    dt = 0.05
    nslice = int(round(beta/dt))
    charge_decomp = True
    stack = PropagatorStack(1, nslice, system.nbasis, numpy.complex128, lowrank=False)
    fields = numpy.random.randint(0, 2, nslice*system.nbasis).reshape(nslice, system.nbasis)
    gamma, auxf, delta, aux_wfac = get_aux_fields(system, dt, charge_decomp)
    BH1, BH1inv = get_one_body(system, dt)
    G1 = recompute_greens_function(fields, stack, auxf, BH1, time_slice=0,
                                   from_scratch=True)
    for i in range(system.nbasis):
        if fields[0,i] == 0:
            fields[0,i] = 1
            update_greens_function(G1, i, 1, delta)
        else:
            fields[0,i] = 0
            update_greens_function(G1, i, 0, delta)
    G2 = recompute_greens_function(fields, stack, auxf, BH1, time_slice=0,
                                   from_scratch=True)
    G3 = recompute_greens_function(fields, stack, auxf, BH1, time_slice=0,
                                   from_scratch=False)
    assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0)
    assert numpy.linalg.norm(G1-G3) == pytest.approx(0.0)

@pytest.mark.unit
def test_update_stack_size():
    system = Hubbard({'nx': 6, 'ny': 1, 'U': 4.0, 'mu': 2, 'nup': 3, 'ndown': 3})
    beta = 2.0
    dt = 0.01
    nslice = 10
    stack_a = PropagatorStack(1, nslice, system.nbasis, numpy.complex128, lowrank=False)
    stack_b = PropagatorStack(10, nslice, system.nbasis, numpy.complex128, lowrank=False)
    fields = numpy.random.randint(0, 2, nslice*system.nbasis).reshape(nslice, system.nbasis)
    charge_decomp = False
    gamma, auxf, delta, aux_wfac = get_aux_fields(system, dt, charge_decomp)
    BH1, BH1inv = get_one_body(system, dt)
    G1 = recompute_greens_function(fields, stack_a, auxf, BH1,
                                   time_slice=nslice,
                                   from_scratch=True)
    G2 = recompute_greens_function(fields, stack_b, auxf, BH1,
                                    time_slice=nslice,
                                    from_scratch=True)
    for islice in range(nslice):
        G1 = propagate_greens_function(G1, fields[(islice+1)%nslice], BH1inv, BH1, auxf)
        G2 = propagate_greens_function(G2, fields[(islice+1)%nslice], BH1inv, BH1, auxf)
        for i in range(system.nbasis):
            if fields[0,i] == 0:
                fields[0,i] = 1
                update_greens_function(G1, i, 1, delta)
                update_greens_function(G2, i, 1, delta)
            else:
                fields[0,i] = 0
                update_greens_function(G1, i, 0, delta)
                update_greens_function(G2, i, 0, delta)
        assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0)
        if islice % 10 == 0 and islice != 0:
            Gx = recompute_greens_function(fields, stack_a, auxf, BH1,
                                           time_slice=islice)
            Gy = recompute_greens_function(fields, stack_b, auxf, BH1,
                                           time_slice=islice)
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
    charge_decomp = True
    stack = PropagatorStack(1, nslice, system.nbasis, numpy.complex128, lowrank=False)
    fields = numpy.random.randint(0, 2, nslice*system.nbasis).reshape(nslice, system.nbasis)
    gamma, auxf, delta, aux_wfac = get_aux_fields(system, dt, charge_decomp)
    BH1, BH1inv = get_one_body(system, dt)
    for islice in range(nslice):
        G1 = recompute_greens_function(fields, stack, auxf,
                                       BH1, time_slice=islice,
                                       from_scratch=True)
        G1 = propagate_greens_function(G1, fields[(islice+1)%nslice], BH1inv, BH1, auxf)
        G2 = recompute_greens_function(fields, stack, auxf,
                                       BH1, time_slice=(islice+1)%nslice,
                                       from_scratch=True)
        assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0, abs=1e-10)

@pytest.mark.unit
def test_stack_size():
    system = Hubbard({'nx': 6, 'ny': 1, 'U': 4.0, 'mu': 2, 'nup': 3, 'ndown': 3})
    beta = 2.0
    dt = 0.05
    nslice = int(round(beta/dt))
    charge_decomp = True
    stack_a = PropagatorStack(1, nslice, system.nbasis, numpy.complex128, lowrank=False)
    stack_b = PropagatorStack(5, nslice, system.nbasis, numpy.complex128, lowrank=False)
    stack_c = PropagatorStack(10, nslice, system.nbasis, numpy.complex128, lowrank=False)
    fields = numpy.random.randint(0, 2, nslice*system.nbasis).reshape(nslice, system.nbasis)
    gamma, auxf, delta, aux_wfac = get_aux_fields(system, dt, charge_decomp)
    BH1, BH1inv = get_one_body(system, dt)
    G1 = recompute_greens_function(fields, stack_a, auxf,
                                    BH1, time_slice=nslice,
                                    from_scratch=True)
    G2 = recompute_greens_function(fields, stack_b, auxf,
                                   BH1, time_slice=nslice,
                                   from_scratch=True)
    G3 = recompute_greens_function(fields, stack_c, auxf,
                                   BH1, time_slice=nslice,
                                   from_scratch=True)
    assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0)
    assert numpy.linalg.norm(G1-G3) == pytest.approx(0.0)
