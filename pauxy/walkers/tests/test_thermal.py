import numpy
import pytest
from pauxy.systems.hubbard import Hubbard
from pauxy.systems.ueg import UEG
from pauxy.trial_density_matrices.onebody import OneBody
from pauxy.walkers.thermal import ThermalWalker
from pauxy.utils.misc import dotdict
from pauxy.estimators.mixed import local_energy
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
def test_greens_function_low_rank():
    numpy.random.seed(7)
    # options = {'nx': 4, 'ny': 4, 'U': 4, 'mu': 2.0, 'nup': 1, 'ndown': 1}
    # system = Hubbard(options, verbose=False)
    system = UEG({'rs': 4.0, 'nup': 7, 'ndown': 7, 'ecut': 2.5})
    beta = 1.0
    dt = 0.05
    nslice = int(round(beta/dt))
    trial = OneBody(system, beta, dt, verbose=False)
    walker_a = ThermalWalker(system, trial, verbose=True,
            walker_opts={'low_rank': False, 'stack_size': 1})
    walker_b = ThermalWalker(system, trial, verbose=True,
            walker_opts={'low_rank': True,
                         'stack_size': 1,
                         'low_rank_thresh': 1e-6})
    numpy.random.seed(7)
    for i in range(nslice):
        X = numpy.random.random(2*system.nbasis*system.nbasis).reshape(2,system.nbasis,system.nbasis)
        Y = numpy.random.random(2*system.nbasis*system.nbasis).reshape(2,system.nbasis,system.nbasis)
        B = (X + 1j*Y)
        walker_b.stack.update_new(B)
        walker_a.stack.update_new(B)
    G1 = walker_a.greens_function(None, slice_ix=None, inplace=False)
    G2 = walker_b.greens_function(None, slice_ix=None, inplace=False)
    # assert False
    assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0, abs=1e-5)

@pytest.mark.unit
def test_greens_function_low_rank_non_diag():
    numpy.random.seed(7)
    options = {'nx': 6, 'ny': 1, 'U': 1, 'mu': 2.0, 'nup': 3, 'ndown': 3}
    system = Hubbard(options, verbose=False)
    beta = 0.4
    dt = 0.05
    nslice = int(round(beta/dt))
    trial = OneBody(system, beta, dt, verbose=False)
    trial_ = OneBody(system, beta, dt, verbose=False)
    ss = 1
    H1 = system.H1
    eigs, eigv = numpy.linalg.eigh(H1[0])
    dmat = numpy.diag(numpy.exp(-dt*(eigs-system.mu)))
    def fermi_factor(ek, mu, beta):
        return 1.0 / (numpy.exp(beta*(ek-mu))+1.0)
    et = numpy.dot(eigs, fermi_factor(eigs, trial.mu, beta))
    dmat_inv = numpy.diag(numpy.exp(dt*(eigs-system.mu)))
    walker_a = ThermalWalker(system, trial_, verbose=True,
            walker_opts={'low_rank': True, 'stack_size': 1})
    walker_c = ThermalWalker(system, trial_, verbose=True,
            walker_opts={'low_rank': False, 'stack_size': 1})
    import scipy.linalg
    left = scipy.linalg.expm(-(beta)*(H1[0]-trial.mu*numpy.eye(6)))
    right = numpy.eye(6)
    GG = greens_function(numpy.dot(left,right))#
    for i in range(nslice):
        X = numpy.random.random(2*system.nbasis*system.nbasis).reshape(2,system.nbasis,system.nbasis)
        Y = numpy.random.random(2*system.nbasis*system.nbasis).reshape(2,system.nbasis,system.nbasis)
        B = (X + 1j*Y)
        left = numpy.dot(left, trial.dmat_inv[0])
        right = numpy.dot(B[0], right)
        Bprod = numpy.dot(left, right)
        GG = greens_function(Bprod)#
        walker_a.stack.update_low_rank_non_diag(B.astype(numpy.complex128))
        G1 = walker_a.greens_function(None, slice_ix=walker_a.stack.ntime_slices, inplace=False)
        walker_c.stack.update_new(B.astype(numpy.complex128))
        G2 = walker_c.greens_function(None, slice_ix=walker_a.stack.ntime_slices, inplace=False)
    d1 = numpy.linalg.slogdet(G2[0])
    d1 = d1[0].conj()*numpy.exp(-d1[1])
    delta = d1-walker_a.stack.sgndet[0]*numpy.exp(walker_a.stack.logdet[0])
    assert abs(delta) == pytest.approx(0.0, abs=1e-6)
    assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0)

@pytest.mark.unit
def test_greens_function_complex():
    options = {'nx': 4, 'ny': 4, 'U': 1, 'mu': 2.0, 'nup': 8, 'ndown': 8}
    system = Hubbard(options, verbose=False)
    beta = 0.5
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
    G1 = walker_a.greens_function_svd(trial, slice_ix=0, inplace=False)
    G2, ld = walker_a.greens_function_qr(trial, slice_ix=0, inplace=False)
    G3 = walker_b.greens_function_qr_strat(trial, slice_ix=0, inplace=False)
    sla = numpy.linalg.slogdet(G2[0])
    slb = numpy.linalg.slogdet(G2[0])
    print(ld, sla[0]*(-sla[1])+slb[0]*(-slb[1]))
    assert False
    assert numpy.linalg.norm(G1-G2) == pytest.approx(0.0, abs=1e-8)
    assert numpy.linalg.norm(G1-G3) == pytest.approx(0.0, abs=1e-8)
    assert numpy.linalg.norm(G2-G3) == pytest.approx(0.0, abs=1e-8)
