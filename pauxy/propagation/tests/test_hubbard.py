import numpy
import pytest
from pauxy.systems.hubbard import Hubbard, decode_basis
from pauxy.propagation.hubbard import HirschSpin
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.trial_wavefunction.uhf import UHF
from pauxy.walkers.single_det import SingleDetWalker
from pauxy.utils.misc import dotdict
from pauxy.estimators.greens_function import gab

options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
system = Hubbard(inputs=options)
eigs, eigv = numpy.linalg.eigh(system.H1[0])
coeffs = numpy.array([1.0+0j])
wfn = numpy.zeros((1,system.nbasis,system.ne))
wfn[0,:,:system.nup] = eigv[:,:system.nup].copy()
wfn[0,:,system.nup:] = eigv[:,:system.ndown].copy()
trial = MultiSlater(system, (coeffs, wfn))
trial.psi = trial.psi[0]

@pytest.mark.unit
def test_hubbard_spin():
    qmc = dotdict({'dt': 0.01, 'nstblz': 5})
    prop = HirschSpin(system, trial, qmc)
    walker = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    numpy.random.seed(7)
    nup = system.nup
    prop.propagate_walker_constrained(walker, system, trial, 0.0)
    walker_ref = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    # Alpha electrons
    walker_ref.phi[:,:nup] = numpy.dot(prop.bt2[0], walker_ref.phi[:,:nup])
    BV = numpy.diag([prop.auxf[int(x.real),0] for x in walker.field_configs.configs[0]])
    walker_ref.phi[:,:nup] = numpy.dot(BV, walker_ref.phi[:,:nup])
    walker_ref.phi[:,:nup] = numpy.dot(prop.bt2[0], walker_ref.phi[:,:nup])
    numpy.testing.assert_allclose(walker.phi[:,:nup], walker_ref.phi[:,:nup], atol=1e-14)
    ovlpa = numpy.linalg.det(numpy.dot(trial.psi[:,:nup].conj().T, walker_ref.phi[:,:nup]))
    # Beta electrons
    BV = numpy.diag([prop.auxf[int(x.real),1] for x in walker.field_configs.configs[0]])
    walker_ref.phi[:,nup:] = numpy.dot(prop.bt2[1], walker_ref.phi[:,nup:])
    walker_ref.phi[:,nup:] = numpy.dot(BV, walker_ref.phi[:,nup:])
    walker_ref.phi[:,nup:] = numpy.dot(prop.bt2[1], walker_ref.phi[:,nup:])
    numpy.testing.assert_allclose(walker.phi[:,nup:], walker_ref.phi[:,nup:], atol=1e-14)
    # Test overlap
    ovlpb = numpy.linalg.det(numpy.dot(trial.psi[:,nup:].conj().T, walker_ref.phi[:,nup:]))
    assert walker.ot == pytest.approx(ovlpa*ovlpb)


@pytest.mark.unit
def test_update_greens_function():
    qmc = dotdict({'dt': 0.01, 'nstblz': 5})
    prop = HirschSpin(system, trial, qmc)
    walker = SingleDetWalker(system, trial)
    numpy.random.seed(7)
    prop.kinetic_importance_sampling(walker, system, trial)
    delta = prop.delta
    nup = system.nup
    soffset = walker.phi.shape[0] - system.nbasis
    fields = [1 if numpy.random.random() > 0.5 else 0 for i in range(system.nbasis)]
    # Reference values
    bvu = numpy.diag([prop.auxf[x][0] for x in fields])
    bvd = numpy.diag([prop.auxf[x][1] for x in fields])
    pnu = numpy.dot(bvu, walker.phi[:,:system.nup])
    pnd = numpy.dot(bvd, walker.phi[:,system.nup:])
    gu = gab(trial.psi[:,:system.nup], pnu)
    gd = gab(trial.psi[:,system.nup:], pnd)
    nup = system.nup
    ovlp = numpy.dot(trial.psi[:,:system.nup].conj().T, walker.phi[:,nup:])
    for i in range(system.nbasis):
        vtup = walker.phi[i,:nup] * delta[fields[i],0]
        vtdn = walker.phi[i,nup:] * delta[fields[i],1]
        walker.phi[i,:nup] = walker.phi[i,:nup] + vtup
        walker.phi[i,nup:] = walker.phi[i,nup:] + vtdn
        walker.update_inverse_overlap(trial, vtup, vtdn, i)
        prop.update_greens_function(walker, trial, i, nup)
        guu = gab(trial.psi[:,:system.nup], walker.phi[:,:nup])
        gdd = gab(trial.psi[:,system.nup:], walker.phi[:,nup:])
        assert guu[i,i] == pytest.approx(walker.G[0,i,i])
        assert gdd[i,i] == pytest.approx(walker.G[1,i,i])

@pytest.mark.unit
def test_hubbard_charge():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Hubbard(inputs=options)
    wfn = numpy.zeros((1,system.nbasis,system.ne), dtype=numpy.complex128)
    count = 0
    uhf = UHF(system, {'ueff': 4.0, 'initial': 'checkerboard'}, verbose=True)
    wfn[0] = uhf.psi.copy()
    trial = MultiSlater(system, (coeffs, wfn))
    trial.psi = trial.psi[0]
    walker = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    qmc = dotdict({'dt': 0.01, 'nstblz': 5})
    options = {'charge_decomposition': True}
    prop = HirschSpin(system, trial, qmc, options=options, verbose=True)
    walker = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    numpy.random.seed(7)
    nup = system.nup
    # prop.propagate_walker_constrained(walker, system, trial, 0.0)
    prop.two_body(walker, system, trial)
    walker_ref = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    # Alpha electrons
    BV = numpy.diag([prop.auxf[int(x.real),0] for x in walker.field_configs.configs[0]])
    ovlp = walker_ref.calc_overlap(trial)
    walker_ref.phi[:,:nup] = numpy.dot(BV, walker_ref.phi[:,:nup])
    walker_ref.phi[:,nup:] = numpy.dot(BV, walker_ref.phi[:,nup:])
    ovlp *= walker_ref.calc_overlap(trial)
    assert ovlp != pytest.approx(walker.ot)
    for i in walker.field_configs.configs[0]:
        ovlp *= prop.aux_fac[int(i.real)]
    assert ovlp.imag == pytest.approx(0.0, abs=1e-10)
    assert ovlp == pytest.approx(walker.ot)
