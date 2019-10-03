import numpy
import os
import pytest
from pauxy.systems.generic import Generic
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.propagation.generic import GenericContinuous
from pauxy.utils.misc import dotdict
from pauxy.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd,
        get_random_phmsd
        )
from pauxy.walkers.multi_det import MultiDetWalker

def test_phmsd():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    options = {'sparse': False}
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec, h1e=h1e, chol=chol, ecore=0, inputs=options)
    wfn = get_random_nomsd(system, ndet=3)
    trial = MultiSlater(system, wfn)
    walker = MultiDetWalker({}, system, trial)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = GenericContinuous(system, trial, qmc)
    fb = prop.construct_force_bias(system, walker, trial)
    prop.construct_VHS(system, fb)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system, ndet=3, init=True)
    trial = MultiSlater(system, wfn, init=init)
    prop = GenericContinuous(system, trial, qmc)
    walker = MultiDetWalker({}, system, trial)
    fb = prop.construct_force_bias(system, walker, trial)
    vhs = prop.construct_VHS(system, fb)
