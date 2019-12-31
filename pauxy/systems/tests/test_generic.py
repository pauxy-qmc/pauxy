import os
import unittest
import numpy
import pytest
from pauxy.systems.generic import Generic
from pauxy.utils.testing import generate_hamiltonian


def test_real():
    numpy.random.seed(7)
    nmo = 17
    nelec = (4,3)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys = Generic(nelec=nelec, h1e=h1e, chol=chol, ecore=enuc)
    assert sys.nup == 4
    assert sys.ndown == 3
    assert numpy.trace(h1e) == pytest.approx(9.38462274882365)


def test_complex():
    numpy.random.seed(7)
    nmo = 17
    nelec = (5,3)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    sys = Generic(nelec=nelec, h1e=h1e, chol=chol, ecore=enuc)
    assert sys.nup == 5
    assert sys.ndown == 3
    assert sys.nbasis == 17

def test_write():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4,3)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    sys = Generic(nelec=nelec, h1e=h1e, chol=chol, ecore=enuc)
    sys.write_integrals()

def test_read():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4,3)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    from pauxy.utils.io import write_qmcpack_dense
    chol_ = chol.reshape((-1,nmo*nmo)).T.copy()
    write_qmcpack_dense(h1e, chol_, nelec, nmo,
                        enuc=enuc, filename='hamil.h5',
                        real_chol=False)
    options = {'nup': nelec[0], 'ndown': nelec[1], 'integrals': 'hamil.h5'}
    sys = Generic(inputs=options)
    schol = sys.chol_vecs
    assert numpy.linalg.norm(chol-schol) == pytest.approx(0.0)

def teardown_module():
    cwd = os.getcwd()
    files = ['hamil.h5']
    for f in files:
        try:
            os.remove(cwd+'/'+f)
        except OSError:
            pass
