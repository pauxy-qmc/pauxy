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
    from pauxy.utils.io import dump_qmcpack_cholesky
    dump_qmcpack_cholesky([h1e,h1e], chol, nelec, nmo, e0=enuc, filename='hamil.h5')
    options = {'nup': nelec[0], 'ndown': nelec[1], 'integrals': 'hamil.h5'}
    sys = Generic(inputs=options)
    eri = sys.chol_vecs.toarray().reshape((nmo,nmo,-1)).transpose((2,0,1))
    assert numpy.linalg.norm(chol-eri) == pytest.approx(0.0)


def teardown_module():
    cwd = os.getcwd()
    files = ['hamil.h5']
    for f in files:
        try:
            os.remove(cwd+'/'+f)
        except OSError:
            pass
