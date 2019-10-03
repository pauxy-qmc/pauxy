import numpy
from pauxy.utils.misc import dotdict
from pauxy.utils.linalg import modified_cholesky
from pauxy.systems.generic import Generic
from pauxy.trial_wavefunction.multi_slater import MultiSlater

def get_random_generic(nmo, nelec):
    h1e = numpy.random.random((nmo,nmo))
    h1e = h1e + h1e.T
    eri = numpy.random.normal(scale=0.01, size=(nmo,nmo,nmo,nmo))
    # Restore symmetry to the integrals.
    eri = eri + eri.transpose((1,0,2,3))
    eri = eri + eri.transpose((0,1,3,2))
    eri = eri + eri.transpose((2,3,0,1))
    eri = eri.reshape((nmo*nmo,nmo*nmo))
    # Make positive semi-definite.
    eri = numpy.dot(eri,eri.T)
    chol = modified_cholesky(eri, 1e-5, verbose=False)
    chol = chol.reshape((-1,nmo,nmo))
    options = {'sparse': False}
    system = Generic(nelec=nelec, h1e=h1e, chol=chol,
                     ecore=0, inputs=options)
    return system

def get_random_nomsd(system, ndet=10):
    a = numpy.random.rand(ndet*system.nbasis*(system.nup+system.ndown))
    b = numpy.random.rand(ndet*system.nbasis*(system.nup+system.ndown))
    wfn = (a + 1j*b).reshape((ndet,system.nbasis,system.nup+system.ndown))
    coeffs = numpy.random.rand(ndet)+1j*numpy.random.rand(ndet)
    trial = MultiSlater(system, (coeffs, wfn))
    return trial

def get_random_phmsd(system, ndet=10, init=None):
    orbs = numpy.arange(system.nbasis)
    oa = [c for c in itertools.combinations(orbs, system.nup)]
    ob = [c for c in itertools.combinations(orbs, system.ndown)]
    oa, ob = zip(*itertools.product(oa,ob))
    oa = oa[:ndet]
    ob = ob[:ndet]
    coeffs = numpy.random.rand(ndet)+1j*numpy.random.rand(ndet)
    wfn = (coeffs,oa,ob)
    if init is not None:
        a = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
        b = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
        init = (a + 1j*b).reshape((system.nbasis,system.nup+system.ndown))
    trial = MultiSlater(system, wfn, init=init)
    return trial

def get_random_wavefunction(nelec, nbasis):
    na = nelec[0]
    nb = nelec[1]
    a = numpy.random.rand(nbasis*(na+nb))
    b = numpy.random.rand(nbasis*(na+nb))
    init = (a + 1j*b).reshape((nbasis,na+nb))
    return init
