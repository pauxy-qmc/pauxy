import itertools
import cmath
from pauxy.systems.hubbard import Hubbard
from pauxy.trial_wavefunction.free_electron import FreeElectron
from pauxy.trial_wavefunction.harmonic_oscillator import HarmonicOscillator
from pauxy.estimators.ci import simple_fci_bose_fermi, simple_fci
from pauxy.estimators.hubbard import local_energy_hubbard_holstein, local_energy_hubbard
from pauxy.systems.hubbard_holstein import HubbardHolstein
from pauxy.utils.linalg import reortho

from pauxy.estimators.greens_function import gab_spin

import scipy
import numpy
import scipy.sparse.linalg
from scipy.linalg import expm
from scipy.optimize import minimize
def decode_basis(nx, ny, i):
    """Return cartesian lattice coordinates from basis index.

    Consider a 3x3 lattice then we index lattice sites like::

        (0,2) (1,2) (2,2)       6 7 8
        (0,1) (1,1) (2,1)  ->   3 4 5
        (0,0) (1,0) (2,0)       0 1 2

    i.e., i = i_x + n_x * i_y, and i_x = i%n_x, i_y = i//nx.

    Parameters
    ----------
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.
    i : int
        Basis index (same for up and down spins).
    """
    if ny == 1:
        return numpy.array([i%nx])
    else:
        return numpy.array([i%nx, i//nx])

# No Jastrow
def objective_function (x, system, psi):
    shift = x[0:system.nbasis]
    C = x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)].copy()
    C = C.reshape((system.nbasis,system.nup+system.ndown))
    Qa, detRa = reortho(C[:,:system.nup])
    Qb, detRb = reortho(C[:,system.nup:])
    C[:,:system.nup] = Qa.copy()
    C[:,system.nup:] = Qb.copy()
    x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)] = C.ravel()

    psi.psi = C.copy()
    psi.update_greens_function(system)

    ni = numpy.diag(psi.G[0]+psi.G[1])
    nia = numpy.diag(psi.G[0])
    nib = numpy.diag(psi.G[1])

    sqrttwomw = numpy.sqrt(system.m * system.w0*2.0)

    gamma = x[-1]
    alpha = gamma * numpy.sqrt(system.m * system.w0 / 2.0)

    Eph = system.w0 * numpy.sum(shift*shift)
    Eeph = (gamma * system.w0 - system.g * sqrttwomw) * numpy.sum (2.0 * shift / sqrttwomw * ni)
    Eeph += (gamma**2 * system.w0 / 2.0 - system.g * gamma * sqrttwomw) * numpy.sum(ni)

    Eee = (system.U + gamma**2 * system.w0 - 2.0 * system.g * gamma * sqrttwomw) * numpy.sum(nia * nib)

    Ekin = numpy.exp (-alpha * alpha) * numpy.sum(system.T[0] * psi.G[0] + system.T[1] * psi.G[1])

    etot = Eph + Eeph + Eee + Ekin

    return etot.real


def objective_function_rotation (x, system, psi, c0, no_orbopt = False):

    shift = x[0:system.nbasis]

    nbsf = system.nbasis
    nocca = system.nup
    noccb = system.ndown
    nvira = system.nbasis - nocca
    nvirb = system.nbasis - noccb
    
    nova = nocca*nvira
    novb = noccb*nvirb
    
    daia = x[nbsf:nbsf+nova] 
    daib = x[nbsf+nova:nbsf+nova+novb]

    daia = daia.reshape((nvira, nocca))
    daib = daib.reshape((nvirb, noccb))

    if (no_orbopt):
        daia = numpy.zeros_like(daia)
        daib = numpy.zeros_like(daib)
    
    Ua = numpy.zeros((nbsf, nbsf))
    Ub = numpy.zeros((nbsf, nbsf))

    Ua[nocca:nbsf,:nocca] = daia.copy()
    Ua[:nocca, nocca:nbsf] = -daia.T.copy()

    Ub[noccb:nbsf,:noccb] = daib.copy()
    Ub[:noccb, noccb:nbsf] = -daib.T.copy()

    C0a = c0[:nbsf*nbsf].reshape((nbsf,nbsf))
    C0b = c0[nbsf*nbsf:].reshape((nbsf,nbsf))

    Ua = expm(Ua)
    Ub = expm(Ub)

    Ca = C0a.dot(Ua)
    Cb = C0b.dot(Ub)

    psi.psi[:,:nocca] = Ca[:,:nocca].copy()
    psi.psi[:,nocca:] = Cb[:,:noccb].copy()

    psi.update_greens_function(system)

    ni = numpy.diag(psi.G[0]+psi.G[1])
    nia = numpy.diag(psi.G[0])
    nib = numpy.diag(psi.G[1])

    sqrttwomw = numpy.sqrt(system.m * system.w0*2.0)

    gamma = x[-1]
    alpha = gamma * numpy.sqrt(system.m * system.w0 / 2.0)

    Eph = system.w0 * numpy.sum(shift*shift)
    Eeph = (gamma * system.w0 - system.g * sqrttwomw) * numpy.sum (2.0 * shift / sqrttwomw * ni)
    Eeph += (gamma**2 * system.w0 / 2.0 - system.g * gamma * sqrttwomw) * numpy.sum(ni)

    Eee = (system.U + gamma**2 * system.w0 - 2.0 * system.g * gamma * sqrttwomw) * numpy.sum(nia * nib)

    Ekin = numpy.exp (-alpha * alpha) * numpy.sum(system.T[0] * psi.G[0] + system.T[1] * psi.G[1])

    etot = Eph + Eeph + Eee + Ekin

    return etot.real


def objective_function (x, system, psi):
    shift = x[0:system.nbasis]
    C = x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)].copy()
    C = C.reshape((system.nbasis,system.nup+system.ndown))
    Qa, detRa = reortho(C[:,:system.nup])
    Qb, detRb = reortho(C[:,system.nup:])
    C[:,:system.nup] = Qa.copy()
    C[:,system.nup:] = Qb.copy()
    x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)] = C.ravel()

    psi.psi = C.copy()
    psi.update_greens_function(system)

    ni = numpy.diag(psi.G[0]+psi.G[1])
    nia = numpy.diag(psi.G[0])
    nib = numpy.diag(psi.G[1])

    sqrttwomw = numpy.sqrt(system.m * system.w0*2.0)

    gamma = x[-1]
    alpha = gamma * numpy.sqrt(system.m * system.w0 / 2.0)

    Eph = system.w0 * numpy.sum(shift*shift)
    Eeph = (gamma * system.w0 - system.g * sqrttwomw) * numpy.sum (2.0 * shift / sqrttwomw * ni)
    Eeph += (gamma**2 * system.w0 / 2.0 - system.g * gamma * sqrttwomw) * numpy.sum(ni)

    Eee = (system.U + gamma**2 * system.w0 - 2.0 * system.g * gamma * sqrttwomw) * numpy.sum(nia * nib)

    Ekin = numpy.exp (-alpha * alpha) * numpy.sum(system.T[0] * psi.G[0] + system.T[1] * psi.G[1])

    etot = Eph + Eeph + Eee + Ekin

    return etot.real

def objective_function_vacuum (x, system, psi):
    shift = x[0:system.nbasis]
    C = x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)].copy()
    C = C.reshape((system.nbasis,system.nup+system.ndown))
    Qa, detRa = reortho(C[:,:system.nup])
    Qb, detRb = reortho(C[:,system.nup:])
    C[:,:system.nup] = Qa.copy()
    C[:,system.nup:] = Qb.copy()
    x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)] = C.ravel()


    Enatana = numpy.sum(system.T[0] * psi.G[0]) 
    Enbtbnb = numpy.sum(system.T[1] * psi.G[1])

    psi.psi = C.copy()
    psi.update_greens_function(system)

    ni = numpy.diag(psi.G[0]+psi.G[1])
    nia = numpy.diag(psi.G[0])
    nib = numpy.diag(psi.G[1])

    sqrttwomw = numpy.sqrt(system.m * system.w0*2.0)
    gamma = x[-1]

    Eph = 0.0
    
    Eeph = 0.0
    Eeph += (gamma**2 * system.w0 / 2.0 - system.g * gamma * sqrttwomw) * numpy.sum(ni)

    Eee = (system.U + gamma**2 * system.w0 - 2.0 * system.g * gamma * sqrttwomw) * numpy.sum(nia * nib)

    alpha = gamma * numpy.sqrt(system.m * system.w0 / 2.0)

    Ekin = numpy.exp (-alpha * alpha) * numpy.sum(system.T[0] * psi.G[0] + system.T[1] * psi.G[1])

    etot = Eph + Eeph + Eee + Ekin

    return etot.real

w0 = 0.1
l = 1

options = {
"name": "HubbardHolstein",
"nup": 1,
"ndown": 1,
"nx": 4,
"ny": 1,
"U": 4.0,
"t": 1.0,
"w0": w0,
"m": 1./w0,
"lambda": l
}

system = HubbardHolstein (options, verbose=True)
psi = FreeElectron(system, False, options, parallel=False, verbose=1)

# x = numpy.random.randn(system.nbasis + system.nbasis*(system.nup+system.ndown) + 1) * 1e-3
# rho = [numpy.diag(psi.G[0]), numpy.diag(psi.G[1])]
# shift = numpy.sqrt(system.w0*2.0 * system.m) * system.g * (rho[0]+ rho[1]) / (system.m * system.w0**2)
# nX = numpy.array([numpy.diag(shift), numpy.diag(shift)], dtype=numpy.float64)
# V = - numpy.real(system.g * cmath.sqrt(system.m * system.w0 * 2.0) * nX)
# psi.update_wfn(system, V)

# x[:system.nbasis] = shift + numpy.random.randn(shift.shape[0])
# x[system.nbasis:-1] = psi.psi.ravel() + numpy.random.randn(system.nbasis*(system.nup+system.ndown)) * 1e-1
# # x[-1] = system.g * numpy.sqrt(2.0 * system.m * system.w0) / system.w0 
# x[-1] = 0.0


# C = x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)].copy()
# C = C.reshape((system.nbasis,system.nup+system.ndown))
# Qa, detRa = reortho(C[:,:system.nup])
# Qb, detRb = reortho(C[:,system.nup:])
# C[:,:system.nup] = Qa.copy()
# C[:,system.nup:] = Qb.copy()
# x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)] = C.ravel()

# res = minimize(objective_function, x, args=(system, psi), method='BFGS', options={'disp':True, 'gtol':1e-8, 'tol':1e-10})

nbsf = system.nbasis
nocca = system.nup
noccb = system.ndown
nvira = system.nbasis - nocca
nvirb = system.nbasis - noccb

nova = nocca*nvira
novb = noccb*nvirb

x = numpy.zeros(system.nbasis + nova+novb + 1)
rho = [numpy.diag(psi.G[0]), numpy.diag(psi.G[1])]
shift = numpy.sqrt(system.w0*2.0 * system.m) * system.g * (rho[0]+ rho[1]) / (system.m * system.w0**2)
nX = numpy.array([numpy.diag(shift), numpy.diag(shift)], dtype=numpy.float64)
V = - numpy.real(system.g * cmath.sqrt(system.m * system.w0 * 2.0) * nX)
psi.update_wfn(system, V)

x[:system.nbasis] = shift + numpy.random.randn(shift.shape[0])
x[nbsf:nbsf+nova+novb] = numpy.random.randn(nova+novb)
x[-1] = 0.0

Ca = numpy.zeros((nbsf,nbsf))
Ca[:,:nocca] = psi.psi[:,:nocca]
Ca[:,nocca:] = psi.virt[:,:nvira]
Cb = numpy.zeros((nbsf,nbsf))
Cb[:,:noccb] = psi.psi[:,nocca:]
Cb[:,noccb:] = psi.virt[:,nvira:]

C0 = numpy.zeros(nbsf*nbsf*2)
C0[:nbsf*nbsf] = Ca.ravel()
C0[nbsf*nbsf:] = Cb.ravel()

res = minimize(objective_function_rotation, x, args=(system, psi, C0, False), method='BFGS', options={'disp':True, 'gtol':1e-8, 'tol':1e-10})

shift = res.x[:system.nbasis]
gamma = res.x[-1]

x0 = system.g * numpy.sqrt(2.0 * system.m * system.w0) / system.w0
print("gamma = {}, {}".format(gamma, x0))

print(shift)
print(psi.psi)
