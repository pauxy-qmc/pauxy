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
from scipy.optimize import minimize

def expJ (alpha, shift):
    nbsf = shift.size
    assert(alpha.size == nbsf*nbsf)
    alpha_mat = alpha.reshape((nbsf, nbsf)).copy()

    beta = numpy.einsum("ij,i->j", alpha_mat, shift)

    expJ = numpy.diag(numpy.exp(beta))

    return expJ

def expJ2 (alpha, shift):
    nbsf = shift.size
    assert(alpha.size == nbsf*nbsf)
    alpha_mat = alpha.reshape((nbsf, nbsf)).copy()

    beta = numpy.einsum("ij,i->j", alpha_mat, shift*shift)

    expJ = numpy.diag(numpy.exp(beta))

    return expJ

###
# No Jastrow
def objective_function (x, system, psi):
    shift = x[0:system.nbasis]
    C = x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)].copy()
    C = C.reshape((system.nbasis,system.nup+system.ndown))
    Qa, detRa = reortho(C[:,:system.nup])
    Qb, detRb = reortho(C[:,system.nup:])
    C[:,:system.nup] = Qa.copy()
    C[:,system.nup:] = Qb.copy()
    # print(C)
    x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)] = C.ravel()

    # system.m = x[-1] # Mass updated
    phi = HarmonicOscillator(system.m, system.w0, order=0, shift = shift)
    # phi = HarmonicOscillator(numpy.abs(x[-1]), system.w0, order=0, shift = shift)
    
    psi.psi = C.copy()
    psi.update_greens_function(system)

    # print(psi.G)

    Lap = phi.laplacian(shift)
    etot, eel, eph = local_energy_hubbard_holstein(system, psi.G, shift, Lap)
    
    # print("etot, eel, eph = {}, {}, {}".format(etot, eel, eph))

    return etot.real


w0 = 0.1
l = 10.0
         # Current function value: -16.500000

options = {
"name": "HubbardHolstein",
"nup": 1,
"ndown": 1,
"nx": 2,
"ny": 1,
"t": 1.0,
"U": 4.0,
"w0": w0,
"m": 1./w0,
"lambda": l
}

system = HubbardHolstein (options, verbose=True)
psi = FreeElectron(system, False, options, parallel=False, verbose=1)

# const = options.get('shift', numpy.sqrt(system.m * system.w0*2.0) * system.g / (system.m * system.w0**2))
# shift = numpy.ones(system.nbasis) * const
# phi = HarmonicOscillator(system.m, system.w0, order=0, shift = shift)
# Lap = phi.laplacian(shift)
# energy = local_energy_hubbard_holstein(system, psi.G, shift, Lap)
# print("initial energy = {}".format(energy))
# print("initial energy = {}".format(objective_function(x, system, psi, phi)))

# Eph = (system.m * system.w0**2) * 0.5 * numpy.sum(shift*shift) - system.g * numpy.sqrt(2.0 * system.m * system.w0)\
# * numpy.sum(shift * (numpy.diag(psi.G[0]) + numpy.diag(psi.G[1])))

# for i in range(10):
#     rho = [numpy.diag(psi.G[0]), numpy.diag(psi.G[1])]
#     shift = numpy.sqrt(system.w0*2.0 * system.m) * system.g * (rho[0]+ rho[1]) / (system.m * system.w0**2)
#     phi = HarmonicOscillator(system.m, system.w0, order=0, shift = shift)
    
#     nX = numpy.array([numpy.diag(shift), numpy.diag(shift)], dtype=numpy.float64)
#     V = - numpy.real(system.g * cmath.sqrt(system.m * system.w0 * 2.0) * nX)
#     psi.update_wfn(system, V)

#     Lap = phi.laplacian(shift)

#     energy = local_energy_hubbard_holstein(system, psi.G, shift, Lap)

#     print("#{} {} {} {} {}".format(i, energy, shift, rho[0], rho[1]))

# No Jastrow
x = numpy.random.randn(system.nbasis + system.nbasis*(system.nup+system.ndown)) * 1e-3
rho = [numpy.diag(psi.G[0]), numpy.diag(psi.G[1])]
shift = numpy.sqrt(system.w0*2.0 * system.m) * system.g * (rho[0]+ rho[1]) / (system.m * system.w0**2)
nX = numpy.array([numpy.diag(shift), numpy.diag(shift)], dtype=numpy.float64)
V = - numpy.real(system.g * cmath.sqrt(system.m * system.w0 * 2.0) * nX)
psi.update_wfn(system, V)

x[:system.nbasis] = shift + numpy.random.randn(shift.shape[0])
x[system.nbasis:] = psi.psi.ravel() + numpy.random.randn(system.nbasis*(system.nup+system.ndown))
# x[:system.nbasis] = shift 
# x[system.nbasis:] = psi.psi.ravel()
res = minimize(objective_function, x, args=(system, psi), method='BFGS', options={'disp':True, 'gtol':1e-10})
xshift = res.x[:system.nbasis]

C = res.x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)].copy()
C = C.reshape((system.nbasis,system.nup+system.ndown))
Qa, detRa = reortho(C[:,:system.nup])
Qb, detRb = reortho(C[:,system.nup:])
C[:,:system.nup] = Qa.copy()
C[:,system.nup:] = Qb.copy()

print(xshift)
print(C)
# print(res.x[-1], system.m)

