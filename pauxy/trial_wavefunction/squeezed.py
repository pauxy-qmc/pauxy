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


    Enatana = numpy.sum(system.T[0] * psi.G[0]) 
    Enbtbnb = numpy.sum(system.T[1] * psi.G[1])

    psi.psi = C.copy()
    psi.update_greens_function(system)

    ni = numpy.diag(psi.G[0]+psi.G[1])
    nia = numpy.diag(psi.G[0])
    nib = numpy.diag(psi.G[1])

    gprime = numpy.sqrt(system.m * system.w0*2.0) * system.g
    gamma = gprime / system.w0

    # print("gamma = {}".format(gamma / numpy.sqrt(2.0)))


    Eph = system.w0 * numpy.sum(shift*shift)

    Eeph = - gprime**2 / (2.0 * system.w0) * numpy.sum(ni)

    Eee = (system.U - gprime**2 / system.w0) * numpy.sum(nia * nib)

    B = numpy.zeros((system.nbasis, system.nbasis))

    for i in range(0, system.nbasis):
        xy1 = decode_basis(system.nx, system.ny, i)
        for j in range(i+1, system.nbasis):
            xy2 = decode_basis(system.nx, system.ny, j)
            dij = abs(xy1-xy2)
            
            phi_i = shift[i]
            phi_i2 = shift[i] - gamma / numpy.sqrt(2.0)
            phi_j = shift[j]
            phi_j2 = shift[j] + gamma / numpy.sqrt(2.0)
            tmp_i = - (phi_i**2 + phi_i2**2) / 2.0 + phi_i **2 - gamma / numpy.sqrt(2.0) * phi_i
            tmp_j = - (phi_j**2 + phi_j2**2) / 2.0 + phi_j **2 + gamma / numpy.sqrt(2.0) * phi_j

            # print("tmp_i = {}".format(tmp_i))

            if sum(dij) == 1:
                B[i, j] = numpy.exp(tmp_i + tmp_j)

            # Take care of periodic boundary conditions
            # there should be a less stupid way of doing this.
            if system.ny == 1 and dij == [system.nx-1]:
                phase = 1.0
                B[i, j] += numpy.exp(tmp_i + tmp_j) * phase
            elif (dij==[system.nx-1, 0]).all():
                phase = 1.0
                B[i, j] += numpy.exp(tmp_i + tmp_j) * phase
            elif (dij==[0, system.ny-1]).all():
                phase = 1.0
                B[i, j] += numpy.exp(tmp_i + tmp_j) * phase

    B = B + B.conj().T
    Ekin = numpy.sum(system.T[0] * psi.G[0] * B + system.T[1] * psi.G[1] * B)

    etot = Eph + Eeph + Eee + Ekin

    print("Eph, Eeph, Eee, Ekin = {}, {}, {}, {}".format(Eph, Eeph, Eee, Ekin))

    # system.m = x[-1] # Mass updated
    # phi = HarmonicOscillator(system.m, system.w0, order=0, shift = shift)
    # Lap = phi.laplacian(shift)
    # etot, eel, eph = local_energy_hubbard_holstein(system, psi.G, shift, Lap)
    return etot.real


w0 = 0.1
l = 5.0

options = {
"name": "HubbardHolstein",
"nup": 1,
"ndown": 1,
"nx": 4,
"ny": 1,
"U": 0.0,
"w0": w0,
"m": 1./w0,
"lambda": l
}

system = HubbardHolstein (options, verbose=True)
psi = FreeElectron(system, False, options, parallel=False, verbose=1)

x = numpy.random.randn(system.nbasis + system.nbasis*(system.nup+system.ndown)+1) * 1e-3
const = options.get('shift', numpy.sqrt(system.m * system.w0*2.0) * system.g / (system.m * system.w0**2))

# x[:system.nbasis] = numpy.ones(system.nbasis) * const * 1e-2
x[:system.nbasis] = numpy.ones(system.nbasis) * const * 1e-2
x[system.nbasis:-1] = psi.psi.ravel() + numpy.random.randn(system.nbasis*(system.nup+system.ndown)) * 1e-2
x[-1] = system.m

res = minimize(objective_function, x, args=(system, psi), method='BFGS', options={'disp':True, 'gtol':1e-8, 'tol':1e-10})
ti = res.x[:system.nbasis]
mi = res.x[-1]

# print("mi = {}".format(mi))

C = res.x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)].copy()
C = C.reshape((system.nbasis,system.nup+system.ndown))
Qa, detRa = reortho(C[:,:system.nup])
Qb, detRb = reortho(C[:,system.nup:])
C[:,:system.nup] = Qa.copy()
C[:,system.nup:] = Qb.copy()
x[system.nbasis:system.nbasis + system.nbasis*(system.nup+system.ndown)] = C.ravel()
print(ti)
print(C)
