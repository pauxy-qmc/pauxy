import numpy
from pauxy.systems.hubbard import Hubbard
from pauxy.estimators.thermal import greens_function, one_rdm_from_G, particle_number
from pauxy.trial_density_matrices.onebody import OneBody
from pauxy.trial_density_matrices.mean_field import MeanField
from pauxy.thermal_propagation.hubbard import ThermalDiscrete
from pauxy.thermal_propagation.continuous import Continuous
from pauxy.walkers.thermal import ThermalWalker
from pauxy.utils.misc import dotdict, update_stack

options = {'nx': 4, 'ny': 4, 'U': 4, 'mu': 2.0, 'nup': 5, 'ndown': 5}
system = Hubbard(options, verbose=False)
from pauxy.propagation.continuous import Continuous as ZContinuous
# prop_b = ZContinuous(system, trial, qmc, {'free_projection': True}, verbose=False)
from pauxy.trial_wavefunction.free_electron import FreeElectron
# dmat = trial.dmat
trial0 = FreeElectron(system, {}, verbose=True)
trial0.calculate_energy(system)
nup = system.nup
p = trial0.eigv_up[:,:nup]
pt = trial0.psi.copy()
import scipy.linalg
from pauxy.utils.linalg import reortho
from pauxy.estimators.mixed import local_energy
for beta in [1.0, 5.0, 10.0, 16.0, 20.0, 32, 64]:
    dt = 0.05
    nslice = int(round(beta/dt))
    trial = OneBody(system, beta, dt, verbose=False)
    B = trial.dmat[0]
    walker_a = ThermalWalker(system, trial,
                             walker_opts={'stack_size': 1,
                                          'low_rank': False,
                                          'low_rank_thresh': 1e-100},
                             verbose=False)
    walker_c = ThermalWalker(system, trial,
                             walker_opts={'stack_size': 1,
                                          'low_rank': True,
                                          'low_rank_thresh': 1e-10},
                             verbose=False)
    trial_b = OneBody(system, beta, dt, verbose=False)
    eigs, eigv = numpy.linalg.eigh(system.H1[0])
    dmat = numpy.diag(numpy.exp(-dt*(eigs-trial.mu)))
    dmat_inv = numpy.diag(numpy.exp(dt*(eigs-trial.mu)))
    trial_b.dmat = numpy.array([dmat, dmat])
    # print(dmat.diagonal())
    trial_b.dmat_inv = numpy.array([dmat_inv, dmat_inv])
    walker_b = ThermalWalker(system, trial_b,
                             walker_opts={'stack_size': 1,
                                          'low_rank_thresh': 1e-6,
                                          'low_rank': True},
                             verbose=False)
    # print(walker_a.ld)
    # gf, ld = walker_a.greens_function_qr(None, inplace=False)
    # print(numpy.linalg.norm(gf-walker_a.G))
    system.mu = trial.mu
    numpy.random.seed(7)
    qmc = dotdict({'dt': dt, 'nstblz': 10})
    from pauxy.thermal_propagation.continuous import Continuous as TContinuous
    # phib = numpy.random.random(system.nbasis*system.nup).reshape(system.nbasis, system.nup)
    # phia = numpy.random.random(system.nbasis*system.nup).reshape(system.nbasis, system.nup)
    virt = numpy.arange(system.nup)
    virt[system.nup-1] = system.nup
    phia = (trial0.eigv_up[:,:system.nup]+trial0.eigv_up[:,virt]).copy()
    phib = (trial0.eigv_dn[:,:system.nup]+trial0.eigv_dn[:,virt]).copy()
    # print(local_energy(system, trial0.G)[1])
    prop_a = TContinuous({'free_projection': True}, qmc, system, trial, verbose=False)
    logR = 0.0
    P1 = one_rdm_from_G(walker_a.stack.G)
    P2 = one_rdm_from_G(walker_b.stack.G)
    # e1b = numpy.sum(trial.eigsa*P[0].diagonal()*2)
    e, e1b_b, e2b = local_energy(system, P1)
    # print(e1b_b, 2*numpy.dot(eigs, P2[0].diagonal()))
    for ts in range(0, nslice):
        B = walker_a.stack.get(ts)
        phia = numpy.dot(B[0], phia)
        phib = numpy.dot(B[1], phib)
        # walker_a.stack.update_low_rank_non_diag(walker_a.stack.get(ts))
        # walker_b.stack.update_low_rank_non_diag(walker_b.stack.get(ts))
        if ts % 10 == 0:
            phia, R = reortho(phia)
            logR += numpy.log(R)
            phib, R = reortho(phib)
            logR += numpy.log(R)
    # print("{:10.8e}".format(numpy.linalg.norm(walker_a.G-walker_c.G)))
    eiga, eigh = numpy.linalg.eigh(walker_a.G[0])
    eigb, eigh = numpy.linalg.eigh(walker_b.G[0])
    # print(eiga-eigb)
    eig, eigh = numpy.linalg.eigh(walker_c.G[0])
    # print(numpy.prod(eig))
    # print(numpy.prod(eig))
    a1a = numpy.linalg.slogdet(walker_a.G[0])
    a1b = numpy.linalg.slogdet(walker_a.G[1])
    a3a = numpy.linalg.slogdet(walker_b.G[0])
    a3b = numpy.linalg.slogdet(walker_b.G[1])
    # det ~ exp(-beta * E_T)
    # - log(det)/beta + mu N = E_T
    a2a = numpy.linalg.slogdet(numpy.dot(pt[:,:nup].conj().T, phia))
    a2b = numpy.linalg.slogdet(numpy.dot(pt[:,nup:].conj().T, phib))
    stack_c = walker_c.stack
    stack_b = walker_b.stack
    det4 = stack_c.sgndet[0]*stack_c.sgndet[0]*(stack_c.logdet[0]+stack_c.logdet[1])
    det2 = a2a[0]*a2b[0]*(a2a[1]+a2b[1])
    det1 = a1a[0]*a1b[0]*(-(a1a[1]+a1b[1]))
    det3 = stack_b.sgndet[0]*stack_b.sgndet[0]*(stack_b.logdet[0]+stack_b.logdet[1])
    det5 = a3a[0]*a3b[0]*(-(a3a[1]+a3b[1]))
    print("{:2d} {: 8.6f} {: 8.6f} {: 8.6f} {: 8.6f} {: 8.6f} ".format(int(beta),
          -det1.real/beta+trial.mu*system.ne,
          -det4.real/beta+trial.mu*system.ne,
          -det3.real/beta+trial.mu*system.ne,
          -det5.real/beta+trial.mu*system.ne,
          -(det2+logR).real/beta+trial.mu*system.ne))
          # e1b.real,
          #-walker_a.ld/beta+trial.mu*system.ne))

