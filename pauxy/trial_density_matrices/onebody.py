import numpy
import scipy.linalg
from pauxy.estimators.thermal import greens_function, particle_number, one_rdm

class OneBody(object):

    def __init__(self, trial, system, beta, dt, verbose=False):
        self.name = 'thermal'
        self.ntime_slices = int(beta/dt)
        dmat_up = scipy.linalg.expm(-dt*(system.H1[0]))
        dmat_down = scipy.linalg.expm(-dt*(system.H1[1]))
        self.dmat = numpy.array([dmat_up, dmat_down])
        self.I = numpy.identity(self.dmat[0].shape[0], dtype=self.dmat.dtype)
        # Ignore factor of 1/L
        self.nav = system.nup + system.ndown
        self.max_it = trial.get('max_it', 1000)
        self.deps = trial.get('threshold', 1e-6)
        self.mu = trial.get('mu', None)
        if self.mu is None:
            self.mu = self.find_chemical_potential(system, beta, verbose)
        self.dmat = self.compute_rho(self.dmat, self.mu, dt)
        self.dmat_inv = numpy.array([scipy.linalg.inv(self.dmat[0]),
                                     scipy.linalg.inv(self.dmat[1])])
        self.G = numpy.array([greens_function(self.dmat[0]), greens_function(self.dmat[1])])
        self.error = False

    def find_chemical_potential(self, system, beta, verbose=False):
        rho = numpy.array([scipy.linalg.expm(-beta*(system.H1[0])),
                           scipy.linalg.expm(-beta*(system.H1[1]))])
        # Todo: some sort of generic starting point independent of
        # system/temperature
        dmu1 = dmu2 = 1
        mu1 = -1
        mu2 = 1
        while (numpy.sign(dmu1)*numpy.sign(dmu2) > 0):
            rho1 = self.compute_rho(rho, mu1, beta)
            dmat = one_rdm(rho1)
            dmu1 = self.delta(dmat)
            rho2 = self.compute_rho(rho, mu2, beta)
            dmat = one_rdm(rho2)
            dmu2 = self.delta(dmat)
            if (numpy.sign(dmu1)*numpy.sign(dmu2) < 0):
                if verbose:
                    print ("# Chemical potential lies within range of [%f,%f]"%(mu1,
                                                                                mu2))
                    print ("# delta_mu1 = %f, delta_mu2 = %f"%(dmu1, dmu2))
                break
            else:
                mu1 -= 2
                mu2 += 2
                if verbose:
                    print ("# Increasing chemical potential search to [%f,%f]"%(mu1, mu2))
        found_mu = False
        for i in range(0, self.max_it):
            mu = 0.5 * (mu1 + mu2)
            rho_mu = self.compute_rho(rho, mu, beta)
            dmat = one_rdm(rho_mu)
            dmu = self.delta(dmat)
            if verbose:
                print ("# %d mu = %.8f dmu = %13.8e nav = %f" % (i, mu, dmu,
                                                           particle_number(dmat)))
            if (abs(dmu) < self.deps):
                found_mu = True
                break
            else:
                if (dmu*dmu1 > 0):
                    mu1 = mu
                elif (dmu*dmu2 > 0):
                    mu2 = mu
        if found_mu:
            if verbose:
                print ("# Chemical potential found to be: %.8f" % mu)
            return mu
        else:
            print ("# Error chemical potential not found")
            return None

    def delta(self, dm):
        return particle_number(dm) - self.nav

    def compute_rho(self, rho, mu, beta):
        return numpy.einsum('ijk,kl->ijl', rho,
                            scipy.linalg.expm(beta*mu*self.I))
