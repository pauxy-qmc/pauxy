import numpy
import scipy.linalg
from pauxy.estimators.thermal import greens_function, particle_number

class OneBody(object):

    def __init__(self, system, beta, dt, verbose=False):
        dmat_up = scipy.linalg.expm(-dt*(system.H1[0]))
        dmat_down = scipy.linalg.expm(-dt*(system.H1[1]))
        self.dmat = numpy.array([dmat_up, dmat_down])
        self.I = numpy.identity(self.dmat[0].shape[0], dtype=self.dmat.dtype)
        # Ignore factor of 1/L
        self.nav = system.nup + system.ndown
        self.max_it = 100
        self.deps = 1e-8
        self.mu = self.find_chemical_potential(system, beta, verbose)
        self.dmat = self.compute_rho(self.dmat, self.mu, dt)

    def find_chemical_potential(self, system, beta, verbose=False):
        # Todo: some sort of generic starting point independent of system
        rho = numpy.array([scipy.linalg.expm(-beta*(system.H1[0])),
                           scipy.linalg.expm(-beta*(system.H1[1]))])
        mu1 = -10
        mu2 = 10
        rho1 = self.compute_rho(rho, mu1, beta)
        G = greens_function(rho1)
        dmu1 = self.delta(G)
        rho2 = self.compute_rho(rho, mu2, beta)
        G = greens_function(rho2)
        dmu2 = self.delta(G)
        if (numpy.sign(dmu1)*numpy.sign(dmu2) < 0 and verbose):
            print ("# Chemical potential lies within range of [%f,%f]"%(mu1, mu2))
            print ("# delta_mu1 = %f, delta_mu2 = %f"%(dmu1, dmu2))
        found_mu = False
        for i in range(0, self.max_it):
            mu = 0.5 * (mu1 + mu2)
            rho_mu = self.compute_rho(rho, mu, beta)
            G = greens_function(rho_mu)
            dmu = self.delta(G)
            if verbose:
                print ("# %d mu = %f dmu = %f nav = %f" % (i, mu, dmu, particle_number(G)))
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
                print ("# Chemical potential found to be: %f" % mu)
            return mu
        else:
            print ("# Error chemical potential not found")
            return None

    def delta(self, G):
        return particle_number(G) - self.nav

    def compute_rho(self, rho, mu, beta):
        return numpy.einsum('ijk,kl->ijl', rho, scipy.linalg.expm(beta*mu*self.I))
