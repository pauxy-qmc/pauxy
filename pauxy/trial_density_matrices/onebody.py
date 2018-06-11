import numpy
import scipy.linalg
from pauxy.estimators.thermal import greens_function

class OneBody(object):

    def __init__(self, system, beta, dt):
        I = numpy.identity(system.H1[0].shape[0])
        self.mu = -1.0
        dmat_up = scipy.linalg.expm(-dt*(system.H1[0]-self.mu*I))
        dmat_down = scipy.linalg.expm(-dt*(system.H1[1]-self.mu*I))
        self.dmat = numpy.array([dmat_up, dmat_down])
        self.full = numpy.array([scipy.linalg.expm(-beta*(system.H1[0]-self.mu*I)),
                                 scipy.linalg.expm(-beta*(system.H1[1]-self.mu*I))])

    def nav(self, beta, chem_pot):
        I = numpy.identity(self.dmat[0].shape[0], dtype=self.dmat.dtype)
        full = numpy.einsum('ijk,kl->ijl', self.full,
                            scipy.linalg.expm(beta*(chem_pot-self.mu)*I))
        G = greens_function(full)
        nav = G[0].trace() + G[1].trace()
        return nav

    def find_chemical_potential(self, system, beta, step=0.1, verbose=False):
        I = numpy.identity(self.dmat[0].shape, dtype=self.dmat.dtype)
        mu_new = system.mu
        G = pauxy.estimators.greens_function(self.full)
        nav = G[0].trace() + G[1].trace()
        delta = nav - system.nav
        # if delta > 0:
            # while (delta_new < 0):
                # self.full = self.full * numpy.expm(beta*(mu_new-mu)*I)
                # G = pauxy.estimators.greens_function(self.full)
                # nav = G[0].trace() + G[1].trace()
                # delta_new = nav - system.nav
        # while (abs(delta) > 1e-8):
            # self.full = self.full * numpy.expm(beta*(mu_new-mu)*I)
            # G = pauxy.estimators.greens_function(self.full)
            # nav = G[0].trace() + G[1].trace()
            # delta_new = nav - system.nav
            # if (delta_new*delta < 0):
                # mu = mu - 0.5 * step
            # else:
                # mu = mu + step
            # if verbose:
                # print (delta
