import math
import numpy
import scipy.linalg
import sys
from pauxy.estimators.thermal import (
        greens_function, particle_number, one_rdm, one_rdm_from_G,
        one_rdm_stable
        )
from pauxy.utils.io import (
        format_fixed_width_strings, format_fixed_width_floats
        )
from pauxy.utils.misc import update_stack

class OneBody(object):

    def __init__(self, options, system, beta, dt, H1=None, verbose=False):
        self.name = 'thermal'
        if H1 is None:
            try:
                self.H1 = system.H1
            except AttributeError:
                self.H1 = system.h1e
        else:
            self.H1 = H1

        if verbose:
            print("# beta in OneBody: {}".format(beta))
            print("# dt in OneBody: {}".format(dt))

        dmat_up = scipy.linalg.expm(-dt*(self.H1[0]))
        dmat_down = scipy.linalg.expm(-dt*(self.H1[1]))
        self.dmat = numpy.array([dmat_up, dmat_down])
        cond = numpy.linalg.cond(self.dmat[0])
        if verbose:
            print("# condition number of BT: {: 10e}".format(cond))

        self.nav = system.nup + system.ndown
        self.max_it = options.get('max_it', 1000)
        self.deps = options.get('threshold', 1e-6)
        self.mu = options.get('mu', None)
        if verbose:
            print("# Estimating stack size from BT.")
        eigs, ev = scipy.linalg.eigh(self.dmat[0])
        emax = numpy.max(eigs)
        emin = numpy.min(eigs)
        self.num_slices = int(beta/dt)
        self.cond = numpy.linalg.cond(self.dmat[0])
        # We will end up multiplying many BTs together. Can roughly determine
        # safe stack size from condition number of BT as the condition number of
        # the product will scale roughly as cond(BT)^(number of products).
        # We can determine a conservative stack size by requiring that the
        # condition number of the product does not exceed 1e3.
        self.stack_size = min(self.num_slices, int(3.0/numpy.log10(self.cond)))
        if verbose:
            print("# Initial stack size: {}".format(self.stack_size))
        # adjust stack size
        self.stack_size = update_stack(self.stack_size,self.num_slices, verbose)
        self.num_bins = int(beta/(self.stack_size*dt))

        if verbose:
            print("# Number of stacks: {}".format(self.num_bins))

        if self.mu is None:
            dtau = self.stack_size * dt
            rho = numpy.array([scipy.linalg.expm(-dtau*(self.H1[0])),
                               scipy.linalg.expm(-dtau*(self.H1[1]))])
            self.mu = self.find_chemical_potential(system, rho,
                                                   dtau, verbose)

        if verbose:
            print("# Chemical potential: {: .10e}".format(self.mu))

        if system.mu is None:
            system.mu = self.mu

        self.dmat = self.compute_rho(self.dmat, self.mu, dt)
        self.dmat_inv = numpy.array([scipy.linalg.inv(self.dmat[0], check_finite=False),
                                     scipy.linalg.inv(self.dmat[1], check_finite=False)])

        self.G = numpy.array([greens_function(self.dmat[0]), greens_function(self.dmat[1])])
        self.error = False

    def find_chemical_potential(self, system, rho, beta, verbose=False):
        # Todo: some sort of generic starting point independent of
        # system/temperature
        dmu1 = dmu2 = 1
        mu1 = -1
        mu2 = 1
        while numpy.sign(dmu1)*numpy.sign(dmu2) > 0:
            rho1 = self.compute_rho(rho, mu1, beta)
            dmat = one_rdm_stable(rho1, self.num_bins)
            dmu1 = self.delta(dmat)
            rho2 = self.compute_rho(rho, mu2, beta)
            dmat = one_rdm_stable(rho2, self.num_bins)
            dmu2 = self.delta(dmat)
            if numpy.sign(dmu1)*numpy.sign(dmu2) < 0:
                if verbose:
                    print ("# Chemical potential lies within range of [%f,%f]"%(mu1,
                                                                                mu2))
                    print ("# delta_mu1 = %f, delta_mu2 = %f"%(dmu1.real,
                                                               dmu2.real))
                break
            else:
                mu1 -= 2
                mu2 += 2
                if verbose:
                    print ("# Increasing chemical potential search to [%f,%f]"%(mu1, mu2))
        found_mu = False
        if verbose:
            print(format_fixed_width_strings(['iteration', 'mu', 'Dmu', '<N>']))
        for i in range(0, self.max_it):
            mu = 0.5 * (mu1 + mu2)
            rho_mu = self.compute_rho(rho, mu, beta)
            dmat = one_rdm_stable(rho_mu, self.num_bins)
            dmu = self.delta(dmat).real
            if verbose:
                out = [i, mu, dmu, particle_number(dmat).real]
                print(format_fixed_width_floats(out))
            if abs(dmu) < self.deps:
                found_mu = True
                break
            else:
                if dmu*dmu1 > 0:
                    mu1 = mu
                elif dmu*dmu2 > 0:
                    mu2 = mu
        if found_mu:
            return mu
        else:
            print("# Error chemical potential not found")
            return None

    def delta(self, dm):
        return particle_number(dm) - self.nav

    def compute_rho(self, rho, mu, beta):
        return numpy.einsum('ijk,k->ijk', rho,
                            numpy.exp(beta*mu*numpy.ones(rho.shape[-1])))
