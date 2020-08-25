import cmath
import numpy
import scipy.linalg
from pauxy.walkers.stack import PropagatorStack

class DiscreteHubbard(object):
    """Base class for determinant updates.
    """
    def __init__(self, system, dt, nslice,
                 stack_size=1,
                 dynamic_force=False,
                 low_rank=False,
                 charge_decomp=False,
                 single_site=False):

        self.dynamic_force = dynamic_force
        self.low_rank = low_rank
        self.charge_decomp = charge_decomp
        self.single_site = single_site
        self.set_aux_fields(system, dt)
        self.nsites = system.nbasis
        self.set_one_body(system, dt)
        self.stack = PropagatorStack(stack_size, nslice, system.nbasis,
                                     numpy.complex128, lowrank=low_rank)
        self.fields = numpy.random.randint(0, 2, nslice*system.nbasis).reshape(nslice, system.nbasis)
        self.set_initial_stack()
        if self.single_site:
            self.update = self.update_single_site
        else:
            self.update = None

    def calculate_determinant_ratio(self, G, ibasis, xi):
        ratio_up = 1.0 + (1.0-G[0,ibasis,ibasis])*self.delta[xi,0]
        ratio_dn = 1.0 + (1.0-G[1,ibasis,ibasis])*self.delta[xi,1]
        ratio = ratio_up * ratio_dn * self.aux_wfac[xi]
        ratio, phi = cmath.polar(ratio)
        return ratio, phi

    def set_initial_stack(self):
        for f in self.fields:
            B = self.construct_bmatrix(f)
            self.stack.update(B)

    def construct_bmatrix(self, field):
        BVa = numpy.diag(numpy.array([self.auxf[xi,0] for xi in field]))
        BVb = numpy.diag(numpy.array([self.auxf[xi,1] for xi in field]))
        Ba = numpy.dot(BVa, self.BH1[0])
        Bb = numpy.dot(BVb, self.BH1[1])
        return numpy.array([Ba,Bb])

    def update_single_site(self, G, islice):
        config_phase = 1 + 0j
        for ibasis in range(self.nsites):
            # Propose spin flip
            field = self.fields[islice,ibasis]
            if field == 0:
                xi = 1
            else:
                xi = 0
            ratio, phi = self.calculate_determinant_ratio(G, ibasis, xi)
            P = ratio / (1.0 + abs(ratio))
            r = numpy.random.random()
            if r < P:
                self.fields[islice,ibasis] = xi
                self.update_greens_function(G, ibasis, xi)
                config_phase += phi
        B = self.construct_bmatrix(self.fields[islice])
        self.stack.update(B)
        return numpy.exp(1j*phi)

    def propagate_greens_function(self, G, ifield):
        B = self.construct_bmatrix(self.fields[ifield])
        return numpy.array([numpy.dot(B[0], numpy.dot(G[0], numpy.linalg.inv(B[0]))),
                            numpy.dot(B[1], numpy.dot(G[1], numpy.linalg.inv(B[1])))])

    def update_greens_function(self, G, i, xi):
        for spin in [0,1]:
            g = G[spin,:,i]
            gbar = -G[spin,i,:]
            gbar[i] += 1
            denom = 1 + (1-g[i]) * self.delta[xi,spin]
            G[spin] = (
                G[spin] - self.delta[xi,spin]*numpy.einsum('i,j->ij', g, gbar) / denom
            )

    def set_aux_fields(self,system, dt):
        if self.charge_decomp:
            gamma = numpy.arccosh(numpy.exp(-0.5*dt*system.U+0j))
            self.auxf = numpy.array([[numpy.exp(gamma), numpy.exp(gamma)],
                                     [numpy.exp(-gamma), numpy.exp(-gamma)]])
            # if current field is +1 want to flip to -1, then delta[1] gives flip factor
            self.delta = numpy.array([[numpy.exp(2*gamma), numpy.exp(2*gamma)],
                                 [numpy.exp(-2*gamma), numpy.exp(-2*gamma)]])
            self.delta -= 1.0
            # e^{-gamma x}
            self.aux_wfac = numpy.exp(0.5*dt*system.U) * numpy.array([numpy.exp(-2*gamma),
                                                                      numpy.exp(2*gamma)])
            self.aux_wfac_ = numpy.exp(0.5*dt*system.U) * numpy.array([numpy.exp(-gamma),
                                                                       numpy.exp(gamma)])
            self.gamma = gamma
        else:
            gamma = numpy.arccosh(numpy.exp(0.5*dt*system.U))
            self.auxf = numpy.array([[numpy.exp(gamma), numpy.exp(-gamma)],
                                [numpy.exp(-gamma), numpy.exp(gamma)]])
            self.delta = numpy.array([[numpy.exp(2*gamma), numpy.exp(-2*gamma)],
                                 [numpy.exp(-2*gamma), numpy.exp(2*gamma)]])
            self.delta -= 1.0
            self.aux_wfac = numpy.array([1.0, 1.0])
            self.gamma = gamma

        self.auxf = self.auxf * numpy.exp(-0.5*dt*system.U)

    def set_one_body(self, system, dt):
        I = numpy.eye(system.nbasis, dtype=numpy.complex128)
        self.BH1 = numpy.array([scipy.linalg.expm(-dt*(system.H1[0]-system.mu*I)),
                                scipy.linalg.expm(-dt*(system.H1[1]-system.mu*I))],
                                dtype=numpy.complex128)
        self.BH1inv = numpy.array([scipy.linalg.expm(dt*(system.H1[0]-system.mu*I)),
                                   scipy.linalg.expm(dt*(system.H1[1]-system.mu*I))],
                                   dtype=numpy.complex128)
