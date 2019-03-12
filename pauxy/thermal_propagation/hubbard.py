import cmath
import copy
import numpy
import math
import scipy.linalg

class ThermalDiscrete(object):

    def __init__(self, options, qmc, system, trial, verbose=False):

        if verbose:
            print ("# Parsing discrete propagator input options.")
        self.free_projection = options.get('free_projection', False)
        self.nstblz = qmc.nstblz
        self.hs_type = 'discrete'
        self.gamma = numpy.arccosh(numpy.exp(0.5*qmc.dt*system.U))
        self.auxf = numpy.array([[numpy.exp(self.gamma), numpy.exp(-self.gamma)],
                                [numpy.exp(-self.gamma), numpy.exp(self.gamma)]])
        self.auxf = self.auxf * numpy.exp(-0.5*qmc.dt*system.U)
        self.delta = self.auxf - 1
        dt = qmc.dt
        dmat_up = scipy.linalg.expm(-dt*(system.H1[0]))
        dmat_down = scipy.linalg.expm(-dt*(system.H1[1]))
        dmat = numpy.array([dmat_up,dmat_down])
        self.construct_one_body_propagator(system, dt)
        self.BT_BP = None
        self.BT = trial.dmat
        self.BT_inv = trial.dmat_inv
        self.BV = numpy.zeros((2,trial.dmat.shape[-1]), dtype=trial.dmat.dtype)
        if self.free_projection:
            self.propagate_walker = self.propagate_walker_free
        else:
            self.propagate_walker = self.propagate_walker_constrained

    def construct_one_body_propagator(self, system, dt):
        """Construct the one-body propagator Exp(-dt/2 H0)
        Parameters
        ----------
        system :
            system class
        dt : float
            time-step
        Returns
        -------
        self.BH1 : numpy array
            Exp(-dt/2 H0)
        """
        H1 = system.H1
        I = numpy.identity(H1[0].shape[0], dtype=H1.dtype)
        # No spin dependence for the moment.
        self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]+0.5*dt*system.mu*I),
                                scipy.linalg.expm(-0.5*dt*H1[1]+0.5*dt*system.mu*I)])

    def update_greens_function_simple(self, walker, time_slice):
        walker.construct_greens_function_stable(time_slice)

    def update_greens_function(self, walker, i, xi):
        for spin in [0,1]:
            g = walker.G[spin,:,i]
            gbar = -walker.G[spin,i,:]
            gbar[i] += 1
            denom = 1 + (1-g[i]) * self.delta[xi,spin]
            walker.G[spin] = (
                walker.G[spin] - self.delta[xi,spin]*numpy.einsum('i,j->ij', g, gbar) / denom
            )

    def propagate_greens_function(self, walker):
        if walker.stack.time_slice < walker.stack.ntime_slices:
            walker.G[0] = self.BT[0].dot(walker.G[0]).dot(self.BT_inv[0])
            walker.G[1] = self.BT[1].dot(walker.G[1]).dot(self.BT_inv[1])

    def calculate_overlap_ratio(self, walker, i):
        R1_up = 1 + (1-walker.G[0,i,i])*self.delta[0,0]
        R1_dn = 1 + (1-walker.G[1,i,i])*self.delta[0,1]
        R2_up = 1 + (1-walker.G[0,i,i])*self.delta[1,0]
        R2_dn = 1 + (1-walker.G[1,i,i])*self.delta[1,1]
        return 0.5 * numpy.array([R1_up*R1_dn, R2_up*R2_dn])

    def propagate_walker_constrained(self, system, walker, time_slice):
        for i in range(0, system.nbasis):
            probs = self.calculate_overlap_ratio(walker, i)
            phaseless_ratio = numpy.maximum(probs.real, [0,0])
            norm = sum(phaseless_ratio)
            r = numpy.random.random()
            if norm > 0:
                walker.weight = walker.weight * norm
                if walker.weight > walker.total_weight * 0.10:
                    walker.weight = walker.total_weight * 0.10
                if r < phaseless_ratio[0] / norm:
                    xi = 0
                else:
                    xi = 1
                self.update_greens_function(walker, i, xi)
                self.BV[0,i] = self.auxf[xi, 0]
                self.BV[1,i] = self.auxf[xi, 1]
            else:
                walker.weight = 0
        B = numpy.einsum('ki,kij->kij', self.BV, self.BH1)
        B = numpy.einsum('kin,knj->kij', self.BH1, B)
        walker.stack.update(B)
        # Need to recompute Green's function from scratch before we propagate it
        # to the next time slice due to stack structure.
        if walker.stack.time_slice % self.nstblz == 0:
            walker.greens_function(None, walker.stack.time_slice-1)
        self.propagate_greens_function(walker)

    def propagate_walker_free(self, system, walker, time_slice):
        for i in range(0, system.nbasis):
            probs = self.calculate_overlap_ratio(walker, i)
            norm = sum(numpy.abs(probs))
            sgns = numpy.sign(probs) / numpy.sign(sum(probs))
            r = numpy.random.random()
            if norm > 0:
                if r < abs(probs[0]) / norm:
                    xi = 0
                    walker.weight *= norm
                    walker.phase *= sgns[0]
                else:
                    xi = 1
                    walker.weight = walker.weight * norm
                    walker.phase *= sgns[1]
                self.update_greens_function(walker, i, xi)
                self.BV[0,i] = self.auxf[xi, 0]
                self.BV[1,i] = self.auxf[xi, 1]
            else:
                walker.weight = 0
        B = numpy.einsum('ki,kij->kij', self.BV, self.BH1)
        B = numpy.einsum('kin,knj->kij', self.BH1, B)
        walker.stack.update(B)
        # Need to recompute Green's function from scratch before we propagate it
        # to the next time slice due to stack structure.
        if walker.stack.time_slice % self.nstblz == 0:
            walker.greens_function(None, walker.stack.time_slice-1)
        self.propagate_greens_function(walker)
