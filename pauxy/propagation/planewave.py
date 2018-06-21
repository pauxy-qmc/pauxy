import cmath
import math
import numpy
import scipy.sparse.linalg
import time
from pauxy.propagation.operations import kinetic_real
from pauxy.utils.linalg import exponentiate_matrix
from pauxy.walkers.single_det import SingleDetWalker

class PlaneWave(object):
    def __init__(self, options, qmc, system, trial, verbose=False):
        if verbose:
            print ("# Parsing plane wave propagator input options.")
        # Input options
        self.hs_type = 'plane_wave'
        self.free_projection = options.get('free_projection', False)
        self.exp_nmax = options.get('expansion_order', 4)
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j*self.sqrt_dt
        self.num_vplus = system.nfields // 2
        
        print("# Number of fields = %i"%system.nfields)

        self.vbias = numpy.zeros(system.nfields, dtype=numpy.complex128)

        # Constant core contribution modified by mean field shift.
        mf_core = system.ecore
        self.construct_one_body_propagator(system, qmc.dt)
        self.mf_const_fac = 1

        # todo : ?
        self.BT_BP = self.BH1
        self.nstblz = qmc.nstblz

        # Temporary array for matrix exponentiation.
        self.Temp = numpy.zeros(trial.psi[:,:system.nup].shape,
                                dtype=trial.psi.dtype)

        self.ebound = (2.0/self.dt)**0.5
        self.mean_local_energy = 0

        if self.free_projection:
            print("# Using free projection")
            self.propagate_walker = self.propagate_walker_free
        else:
            print("# Using phaseless approximation")
            self.propagate_walker = self.propagate_walker_phaseless

        if verbose:
            print ("# Finished setting up propagator.")

    def two_body_potentials(self, system, q):
        rho_q = system.density_operator(q)
        qscaled = system.kfac * q
        # twopiovol = 2.0*math.pi / system.vol

        # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol
        piovol = math.pi / (system.vol)
        factor = (piovol/numpy.dot(qscaled,qscaled))**0.5

        # JOONHO: include a factor of 1j
        iA = 1j * factor * (rho_q + rho_q.conj().T) 
        iB = - factor * (rho_q - rho_q.conj().T) 
        return (iA, iB)

    def construct_one_body_propagator(self, system, dt):
        H1 = system.h1e_mod
        # No spin dependence for the moment.
        self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]),
                                scipy.linalg.expm(-0.5*dt*H1[1])])

    def apply_exponential(self, phi, VHS, debug=False):
        # JOONHO: exact exponential
        copy = numpy.copy(phi)
        # phi = scipy.sparse.linalg.expm(VHS).dot(copy)
        phi = scipy.linalg.expm(VHS).dot(copy)
        # if debug:
        #     copy = numpy.copy(phi)
        #     c2 = scipy.linalg.expm(VHS).dot(copy)
        # numpy.copyto(self.Temp, phi)
        # for n in range(1, self.exp_nmax+1):
        #     self.Temp = VHS.dot(self.Temp) / n
        #     phi += self.Temp
        # if debug:
        #     print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))
        # print("apply_exponential (seconds) %10.5f"%(end - start))
        return phi

    def construct_force_bias(self, system, G):
        for (i, qi) in enumerate(system.qvecs):
            (iA, iB) = self.two_body_potentials(system, qi)
            # Deal with spin more gracefully
            self.vbias[i] = numpy.einsum('ij,kij->', iA, G)
            self.vbias[i+self.num_vplus] = numpy.einsum('ij,kij->', iB, G)
        return - self.sqrt_dt * self.vbias

    def construct_VHS(self, system, xshifted):
        VHS = numpy.zeros(shape=(system.nbasis,system.nbasis),
                          dtype=numpy.complex128)
        # VHS = scipy.sparse.csc_matrix((system.nbasis,system.nbasis), dtype=numpy.complex128)
        for (i, qi) in enumerate(system.qvecs):
            (iA, iB) = self.two_body_potentials(system, qi)
            VHS = VHS + xshifted[i] * iA 
            VHS = VHS + xshifted[i+self.num_vplus] * iB 
        return  VHS * self.sqrt_dt

    def two_body(self, walker, system, trial, fb = True):
        # Normally distrubted auxiliary fields.
        xi = numpy.random.normal(0.0, 1.0, system.nfields)

        # Optimal force bias.
        xbar = numpy.zeros(system.nfields)
        if (fb):
            xbar = self.construct_force_bias(system, walker.G)
        
        xshifted = xi - xbar

        # Constant factor arising from force bias and mean field shift
        # Mean field shift is zero for UEG in HF basis
        cxf = 1.0
        # Constant factor arising from shifting the propability distribution.
        cfb = cmath.exp(xi.dot(xbar)-0.5*xbar.dot(xbar))

        # Operator terms contributing to propagator.
        VHS = self.construct_VHS(system, xshifted)

        # Apply propagator
        walker.phi[:,:system.nup] = self.apply_exponential(walker.phi[:,:system.nup], VHS, False)
        if (system.ndown >0):
            walker.phi[:,system.nup:] = self.apply_exponential(walker.phi[:,system.nup:], VHS, False)

        return (cxf, cfb, xshifted)

    def propagate_walker_free(self, walker, system, trial):
        # 1. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.BH1)
        #
        # 2. Apply 2-body projector
        (cxf, cfb, xmxbar) = self.two_body(walker, system, trial, False)
        #
        # 3. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.BH1)
        walker.inverse_overlap(trial.psi)
        walker.ot = walker.calc_otrial(trial.psi)
        walker.greens_function(trial)
        # Constant terms are included in the walker's weight.
        walker.weight = walker.weight * cxf

    def propagate_walker_phaseless(self, walker, system, trial):
        # 1. Apply one_body propagator.
        kinetic_real(walker.phi, system, self.BH1)
        # 2. Apply two_body propagator.
        (cxf, cfb, xmxbar) = self.two_body(walker, system, trial)
        # 3. Apply one_body propagator.
        kinetic_real(walker.phi, system, self.BH1)

        # Now apply hybrid phaseless approximation
        walker.inverse_overlap(trial.psi)
        walker.greens_function(trial)
        ot_new = walker.calc_otrial(trial.psi)

        # Walker's phase.
        importance_function = self.mf_const_fac*cxf*cfb * ot_new / walker.ot

        dtheta = cmath.phase(importance_function)

        cfac = max(0, math.cos(dtheta))

        rweight = abs(importance_function)
        walker.weight *= rweight * cfac
        walker.ot = ot_new
        walker.field_configs.push_full(xmxbar, cfac, importance_function/rweight)

def unit_test():
    from pauxy.systems.ueg import UEG
    from pauxy.qmc.options import QMCOpts
    from pauxy.trial_wavefunction.hartree_fock import HartreeFock

    inputs = {'nup':1, 'ndown':1,
    'rs':1.0, 'ecut':1.0, 'dt':0.05, 'nwalkers':10}

    system = UEG(inputs, True)

    qmc = QMCOpts(inputs, system, True)

    trial = HartreeFock(system, False, inputs, True)

    propagator = PlaneWave(inputs, qmc, system, trial, True)


if __name__=="__main__":
    unit_test()
