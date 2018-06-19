import cmath
import math
import numpy
import scipy.linalg
import time
from pauxy.propagation.operations import kinetic_real
from pauxy.utils.linalg import exponentiate_matrix
from pauxy.walkers.single_det import SingleDetWalker

class PlaneWave(object):
    '''Base propagator class'''

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
            self.propagate_walker = self.propagate_walker_free
        else:
            self.propagate_walker = self.propagate_walker_phaseless

        if verbose:
            print ("# Finished setting up propagator.")

    def two_body_potentials(self, system, q):
        rho_q = system.density_operator(q)
        qscaled = system.kfac * q
        factor = ((2.0*math.pi)/(system.vol*numpy.dot(qscaled,qscaled)))**0.5

        # JOONHO: include a factor of 1j
        A = 1j * factor * (rho_q + rho_q.conj().T) * 0.5
        B = - factor * (rho_q - rho_q.conj().T) * 0.5

        return (A, B)

    def construct_one_body_propagator(self, system, dt):
        H1 = system.h1e_mod
        # No spin dependence for the moment.
        self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]),
                                scipy.linalg.expm(-0.5*dt*H1[1])])

    def construct_force_bias(self, system, G):
        for (i, qi) in enumerate(system.qvecs):
            (A, B) = self.two_body_potentials(system, qi)
            # Deal with spin more gracefully
            self.vbias[i] = numpy.einsum('ij,kij->', A, G)
            self.vbias[i+self.num_vplus] = numpy.einsum('ij,kij->', B, G)
        return - self.sqrt_dt * self.vbias

    def construct_VHS(self, system, xshifted):
        VHS = numpy.zeros(shape=(system.nbasis,system.nbasis),
                          dtype=numpy.complex128)
        for (i, qi) in enumerate(system.qvecs):
            (A, B) = self.two_body_potentials(system, qi)
            VHS = VHS + xshifted[i] * A 
            VHS = VHS + xshifted[i+self.num_vplus] * B 
        return  VHS * self.sqrt_dt

    def two_body(self, walker, system, trial, fb = True):
        r"""Continuous Hubbard-Statonovich transformation for Hubbard model.
        Only requires M auxiliary fields.
        Parameters
        ----------
        walker : :class:`pauxy.walker.Walker`
            walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`b_v` and updated the weight appropriately.
            updates inplace.
        state : :class:`pauxy.state.State`
            Simulation state.
        """
        # Construct walker modified Green's function.
        # walker.rotated_greens_function()
        walker.inverse_overlap(trial.psi)
        # print("inverse_overlap (seconds) %10.5f"%(end - start0))
        walker.greens_function(trial)
        # print("greens_function (seconds) %10.5f"%(end - start))

        # Normally distrubted auxiliary fields.
        xi = numpy.random.normal(0.0, 1.0, system.nfields)

        # Optimal force bias.
        xbar = numpy.zeros(system.nfields)
        if (fb):
            xbar = self.construct_force_bias(system, walker.G)
            # xbar = numpy.zeros(system.nfields)
        
        xshifted = xi - xbar

        # Constant factor arising from force bias and mean field shift
        # Mean field shift is zero for UEG in HF basis
        c_xf = 1.0

        # Constant factor arising from shifting the propability distribution.
        c_fb = cmath.exp(xi.dot(xbar)-0.5*xbar.dot(xbar))

        # Operator terms contributing to propagator.
        VHS = self.construct_VHS(system, xshifted)
        # print("construct_VHS (seconds) %10.5f"%(end - start))
        nup = system.nup

        # Apply propagator
        walker.phi[:,:nup] = self.apply_exponential(walker.phi[:,:nup], VHS, False)
        walker.phi[:,nup:] = self.apply_exponential(walker.phi[:,nup:], VHS, False)
        # print("apply_exponential (seconds) %10.5f"%(end - start))

        # print("two_body (seconds) %10.5f"%(end - start0))
        # exit()
        return (c_xf, c_fb, xshifted)

    def apply_exponential(self, phi, VHS, debug=False):
        # JOONHO: exact exponential
        # copy = numpy.copy(phi)
        # # print ("phi before expm")
        # # print (phi)
        # phi = scipy.linalg.expm(VHS).dot(copy)
        # # print ("phi after expm")
        # # print (phi)
        # return phi
        if debug:
            copy = numpy.copy(phi)
            c2 = scipy.linalg.expm(VHS).dot(copy)
        numpy.copyto(self.Temp, phi)
        for n in range(1, self.exp_nmax+1):
            self.Temp = VHS.dot(self.Temp) / n
            phi += self.Temp
        if debug:
            print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))
        # print("apply_exponential (seconds) %10.5f"%(end - start))
        return phi

    def propagate_walker_free(self, walker, system, trial):
        r"""Free projection for continuous HS transformation.
        TODO: update if ever adapted to other model types.
        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B` and updated the weight
            appropriately. Updates inplace.
        state : :class:`state.State`
            Simulation state.
        """
        nup = system.nup
        # 1. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.BH1)
        #
        (cxf, cfb, xmxbar) = self.two_body(walker, system, trial)
        # # Normally distributed random numbers.
        # xfields =  numpy.random.normal(0.0, 1.0, system.nbasis)
        # sxf = sum(xfields)
        # # Constant, field dependent term emerging when subtracting mean-field.
        # sc = 0.5*self.ut_fac*self.mf_nsq-self.iut_fac*self.mf_shift*sxf
        # c_xf = cmath.exp(sc)
        # # Potential propagator.
        # s = self.iut_fac*xfields + 0.5*self.ut_fac*(1-2*self.mf_shift)
        # bv = numpy.diag(numpy.exp(s))
        # # 2. Apply potential projector.
        # walker.phi[:,:nup] = bv.dot(walker.phi[:,:nup])
        # walker.phi[:,nup:] = bv.dot(walker.phi[:,nup:])
        # 3. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.BH1)
        walker.inverse_overlap(trial.psi)
        walker.ot = walker.calc_otrial(trial.psi)
        walker.greens_function(trial)
        # Constant terms are included in the walker's weight.
        walker.weight = walker.weight * cxf

    def propagate_walker_phaseless(self, walker, system, trial):
        r"""Wrapper function for propagation using continuous transformation.
        This applied the phaseless, local energy approximation and uses importance
        sampling.
        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B` and updated the weight
            appropriately. Updates inplace.
        state : :class:`state.State`
            Simulation state.
        """

        # 1. Apply one_body propagator.
        # print ("before kinetic_real")
        # print (walker.phi)
        kinetic_real(walker.phi, system, self.BH1)
        # print ("after kinetic_real")
        # print (walker.phi)
        # exit()
        # 2. Apply two_body propagator.
        (cxf, cfb, xmxbar) = self.two_body(walker, system, trial)
        # print ("after two_body")
        # print (walker.phi)
        # exit()
        # 3. Apply one_body pQropagator.
        kinetic_real(walker.phi, system, self.BH1)
        # Now apply hybrid phaseless approximation
        walker.inverse_overlap(trial.psi)
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
