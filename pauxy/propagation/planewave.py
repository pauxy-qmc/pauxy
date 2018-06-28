import cmath
import math
import numpy
import scipy.sparse.linalg
import time
from pauxy.propagation.operations import kinetic_real
from pauxy.utils.linalg import exponentiate_matrix
from pauxy.walkers.single_det import SingleDetWalker

class PlaneWave(object):
    """PlaneWave class
    """
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
        H1 = system.h1e_mod
        # No spin dependence for the moment.
        self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]),
                                scipy.linalg.expm(-0.5*dt*H1[1])])

    def apply_exponential(self, phi, VHS, debug=False):
        """Apply exponential propagator of the HS transformation
        Parameters
        ----------
        system :
            system class
        phi : numpy array
            a state
        VHS : numpy array
            HS transformation potential
        Returns
        -------
        phi : numpy array
            Exp(VHS) * phi
        """
        # JOONHO: exact exponential
        # copy = numpy.copy(phi)
        # phi = scipy.linalg.expm(VHS).dot(copy)
        if debug:
            copy = numpy.copy(phi)
            c2 = scipy.linalg.expm(VHS).dot(copy)
        
        numpy.copyto(self.Temp, phi)
        for n in range(1, self.exp_nmax+1):
            self.Temp = VHS.dot(self.Temp) / n
            phi += self.Temp
        if debug:
            print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))
        return phi

    def two_body_potentials(self, system, iq):
        """Calculatate A and B of Eq.(13) of PRB(75)245123 for a given plane-wave vector q
        Parameters
        ----------
        system :
            system class
        q : float
            a plane-wave vector
        Returns
        -------
        iA : numpy array
            Eq.(13a)
        iB : numpy array
            Eq.(13b)
        """
        rho_q = system.density_operator(iq)
        qscaled = system.kfac * system.qvecs[iq]

        # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol
        piovol = math.pi / (system.vol)
        factor = (piovol/numpy.dot(qscaled,qscaled))**0.5

        # JOONHO: include a factor of 1j
        iA = 1j * factor * (rho_q + rho_q.getH()) 
        iB = - factor * (rho_q - rho_q.getH()) 
        return (iA, iB)

    def construct_force_bias(self, system, G):
        """Compute the force bias term as in Eq.(33) of DOI:10.1002/wcms.1364
        Parameters
        ----------
        system :
            system class
        G : numpy array
            Green's function
        Returns
        -------
        force bias : numpy array
            -sqrt(dt) * vbias
        """
        for (i, qi) in enumerate(system.qvecs):
            (iA, iB) = self.two_body_potentials(system, i)
            # Deal with spin more gracefully
            self.vbias[i] = iA.dot(G[0]).diagonal().sum() + iA.dot(G[1]).diagonal().sum()
            self.vbias[i+self.num_vplus] = iB.dot(G[0]).diagonal().sum() + iB.dot(G[1]).diagonal().sum()
        return - self.sqrt_dt * self.vbias

    def construct_VHS(self, system, xshifted):
        import numpy.matlib
        """Construct the one body potential from the HS transformation
        Parameters
        ----------
        system :
            system class
        xshifted : numpy array
            shifited auxiliary field
        Returns
        -------
        VHS : numpy array
            the HS potential
        """
        VHS = numpy.zeros((system.nbasis, system.nbasis), dtype=numpy.complex128 )

        for (i, qi) in enumerate(system.qvecs):
            (iA, iB) = self.two_body_potentials(system, i)
            VHS = VHS + (xshifted[i] * iA).todense()
            VHS = VHS + (xshifted[i+self.num_vplus] * iB).todense()
        return  VHS * self.sqrt_dt


    def construct_VHS_incore(self, system, xshifted):
        import numpy.matlib
        """Construct the one body potential from the HS transformation
        Parameters
        ----------
        system :
            system class
        xshifted : numpy array
            shifited auxiliary field
        Returns
        -------
        VHS : numpy array
            the HS potential
        """
        VHS = numpy.zeros((system.nbasis, system.nbasis), dtype=numpy.complex128 )
        VHS = system.iA * xshifted[:self.num_vplus] + system.iB * xshifted[self.num_vplus:]
        VHS = VHS.reshape(system.nbasis, system.nbasis)
        return  VHS * self.sqrt_dt

    def construct_force_bias_incore(self, system, G):
        """Compute the force bias term as in Eq.(33) of DOI:10.1002/wcms.1364
        Parameters
        ----------
        system :
            system class
        G : numpy array
            Green's function
        Returns
        -------
        force bias : numpy array
            -sqrt(dt) * vbias
        """
        Gvec = G.reshape(2, system.nbasis*system.nbasis)
        self.vbias[:self.num_vplus] = Gvec[0].T*system.iA + Gvec[1].T*system.iA
        self.vbias[self.num_vplus:] = Gvec[0].T*system.iB + Gvec[1].T*system.iB
        return - self.sqrt_dt * self.vbias

    def two_body_propagator(self, walker, system, fb = True):
        """It appliese the two-body propagator
        Parameters
        ----------
        walker :
            walker class
        system :
            system class
        fb : boolean
            wheter to use force bias
        Returns
        -------
        cxf : float
            the constant factor arises from mean-field shift (hard-coded for UEG for now)
        cfb : float
            the constant factor arises from the force-bias
        xshifted : numpy array
            shifited auxiliary field
        """
        # Normally distrubted auxiliary fields.
        xi = numpy.random.normal(0.0, 1.0, system.nfields)

        # Optimal force bias.
        xbar = numpy.zeros(system.nfields)
        if (fb):
            xbar = self.construct_force_bias_incore(system, walker.G)
        
        xshifted = xi - xbar

        # Constant factor arising from force bias and mean field shift
        # Mean field shift is zero for UEG in HF basis
        cxf = 1.0
        # Constant factor arising from shifting the propability distribution.
        cfb = cmath.exp(xi.dot(xbar)-0.5*xbar.dot(xbar))

        # Operator terms contributing to propagator.
        # VHS = self.construct_VHS(system, xshifted)
        VHS = self.construct_VHS_incore(system, xshifted)

        # Apply propagator
        walker.phi[:,:system.nup] = self.apply_exponential(walker.phi[:,:system.nup], VHS, False)
        if (system.ndown >0):
            walker.phi[:,system.nup:] = self.apply_exponential(walker.phi[:,system.nup:], VHS, False)

        return (cxf, cfb, xshifted)

    def propagate_walker_free(self, walker, system, trial):
        """Free projection propagator
        Parameters
        ----------
        walker :
            walker class
        system :
            system class
        trial : 
            trial wavefunction class
        Returns
        -------
        """
        # 1. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.BH1)
        #
        # 2. Apply 2-body projector
        (cxf, cfb, xmxbar) = self.two_body_propagator(walker, system, False)
        #
        # 3. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.BH1)
        walker.inverse_overlap(trial.psi)
        walker.ot = walker.calc_otrial(trial.psi)
        walker.greens_function(trial)
        # Constant terms are included in the walker's weight.
        walker.weight = walker.weight * cxf

    def propagate_walker_phaseless(self, walker, system, trial):
        """Phaseless propagator
        Parameters
        ----------
        walker :
            walker class
        system :
            system class
        trial : 
            trial wavefunction class
        Returns
        -------
        """
        # 1. Apply one_body propagator.
        kinetic_real(walker.phi, system, self.BH1)
        # 2. Apply two_body propagator.
        (cxf, cfb, xmxbar) = self.two_body_propagator(walker, system)
        # 3. Apply one_body propagator.
        kinetic_real(walker.phi, system, self.BH1)

        # Now apply hybrid phaseless approximation
        walker.inverse_overlap(trial.psi)
        walker.greens_function(trial)
        ot_new = walker.calc_otrial(trial.psi)

        # Walker's phase.
        if (walker.ot < 1e-8):
            walker.ot = ot_new
            walker.weight = 0.0
            walker.field_configs.push_full(xmxbar, 0.0, 0.0)
        else:
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
