import cmath
import math
import numpy
import scipy.sparse.linalg
from scipy.linalg import sqrtm
import time
from pauxy.estimators.thermal import one_rdm_from_G
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
        self.nstblz = qmc.nstblz
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j*self.sqrt_dt
        self.num_vplus = system.nfields // 2
        
        print("# Number of fields = %i"%system.nfields)

        self.vbias = numpy.zeros(system.nfields, dtype=numpy.complex128)

        # Constant core contribution modified by mean field shift.
        mf_core = system.ecore

        # square root is necessary for symmetric Trotter split
        # self.BH1 = numpy.array([sqrtm(trial.dmat[0]),sqrtm(trial.dmat[1])])
        # self.BH1inv = numpy.array([sqrtm(trial.dmat_inv[0]),sqrtm(trial.dmat_inv[1])])
        self.BH1 = numpy.array([(trial.dmat[0]),(trial.dmat[1])])
        self.BH1inv = numpy.array([(trial.dmat_inv[0]),(trial.dmat_inv[1])])
        self.mf_const_fac = 1

        # todo : ?
        self.BT_BP = self.BH1
        self.nstblz = qmc.nstblz

        # Temporary array for matrix exponentiation.
        self.Temp = numpy.zeros(trial.dmat.shape,dtype=trial.dmat.dtype)

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

    # def apply_exponential(self, phi, VHS, debug=False):
    #     """Apply exponential propagator of the HS transformation
    #     Parameters
    #     ----------
    #     system :
    #         system class
    #     phi : numpy array
    #         a state
    #     VHS : numpy array
    #         HS transformation potential
    #     Returns
    #     -------
    #     phi : numpy array
    #         Exp(VHS) * phi
    #     """
    #     # JOONHO: exact exponential
    #     # copy = numpy.copy(phi)
    #     # phi = scipy.linalg.expm(VHS).dot(copy)
    #     if debug:
    #         copy = numpy.copy(phi)
    #         c2 = scipy.linalg.expm(VHS).dot(copy)
        
    #     numpy.copyto(self.Temp, phi)
    #     for n in range(1, self.exp_nmax+1):
    #         self.Temp = VHS.dot(self.Temp) / n
    #         phi += self.Temp
    #     if debug:
    #         print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))
    #     return phi
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

    def propagate_greens_function(self, walker, B, Binv):
        if walker.stack.time_slice < walker.stack.ntime_slices:
            walker.G[0] = B[0].dot(walker.G[0]).dot(Binv[0])
            walker.G[1] = B[1].dot(walker.G[1]).dot(Binv[1])

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
            rdm = one_rdm_from_G(walker.G)
            xbar = self.construct_force_bias_incore(system, rdm)
        
        xshifted = xi - xbar

        # Constant factor arising from force bias and mean field shift
        # Mean field shift is zero for UEG in HF basis
        cxf = 1.0
        # Constant factor arising from shifting the propability distribution.
        cfb = cmath.exp(xi.dot(xbar)-0.5*xbar.dot(xbar))

        # Operator terms contributing to propagator.
        VHS = self.construct_VHS_incore(system, xshifted)


        return (cxf, cfb, xshifted, VHS)

    def propagate_walker_free(self, system, walker, trial):
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

        (cxf, cfb, xmxbar, VHS) = self.two_body_propagator(walker, system, False)
        BV = scipy.linalg.expm(VHS) # could use a power-series method to build this
        # B = numpy.array([BV.dot(self.BH1[0]),BV.dot(self.BH1[1])])
        # B = numpy.array([self.BH1[0].dot(B[0]),self.BH1[1].dot(B[1])])
        
        # B = numpy.array([self.BH1[0].dot(BV),self.BH1[1].dot(BV)])
        B = numpy.array([BV.dot(self.BH1[0]),BV.dot(self.BH1[1])])

        # B = numpy.array([BV.dot(self.BH1inv[0]),BV.dot(self.BH1inv[1])])
        # B = numpy.array([self.BH1[0].dot(B[0]),self.BH1[1].dot(B[1])])
        walker.stack.update(B)

        walker.ot = 1.0
        # Constant terms are included in the walker's weight.
        walker.weight = walker.weight * cxf

        if walker.stack.time_slice % self.nstblz == 0:
            walker.greens_function(None, walker.stack.time_slice-1)

    def propagate_walker_phaseless(self, system, walker, time_slice):
        # """Phaseless propagator
        # Parameters
        # ----------
        # walker :
        #     walker class
        # system :
        #     system class
        # trial : 
        #     trial wavefunction class
        # Returns
        # -------
        # """
        (cxf, cfb, xmxbar) = self.two_body_propagator(walker, system)
        BV = scipy.linalg.expm(VHS) # could use a power-series method to build this
        BVinv = scipy.linalg.expm(-VHS) # could use a power-series method to build this
        B = numpy.array([BV.dot(self.BH1[0]),BV.dot(self.BH1[1])])
        B = numpy.array([self.BH1[0].dot(B[0]),self.BH1[1].dot(B[1])])
        walker.stack.update(B)

        Binv = numpy.array([BVinv.dot(self.BH1inv[0]),BVinv.dot(self.BH1inv[1])])
        Binv = numpy.array([self.BH1inv[0].dot(Binv[0]),self.BH1inv[1].dot(Binv[1])])

    # def propagate_greens_function(self, walker, B, Binv):

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
    # from pauxy.trial_wavefunction.hartree_fock import HartreeFock

    inputs = {'nup':1, 'ndown':1,
    'rs':1.0, 'ecut':1.0, 'dt':0.05, 'nwalkers':10}

    # system = UEG(inputs, True)

    # qmc = QMCOpts(inputs, system, True)

    # trial = HartreeFock(system, False, inputs, True)

    # propagator = PlaneWave(inputs, qmc, system, trial, True)


if __name__=="__main__":
    unit_test()
