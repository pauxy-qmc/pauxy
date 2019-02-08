import cmath
import math
import numpy
import scipy.sparse.linalg
from scipy.linalg import sqrtm
import time
from pauxy.estimators.thermal import (
        one_rdm_from_G, inverse_greens_function_qr,
        )
from pauxy.propagation.operations import kinetic_real
from pauxy.utils.linalg import exponentiate_matrix
from pauxy.walkers.single_det import SingleDetWalker

class PlaneWave(object):
    """PlaneWave class
    """
    def __init__(self, options, qmc, system, trial, verbose=False):
        self.verbose = verbose
        if verbose:
            print ("# Parsing plane wave propagator input options.")
        # Input options
        self.hs_type = 'plane_wave'
        self.free_projection = options.get('free_projection', False)
        self.optimised = options.get('optimised', True)
        self.exp_nmax = options.get('expansion_order', 4)
        self.nstblz = qmc.nstblz
        self.fb_bound = options.get('fb_bound', 1.0)
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j*self.sqrt_dt
        self.num_vplus = system.nfields // 2
        if verbose:
            print("# Number of fields = %i"%system.nfields)

        self.vbias = numpy.zeros(system.nfields, dtype=numpy.complex128)

        # Constant core contribution modified by mean field shift.
        mf_core = system.ecore

        self.construct_one_body_propagator(system, qmc.dt)

        self.BT = numpy.array([(trial.dmat[0]),(trial.dmat[1])])
        self.BTinv = numpy.array([(trial.dmat_inv[0]),(trial.dmat_inv[1])])

        self.mf_const_fac = 1

        # todo : ?
        self.BT_BP = self.BT
        self.nstblz = qmc.nstblz

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
        self.nfb_trig = False

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
        I = numpy.identity(H1[0].shape[0], dtype=H1.dtype)
        # No spin dependence for the moment.
        self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]+0.5*dt*system.mu*I),
                                scipy.linalg.expm(-0.5*dt*H1[1]+0.5*dt*system.mu*I)])

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

    def construct_VHS_outofcore(self, system, xshifted):
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

    def propagate_greens_function(self, walker, B, Binv):
        if walker.stack.time_slice < walker.stack.ntime_slices:
            walker.G[0] = B[0].dot(walker.G[0]).dot(Binv[0])
            walker.G[1] = B[1].dot(walker.G[1]).dot(Binv[1])

    def two_body_propagator(self, walker, system, force_bias=True):
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
        if force_bias:
            rdm = one_rdm_from_G(walker.G)
            xbar = self.construct_force_bias_incore(system, rdm)

        for i in range(system.nfields):
            if numpy.absolute(xbar[i]) > self.fb_bound:
                if not self.nfb_trig and self.verbose:
                    print("# Rescaling force bias is triggered.")
                    print("# Warning will only be printed once per thread.")
                    print("# Bound = {}".format(self.fb_bound))
                    xb = (xbar[i].real, xbar[i].imag)
                    vb = abs(xbar[i]) / self.sqrt_dt
                    vb = (vb.real, vb.imag)
                    print("XBAR: (%f,%f)"%xb)
                    print("<v>: (%f,%f)"%vb)
                    self.nfb_trig = True
                walker.rescaled_fb = True
                xbar[i] /= numpy.absolute(xbar[i])

        xshifted = xi - xbar

        # Constant factors: Note they are not exponentiated.
        # Constant factor arising from force bias and mean field shift
        # Mean field shift is zero for UEG in HF basis
        cmf = 0.0
        # Constant factor arising from shifting the propability distribution.
        cfb = xi.dot(xbar) - 0.5*xbar.dot(xbar)

        # print(xbar.dot(xbar))

        # Operator terms contributing to propagator.
        VHS = self.construct_VHS_incore(system, xshifted)

        return (cmf, cfb, xshifted, VHS)

    def exponentiate(self, VHS, debug=False):
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
        phi = numpy.identity(VHS.shape[0], dtype = numpy.complex128)
        if debug:
            copy = numpy.copy(phi)
            c2 = scipy.linalg.expm(VHS).dot(copy)

        Temp = numpy.identity(VHS.shape[0], dtype = numpy.complex128)

        for n in range(1, self.exp_nmax+1):
            Temp = VHS.dot(Temp) / n
            phi += Temp
        if debug:
            print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))
        return phi

    def propagate_walker_free(self, system, walker, trial, force_bias=False):
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

        (cmf, cfb, xmxbar, VHS) = self.two_body_propagator(walker, system,
                                                           force_bias=force_bias)
        BV = self.exponentiate(VHS) # could use a power-series method to build this

        B = numpy.array([
            numpy.einsum('ij,jj->ij',BV,self.BH1[0]),
            numpy.einsum('ij,jj->ij',BV,self.BH1[1])
            ])
        B = numpy.array([
            numpy.einsum('ii,ij->ij',self.BH1[0],B[0]),
            numpy.einsum('ii,ij->ij',self.BH1[1],B[1])
            ])

        # Compute determinant ratio det(1+A')/det(1+A).
        if self.optimised:
            icur = walker.stack.time_slice // walker.stack.stack_size
            inext = (walker.stack.time_slice+1) // walker.stack.stack_size
            if (walker.stack.counter == 0):
                walker.compute_left_right(icur)
            # else:
            #     walker.compute_right(icur)
            # print(walker.Tr[0])
            # print(walker.Tr[1])
            # 1. Current walker's green's function.
            # Green's function that takes Left Right and Center
            G = walker.greens_function_left_right(icur, inplace=False)
            # 2. Compute updated green's function.
            walker.stack.update_new(B)
            walker.greens_function_left_right(icur, inplace=True)
        else:
            # Compute determinant ratio det(1+A')/det(1+A).
            # 1. Current walker's green's function.
            G = walker.greens_function(None, slice_ix=walker.stack.ntime_slices,
                                        inplace=False)
            # 2. Compute updated green's function.
            walker.stack.update_new(B)
            walker.greens_function(None, slice_ix=walker.stack.ntime_slices,
                                        inplace=True)

        # 3. Compute det(G/G')
        M0 = [scipy.linalg.det(G[0], check_finite=False), scipy.linalg.det(G[1], check_finite=False)]
        Mnew = [scipy.linalg.det(walker.G[0], check_finite=False), scipy.linalg.det(walker.G[1], check_finite=False)]
        try:
            # Could save M0 rather than recompute.
            oratio = (M0[0] * M0[1]) / (Mnew[0] * Mnew[1])

            walker.ot = 1.0
            # Constant terms are included in the walker's weight.
            (magn, phase) = cmath.polar(cmath.exp(cmf+cfb)*oratio)
            walker.weight *= magn
            walker.phase *= cmath.exp(1j*phase)
        except ZeroDivisionError:
            walker.weight = 0.0


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

        (cmf, cfb, xmxbar, VHS) = self.two_body_propagator(walker, system, True)
        BV = self.exponentiate(VHS) # could use a power-series method to build this

        B = numpy.array([
            numpy.einsum('ij,jj->ij',BV,self.BH1[0]),
            numpy.einsum('ij,jj->ij',BV,self.BH1[1])
            ])
        B = numpy.array([
            numpy.einsum('ii,ij->ij',self.BH1[0],B[0]),
            numpy.einsum('ii,ij->ij',self.BH1[1],B[1])
            ])

        if self.optimised:
            icur = walker.stack.time_slice // walker.stack.stack_size
            inext = (walker.stack.time_slice+1) // walker.stack.stack_size

            if (walker.stack.counter == 0):
                walker.compute_left_right(icur)
            # else:
            #     walker.compute_right(icur)
            # 1. Current walker's green's function.
            # Green's function that takes Left Right and Center
            G = walker.greens_function_left_right(icur, inplace=False)
            # 2. Compute updated green's function.
            walker.stack.update_new(B)
            walker.greens_function_left_right(icur, inplace=True)
        else:
            # Compute determinant ratio det(1+A')/det(1+A).
            # 1. Current walker's green's function.
            G = walker.greens_function(None, slice_ix=walker.stack.ntime_slices,
                                        inplace=False)
            # 2. Compute updated green's function.
            walker.stack.update_new(B)
            walker.greens_function(None, slice_ix=walker.stack.ntime_slices,
                                        inplace=True)

        # 3. Compute det(G/G')
        M0 = [scipy.linalg.det(G[0], check_finite = False), scipy.linalg.det(G[1], check_finite=False)]
        Mnew = [scipy.linalg.det(walker.G[0], check_finite = False), scipy.linalg.det(walker.G[1], check_finite=False)]
        # Could save M0 rather than recompute.
        try:
            oratio = (M0[0] * M0[1]) / (Mnew[0] * Mnew[1])

            # Might want to cap this at some point
            hybrid_energy = cmath.log(oratio) + cfb + cmf
            Q = cmath.exp(hybrid_energy)
            expQ = self.mf_const_fac * Q
            (magn, phase) = cmath.polar(expQ)

            if not math.isinf(magn):
                # Determine cosine phase from Arg(det(1+A'(x))/det(1+A(x))).
                # Note this doesn't include exponential factor from shifting
                # propability distribution.
                dtheta = cmath.phase(cmath.exp(hybrid_energy-cfb))
                cosine_fac = max(0, math.cos(dtheta))
                walker.weight *= magn * cosine_fac
            else:
                walker.weight = 0.0
        except ZeroDivisionError:
            walker.weight = 0.0

    def propagate_greens_function(self, walker):
        if walker.stack.time_slice < walker.stack.ntime_slices:
            walker.G[0] = self.BT[0].dot(walker.G[0]).dot(self.BTinv[0])
            walker.G[1] = self.BT[1].dot(walker.G[1]).dot(self.BTinv[1])


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
