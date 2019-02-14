import cmath
import math
import numpy
import scipy.sparse.linalg
from scipy.linalg import sqrtm
import time
from pauxy.estimators.thermal import one_rdm_from_G, inverse_greens_function_qr
from pauxy.propagation.operations import kinetic_real
from pauxy.utils.linalg import exponentiate_matrix
from pauxy.walkers.single_det import SingleDetWalker

class GenericContinuous(object):
    """Propagator for generic many-electron Hamiltonian.

    Uses continuous HS transformation for exponential of two body operator.

    Parameters
    ----------
    options : dict
        Propagator input options.
    qmc : :class:`pauxy.qmc.options.QMCOpts`
        QMC options.
    system : :class:`pauxy.system.System`
        System object.
    trial : :class:`pauxy.trial_wavefunctioin.Trial`
        Trial wavefunction object.
    verbose : bool
        If true print out more information during setup.
    """

    def __init__(self, options, qmc, system, trial, verbose=False):
        if verbose:
            print ("# Parsing continuous propagator input options.")

        # Input options
        self.hs_type = 'continuous'
        self.free_projection = options.get('free_projection', False)
        if verbose:
            print("# Using phaseless approximation: %r"%(not self.free_projection))
        self.exp_nmax = options.get('expansion_order', 6)
        self.force_bias = options.get('force_bias', True)
        if self.free_projection:
            if verbose:
                print("# Setting force_bias to False with free projection.")
            self.force_bias = False
        else:
            print("# Setting force bias to %r."%self.force_bias)

        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j*self.sqrt_dt

        # Mean field shifts (2,nchol_vec).
        P = one_rdm_from_G(trial.G)
        self.mf_shift = 1j*numpy.einsum('lpq,spq->l', system.chol_vecs, P)

        # Mean field shifted one-body propagator
        self.mu = system.mu
        self.construct_one_body_propagator(qmc.dt, system.chol_vecs,
                                           system.h1e_mod)

        self.BT = numpy.array([(trial.dmat[0]),(trial.dmat[1])])
        self.BTinv = numpy.array([(trial.dmat_inv[0]),(trial.dmat_inv[1])])

        # Constant core contribution modified by mean field shift.
        mf_core = system.ecore + 0.5*numpy.dot(self.mf_shift, self.mf_shift)
        self.mf_const_fac = cmath.exp(-self.dt*mf_core)
        self.BT_BP = self.BH1
        self.nstblz = qmc.nstblz

        self.chol_vecs = system.chol_vecs
        self.ebound = (2.0/self.dt)**0.5
        self.mean_local_energy = 0
        if self.free_projection:
            self.propagate_walker = self.propagate_walker_free
        else:
            self.propagate_walker = self.propagate_walker_phaseless
        if verbose:
            print ("# Finished setting up propagator.")


    def construct_one_body_propagator(self, dt, chol_vecs, h1e_mod):
        """Construct mean-field shifted one-body propagator.

        Parameters
        ----------
        dt : float
            Timestep.
        chol_vecs : :class:`numpy.ndarray`
            Cholesky vectors.
        h1e_mod : :class:`numpy.ndarray`
            One-body operator including factor from factorising two-body
            Hamiltonian.
        """
        shift = 1j*numpy.einsum('l,lpq->pq', self.mf_shift, chol_vecs)
        H1 = h1e_mod - numpy.array([shift,shift])

        I = numpy.identity(H1[0].shape[0], dtype=H1.dtype)
        # No spin dependence for the moment.
        self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]+0.5*dt*self.mu*I),
                                scipy.linalg.expm(-0.5*dt*H1[1]+0.5*dt*self.mu*I)])

    def construct_force_bias(self, Gmod):
        """Compute optimal force bias.

        Uses rotated Green's function.

        Parameters
        ----------
        Gmod : :class:`numpy.ndarray`
            Half-rotated walker's Green's function.

        Returns
        -------
        xbar : :class:`numpy.ndarray`
            Force bias.
        """
        vbias = 1j*numpy.einsum('slrp,spr->l', self.rchol_vecs, Gmod)
        return - self.sqrt_dt * (vbias-self.mf_shift)

    def construct_force_bias_full(self, G):
        """Compute optimal force bias.

        Uses explicit expression.

        Parameters
        ----------
        G: :class:`numpy.ndarray`
            Walker's Green's function.

        Returns
        -------
        xbar : :class:`numpy.ndarray`
            Force bias.
        """
        vbias = numpy.einsum('lpq,pq->l', self.chol_vecs, G[0])
        vbias += numpy.einsum('lpq,pq->l', self.chol_vecs, G[1])
        return - self.sqrt_dt * (1j*vbias-self.mf_shift)

    def two_body(self, walker, system, trial):
        r"""Continuous Hubbard-Statonovich transformation.

        Parameters
        ----------
        walker : :class:`pauxy.walker.Walker` walker object to be updated. On
            output we have acted on phi by B_V(x).
        system : :class:`pauxy.system.System`
            System object.
        trial : :class:`pauxy.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """
        # Normally distrubted auxiliary fields.
        xi = numpy.random.normal(0.0, 1.0, system.nchol_vec)
        if self.force_bias:
            rdm = one_rdm_from_G(walker.G)
            xbar = self.construct_force_bias_full(rdm)
        else:
            xbar = numpy.zeros(xi.shape)
        # Constant factor arising from shifting the propability distribution.
        cfb = xi.dot(xbar) - 0.5*xbar.dot(xbar)
        shifted = xi - xbar
        # Constant factor arising from force bias and mean field shift
        cmf = -self.sqrt_dt * shifted.dot(self.mf_shift)

        # Operator terms contributing to propagator.
        VHS = self.isqrt_dt*numpy.einsum('l,lpq->pq',
                                         shifted, system.chol_vecs)

        return (cmf, cfb, shifted, VHS)

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

    def propagate_walker_free(self, system, walker, trial):
        r"""Free projection for continuous HS transformation.

        .. Warning::
            Currently not implemented.


        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B` and updated the weight
            appropriately. Updates inplace.
        state : :class:`state.State`
            Simulation state.
        """
        (cmf, cfb, xmxbar, VHS) = self.two_body(walker, system, trial)
        BV = scipy.linalg.expm(VHS) # could use a power-series method to build this

        B = numpy.array([BV.dot(self.BH1[0]),BV.dot(self.BH1[1])])
        B = numpy.array([self.BH1[0].dot(B[0]),self.BH1[1].dot(B[1])])

        # Compute determinant ratio det(1+A')/det(1+A).
        # 1. Current walker's green's function.
        G = walker.greens_function(trial, inplace=False)
        # 2. Compute updated green's function.
        walker.stack.update_new(B)
        walker.greens_function(trial, inplace=True)
        # 3. Compute det(G/G')
        M0 = [scipy.linalg.det(G[0]), scipy.linalg.det(G[1])]
        Mnew = [scipy.linalg.det(walker.G[0]), scipy.linalg.det(walker.G[1])]
        # Could save M0 rather than recompute.
        oratio = (M0[0] * M0[1]) / (Mnew[0] * Mnew[1])

        walker.ot = 1.0
        # Constant terms are included in the walker's weight.
        # TODO: Think about mean field subtraction.
        (magn, phase) = cmath.polar(cmath.exp(cmf+cfb)*oratio)
        walker.weight *= magn
        walker.phase *= cmath.exp(1j*phase)

    def propagate_walker_phaseless(self, system, walker, trial):
        r"""Propagate walker using phaseless approximation.

        Uses importance sampling and the hybrid method.

        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. On output we have acted on phi with the
            propagator B(x), and updated the weight appropriately.  Updates
            inplace.
        system : :class:`pauxy.system.System`
            System object.
        trial : :class:`pauxy.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """

        (cmf, cfb, xmxbar, VHS) = self.two_body(walker, system, trial)
        BV = self.exponentiate(VHS)

        B = numpy.array([BV.dot(self.BH1[0]),BV.dot(self.BH1[1])])
        B = numpy.array([self.BH1[0].dot(B[0]),self.BH1[1].dot(B[1])])

        # Compute determinant ratio det(1+A')/det(1+A).
        # 1. Current walker's green's function.
        G = walker.greens_function(trial, inplace=False)
        # 2. Compute updated green's function.
        walker.stack.update(B)
        walker.greens_function(trial, inplace=True)
        # 3. Compute det(G/G')
        M0 = [scipy.linalg.det(G[0], check_finite=False),
              scipy.linalg.det(G[1], check_finite=False)]
        Mnew = [scipy.linalg.det(walker.G[0], check_finite=False),
                scipy.linalg.det(walker.G[1], check_finite=False)]
        # Could save M0 rather than recompute.
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
            # TODO: Think about mean field subtraction.
            dtheta = cmath.phase(cmath.exp(hybrid_energy-cfb))
            cosine_fac = max(0, math.cos(dtheta))
            walker.weight *= magn * cosine_fac
        else:
            walker.weight = 0.0
