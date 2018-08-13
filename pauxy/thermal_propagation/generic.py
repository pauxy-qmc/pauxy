import cmath
import math
import numpy
import scipy.linalg
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
        self.exp_nmax = options.get('expansion_order', 6)
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j*self.sqrt_dt
        # Mean field shifts (2,nchol_vec).
        self.mf_shift = 1j*numpy.einsum('lpq,spq->l', system.chol_vecs, trial.G)
        # Mean field shifted one-body propagator
        self.construct_one_body_propagator(qmc.dt, system.chol_vecs,
                                           system.h1e_mod)
        # Constant core contribution modified by mean field shift.
        mf_core = system.ecore + 0.5*numpy.dot(self.mf_shift, self.mf_shift)
        self.mf_const_fac = cmath.exp(-self.dt*mf_core)
        self.BT_BP = self.BH1
        self.nstblz = qmc.nstblz

        self.mu = system.mu

        print("self.mu = {}".format(self.mu))

        # Temporary array for matrix exponentiation.
        # self.Temp = numpy.zeros(trial.psi[:,:system.nup].shape,
        #                         dtype=trial.psi.dtype)
        # Half rotated cholesky vectors (by trial wavefunction).
        # Assuming nup = ndown here
        # rotated_up = numpy.einsum('rp,lpq->lrq',
        #                           trial.psi[:,:system.nup].conj().T,
        #                           system.chol_vecs)
        # rotated_down = numpy.einsum('rp,lpq->lrq',
        #                             trial.psi[:,system.nup:].conj().T,
        #                             system.chol_vecs)
        # self.rchol_vecs = numpy.array([rotated_up, rotated_down])
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

        # self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]),
                                # scipy.linalg.expm(-0.5*dt*H1[1])])

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

    def two_body(self, walker, system, trial, fb=True):
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
        if (fb):
            # Construct walker modified Green's function.
            walker.inverse_overlap(trial.psi)
            walker.rotated_greens_function()
            # Optimal force bias.
            xbar = self.construct_force_bias(walker.Gmod)
        else:
            xbar = numpy.zeros(xi.shape)
        # Constant factor arising from shifting the propability distribution.
        c_fb = cmath.exp(xi.dot(xbar)-0.5*xbar.dot(xbar))
        shifted = xi - xbar
        # Constant factor arising from force bias and mean field shift
        c_xf = cmath.exp(-self.sqrt_dt*shifted.dot(self.mf_shift))

        # Operator terms contributing to propagator.
        VHS = self.isqrt_dt*numpy.einsum('l,lpq->pq', shifted, system.chol_vecs)

        return (c_xf, c_fb, shifted, VHS)

    def apply_exponential(self, phi, VHS, debug=False):
        """Apply matrix expoential to wavefunction approximately.

        Parameters
        ----------
        phi : :class:`numpy.ndarray`
            Walker's wavefunction. On output phi = exp(VHS)*phi.
        VHS : :class:`numpy.ndarray`
            Hubbard Stratonovich matrix (~1j*sqrt_dt*\sum_\gamma x_\gamma v_\gamma)
        debug : bool
            If true check accuracy of matrix exponential through direct
            exponentiation.
        """
        if debug:
            copy = numpy.copy(phi)
            c2 = scipy.linalg.expm(VHS).dot(copy)
        numpy.copyto(self.Temp, phi)
        for n in range(1, self.exp_nmax+1):
            self.Temp = VHS.dot(self.Temp) / n
            phi += self.Temp
        if debug:
            print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))

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

        (cxf, cfb, xmxbar, VHS) = self.two_body(walker, system, trial, False)
        BV = scipy.linalg.expm(VHS) # could use a power-series method to build this

        B = numpy.array([BV.dot(self.BH1[0]),BV.dot(self.BH1[1])])
        B = numpy.array([self.BH1[0].dot(B[0]),self.BH1[1].dot(B[1])])

        walker.stack.update(B)

        walker.ot = 1.0
        # Constant terms are included in the walker's weight.
        (magn, dtheta) = cmath.polar(cxf)
        walker.weight *= magn
        walker.phase *= cmath.exp(1j*dtheta)


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

        # 1. Apply one_body propagator.
        kinetic_real(walker.phi, system, self.BH1)
        # 2. Apply two_body propagator.
        (cmf, cfb, xmxbar) = self.two_body(walker, system, trial)
        # 3. Apply one_body propagator.
        kinetic_real(walker.phi, system, self.BH1)

        # Now apply hybrid phaseless approximation
        walker.inverse_overlap(trial.psi)
        ot_new = walker.calc_otrial(trial.psi)
        # Walker's phase.
        importance_function = self.mf_const_fac*cmf*cfb*ot_new / walker.ot
        dtheta = cmath.phase(importance_function)
        cfac = max(0, math.cos(dtheta))
        rweight = abs(importance_function)
        walker.weight *= rweight * cfac
        walker.ot = ot_new
        walker.field_configs.push_full(xmxbar, cfac, importance_function/rweight)
