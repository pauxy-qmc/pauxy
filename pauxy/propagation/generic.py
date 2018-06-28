import cmath
import math
import numpy
import scipy.linalg
from pauxy.propagation.operations import kinetic_real
from pauxy.utils.linalg import exponentiate_matrix
from pauxy.walkers.single_det import SingleDetWalker

class GenericContinuous(object):
    '''Base propagator class'''

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
        # todo : ?
        self.BT_BP = self.BH1
        self.nstblz = qmc.nstblz
        # Temporary array for matrix exponentiation.
        self.Temp = numpy.zeros(trial.psi[:,:system.nup].shape,
                                dtype=trial.psi.dtype)
        # Rotated cholesky vectors.
        # Assuming nup = ndown here
        rotated_up = numpy.einsum('rp,lpq->lrq',
                                  trial.psi[:,:system.nup].conj().T,
                                  system.chol_vecs)
        rotated_down = numpy.einsum('rp,lpq->lrq',
                                    trial.psi[:,system.nup:].conj().T,
                                    system.chol_vecs)
        self.rchol_vecs = numpy.array([rotated_up, rotated_down])
        # todo : remove
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
        shift = 1j*numpy.einsum('l,lpq->pq', self.mf_shift, chol_vecs)
        H1 = h1e_mod - numpy.array([shift,shift])
        self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]),
                                scipy.linalg.expm(-0.5*dt*H1[1])])

    def construct_force_bias_opt(self, Gmod):
        vbias = 1j*numpy.einsum('slrp,spr->l', self.rchol_vecs, Gmod)
        return - self.sqrt_dt * (vbias-self.mf_shift)

    def construct_force_bias(self, G):
        vbias = numpy.einsum('lpq,pq->l', self.chol_vecs, G[0])
        vbias += numpy.einsum('lpq,pq->l', self.chol_vecs, G[1])
        return - self.sqrt_dt * (1j*vbias-self.mf_shift)

    def two_body(self, walker, system, trial):
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
        # walker.greens_function(trial)
        walker.rotated_greens_function()
        # Normally distrubted auxiliary fields.
        xi = numpy.random.normal(0.0, 1.0, system.nchol_vec)
        # Optimal force bias.
        xbar = self.construct_force_bias_opt(walker.Gmod)
        # xbar2 = self.construct_force_bias(walker.G)
        shifted = xi - xbar
        # Constant factor arising from force bias and mean field shift
        c_xf = cmath.exp(-self.sqrt_dt*shifted.dot(self.mf_shift))
        # Constant factor arising from shifting the propability distribution.
        c_fb = cmath.exp(xi.dot(xbar)-0.5*xbar.dot(xbar))
        # Operator terms contributing to propagator.
        VHS = self.isqrt_dt*numpy.einsum('l,lpq->pq', shifted, system.chol_vecs)
        nup = system.nup
        # Apply propagator
        self.apply_exponential(walker.phi[:,:nup], VHS)
        self.apply_exponential(walker.phi[:,nup:], VHS)

        return (c_xf, c_fb, shifted)

    def apply_exponential(self, phi, VHS, debug=False):
        if debug:
            copy = numpy.copy(phi)
            c2 = scipy.linalg.expm(VHS).dot(copy)
        numpy.copyto(self.Temp, phi)
        for n in range(1, self.exp_nmax+1):
            self.Temp = VHS.dot(self.Temp) / n
            phi += self.Temp
        if debug:
            print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))

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
        kinetic_real(walker.phi, system, self.bt2)
        # Normally distributed random numbers.
        xfields =  numpy.random.normal(0.0, 1.0, system.nbasis)
        sxf = sum(xfields)
        # Constant, field dependent term emerging when subtracting mean-field.
        sc = 0.5*self.ut_fac*self.mf_nsq-self.iut_fac*self.mf_shift*sxf
        c_xf = cmath.exp(sc)
        # Potential propagator.
        s = self.iut_fac*xfields + 0.5*self.ut_fac*(1-2*self.mf_shift)
        bv = numpy.diag(numpy.exp(s))
        # 2. Apply potential projector.
        walker.phi[:,:nup] = bv.dot(walker.phi[:,:nup])
        walker.phi[:,nup:] = bv.dot(walker.phi[:,nup:])
        # 3. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.bt2)
        walker.inverse_overlap(trial.psi)
        walker.ot = walker.calc_otrial(trial.psi)
        walker.greens_function(trial)
        # Constant terms are included in the walker's weight.
        walker.weight = walker.weight * c_xf

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
        kinetic_real(walker.phi, system, self.BH1)
        # 2. Apply two_body propagator.
        (cxf, cfb, xmxbar) = self.two_body(walker, system, trial)
        # 3. Apply one_body propagator.
        kinetic_real(walker.phi, system, self.BH1)

        # Now apply hybrid phaseless approximation
        walker.inverse_overlap(trial.psi)
        ot_new = walker.calc_otrial(trial.psi)
        # Walker's phase.
        importance_function = self.mf_const_fac*cxf*cfb*ot_new / walker.ot
        dtheta = cmath.phase(importance_function)
        cfac = max(0, math.cos(dtheta))
        rweight = abs(importance_function)
        walker.weight *= rweight * cfac
        walker.ot = ot_new
        walker.field_configs.push_full(xmxbar, cfac, importance_function/rweight)

def construct_propagator_matrix_generic(system, BT2, config, dt, conjt=False):
    """Construct the full projector from a configuration of auxiliary fields.

    For use with generic system object.

    Parameters
    ----------
    system : class
        System class.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    config : numpy array
        Auxiliary field configuration.
    conjt : bool
        If true return Hermitian conjugate of matrix.

    Returns
    -------
    B : :class:`numpy.ndarray`
        Full projector matrix.
    """
    VHS = 1j*dt**0.5*numpy.einsum('l,lpq->pq', config, system.chol_vecs)
    EXP_VHS = exponentiate_matrix(VHS)
    Bup = BT2[0].dot(EXP_VHS).dot(BT2[0])
    Bdown = BT2[1].dot(EXP_VHS).dot(BT2[1])

    if conjt:
        return [Bup.conj().T, Bdown.conj().T]
    else:
        return [Bup, Bdown]


def back_propagate(system, psi, trial, nstblz, BT2, dt):
    r"""Perform back propagation for UHF style wavefunction.

    For use with generic system hamiltonian.

    Parameters
    ---------
    system : system object in general.
        Container for model input options.
    psi : :class:`pauxy.walkers.Walkers` object
        CPMC wavefunction.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    nstblz : int
        Number of steps between GS orthogonalisation.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    dt : float
        Timestep.

    Returns
    -------
    psi_bp : list of :class:`pauxy.walker.Walker` objects
        Back propagated list of walkers.
    """
    psi_bp = [SingleDetWalker(1,system,trial,w) for w in range(len(psi))]
    nup = system.nup
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, c) in enumerate(w.field_configs.get_block()[0][::-1]):
            # could make this system specific to reduce need for multiple
            # routines.
            B = construct_propagator_matrix_generic(system, BT2,
                                                    c, dt, conjt=True)
            psi_bp[iw].phi[:, :nup] = B[0].dot(psi_bp[iw].phi[:, :nup])
            psi_bp[iw].phi[:, nup:] = B[1].dot(psi_bp[iw].phi[:, nup:])
            if i != 0 and i % nstblz == 0:
                psi_bp[iw].reortho(trial)
    return psi_bp
