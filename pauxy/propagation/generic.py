import cmath
import math
import numpy
import scipy.linalg
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
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j*self.sqrt_dt
        # Mean field shifts (2,nchol_vec).
        self.mf_shift = 1j*numpy.einsum('lpq,spq->l', system.chol_vecs, trial.G)
        # Mean field shifted one-body propagator
        self.construct_one_body_propagator(system, qmc.dt)
        # Constant core contribution modified by mean field shift.
        self.mf_core = system.ecore + 0.5*numpy.dot(self.mf_shift, self.mf_shift)
        self.nstblz = qmc.nstblz
        self.vbias = numpy.zeros(system.nfields, dtype=numpy.complex128)
        # Half rotated cholesky vectors (by trial wavefunction).
        # Assuming nup = ndown here
        rotated_up = numpy.einsum('rp,lpq->lrq',
                                  trial.psi[:,:system.nup].conj().T,
                                  system.chol_vecs)
        rotated_down = numpy.einsum('rp,lpq->lrq',
                                    trial.psi[:,system.nup:].conj().T,
                                    system.chol_vecs)
        self.rchol_vecs = numpy.array([rotated_up, rotated_down])
        self.chol_vecs = system.chol_vecs
        self.ebound = (2.0/self.dt)**0.5
        self.mean_local_energy = 0
        if verbose:
            print ("# Finished setting up Generic propagator.")


    def construct_one_body_propagator(self, system, dt):
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
        shift = 1j*numpy.einsum('l,lpq->pq', self.mf_shift, system.chol_vecs)
        H1 = system.h1e_mod - numpy.array([shift,shift])
        self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]),
                                scipy.linalg.expm(-0.5*dt*H1[1])])

    def construct_force_bias(self, system, walker, trial):
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
        # Construct walker modified Green's function.
        walker.inverse_overlap(trial.psi)
        walker.rotated_greens_function()
        vbias = 1j*numpy.einsum('slrp,spr->l', self.rchol_vecs, walker.Gmod)
        return - self.sqrt_dt * (vbias-self.mf_shift)

    def construct_force_bias_full(self, system, walker, trial):
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

    def construct_VHS(self, system, shifted):
        return self.isqrt_dt * numpy.einsum('l,lpq->pq', shifted,
                                            system.chol_vecs)

    def construct_VHS_incore(self, system, shifted):
        pass

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
        Full propagator matrix.
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
    r"""Perform back propagation for RHF/UHF style wavefunction.

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
    psi_bp = [SingleDetWalker({}, system, trial, index=w) for w in range(len(psi))]
    nup = system.nup
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, c) in enumerate(w.field_configs.get_block()[0][::-1]):
            # could make this system specific to reduce need for multiple
            # routines.
            B = construct_propagator_matrix_generic(system, BT2,
                                                    c, dt, conjt=True)
            psi_bp[iw].phi[:,:nup] = B[0].dot(psi_bp[iw].phi[:,:nup])
            psi_bp[iw].phi[:,nup:] = B[1].dot(psi_bp[iw].phi[:,nup:])
            if i != 0 and i % nstblz == 0:
                psi_bp[iw].reortho(trial)
    return psi_bp
