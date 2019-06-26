import cmath
import copy
import numpy
import math
import scipy.linalg
from pauxy.propagation.operations import kinetic_real, local_energy_bound
from pauxy.utils.fft import fft_wavefunction, ifft_wavefunction
from pauxy.utils.linalg import reortho
from pauxy.walkers.multi_ghf import MultiGHFWalker
from pauxy.walkers.single_det import SingleDetWalker

class HirschSpin(object):
    """Propagator for discrete HS transformation.

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

    def __init__(self, system, trial, qmc, options={}, verbose=False):

        if verbose:
            print ("# Parsing discrete propagator input options.")
        if trial.type == 'GHF':
            self.bt2 = scipy.linalg.expm(-0.5*qmc.dt*system.T[0])
        else:
            self.bt2 = numpy.array([scipy.linalg.expm(-0.5*qmc.dt*system.T[0]),
                                    scipy.linalg.expm(-0.5*qmc.dt*system.T[1])])
        if trial.type == 'GHF' and trial.bp_wfn is not None:
            self.BT_BP = scipy.linalg.block_diag(self.bt2, self.bt2)
            self.back_propagate = back_propagate_ghf
        else:
            self.BT_BP = self.bt2
            self.back_propagate = back_propagate
        self.nstblz = qmc.nstblz
        self.btk = numpy.exp(-0.5*qmc.dt*system.eks)
        self.hs_type = 'discrete'
        self.free_projection = options.get('free_projection', False)
        self.gamma = numpy.arccosh(numpy.exp(0.5*qmc.dt*system.U))
        self.auxf = numpy.array([[numpy.exp(self.gamma), numpy.exp(-self.gamma)],
                                [numpy.exp(-self.gamma), numpy.exp(self.gamma)]])
        self.auxf = self.auxf * numpy.exp(-0.5*qmc.dt*system.U)
        self.delta = self.auxf - 1
        if self.free_projection:
            self.propagate_walker = self.propagate_walker_free
        else:
            self.propagate_walker = self.propagate_walker_constrained
        if trial.name == 'multi_determinant':
            if trial.type == 'GHF':
                self.calculate_overlap_ratio = calculate_overlap_ratio_multi_ghf
                self.kinetic = kinetic_ghf
                self.update_greens_function = self.update_greens_function_ghf
            else:
                self.calculate_overlap_ratio = calculate_overlap_ratio_multi_det
                self.kinetic = kinetic_real
        else:
            self.calculate_overlap_ratio = calculate_overlap_ratio_single_det
            self.update_greens_function = self.update_greens_function_uhf
            if qmc.ffts:
                self.kinetic = kinetic_kspace
            else:
                self.kinetic = kinetic_real
        if verbose:
            print ("# Finished setting up propagator.")

    def update_greens_function_uhf(self, walker, trial, i, nup):
        """Fast update of walker's Green's function for RHF/UHF walker.

        Parameters
        ----------
        walker : :class:`pauxy.walkers.SingleDet`
            Walker's wavefunction.
        trial : :class:`pauxy.trial_wavefunction`
            Trial wavefunction.
        i : int
            Basis index.
        nup : int
            Number of up electrons.
        """
        vup = trial.psi.conj()[i,:nup]
        uup = walker.phi[i,:nup]
        q = numpy.dot(walker.inv_ovlp[0], vup)
        walker.G[0][i,i] = numpy.dot(uup, q)
        vdown = trial.psi.conj()[i,nup:]
        udown = walker.phi[i,nup:]
        q = numpy.dot(walker.inv_ovlp[1], vdown)
        walker.G[1][i,i] = numpy.dot(udown, q)

    def update_greens_function_ghf(self, walker, trial, i, nup):
        """Update of walker's Green's function for UHF walker.

        Parameters
        ----------
        walker : :class:`pauxy.walkers.SingleDet`
            Walker's wavefunction.
        trial : :class:`pauxy.trial_wavefunction`
            Trial wavefunction.
        i : int
            Basis index.
        nup : int
            Number of up electrons.
        """
        walker.greens_function(trial)

    def kinetic_importance_sampling(self, walker, system, trial):
        r"""Propagate by the kinetic term by direct matrix multiplication.

        Parameters
        ----------
        walker : :class:`pauxy.walker`
            Walker object to be updated. On output we have acted on phi by
            B_{T/2} and updated the weight appropriately. Updates inplace.
        system : :class:`pauxy.system.System`
            System object.
        trial : :class:`pauxy.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """
        self.kinetic(walker.phi, system, self.bt2)
        # Update inverse overlap
        walker.inverse_overlap(trial)
        # Update walker weight
        ot_new = walker.calc_otrial(trial)
        ratio = (ot_new/walker.ot)
        phase = cmath.phase(ratio)
        if abs(phase) < 0.5*math.pi:
            walker.weight = walker.weight * ratio.real
            walker.ot = ot_new
        else:
            walker.weight = 0.0

    def two_body(self, walker, system, trial):
        r"""Propagate by potential term using discrete HS transform.

        Parameters
        ----------
        walker : :class:`pauxy.walker` object
            Walker object to be updated. On output we have acted on phi by
            B_V(x) and updated the weight appropriately. Updates inplace.
        system : :class:`pauxy.system.System`
            System object.
        trial : :class:`pauxy.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """
        # Construct random auxilliary field.
        delta = self.delta
        nup = system.nup
        soffset = walker.phi.shape[0] - system.nbasis
        for i in range(0, system.nbasis):
            self.update_greens_function(walker, trial, i, nup)
            # Ratio of determinants for the two choices of auxilliary fields
            probs = self.calculate_overlap_ratio(walker, delta, trial, i)
            # issues here with complex numbers?
            phaseless_ratio = numpy.maximum(probs.real, [0,0])
            norm = sum(phaseless_ratio)
            r = numpy.random.random()
            # Is this necessary?
            # todo : mirror correction
            if norm > 0:
                walker.weight = walker.weight * norm
                if r < phaseless_ratio[0]/norm:
                    xi = 0
                else:
                    xi = 1
                vtup = walker.phi[i,:nup] * delta[xi, 0]
                vtdown = walker.phi[i+soffset,nup:] * delta[xi, 1]
                walker.phi[i,:nup] = walker.phi[i,:nup] + vtup
                walker.phi[i+soffset,nup:] = walker.phi[i+soffset,nup:] + vtdown
                walker.update_overlap(probs, xi, trial.coeffs)
                walker.field_configs.push(xi)
                walker.update_inverse_overlap(trial, vtup, vtdown, i)
            else:
                walker.weight = 0
                return

    def propagate_walker_constrained(self, walker, system, trial):
        r"""Wrapper function for propagation using discrete transformation

        The discrete transformation allows us to split the application of the
        projector up a bit more, which allows up to make use of fast matrix
        update routines since only a row might change.

        Parameters
        ----------
        walker : :class:`pauxy.walker` object
            Walker object to be updated. On output we have acted on phi by
            B_V(x) and updated the weight appropriately. Updates inplace.
        system : :class:`pauxy.system.System`
            System object.
        trial : :class:`pauxy.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """

        if abs(walker.weight) > 0:
            self.kinetic_importance_sampling(walker, system, trial)
        if abs(walker.weight) > 0:
            self.two_body(walker, system, trial)
        if abs(walker.weight.real) > 0:
            self.kinetic_importance_sampling(walker, system, trial)

    def propagate_walker_free(self, walker, system, trial):
        r"""Propagate walker without imposing constraint.

        Uses single-site updates for potential term.

        Parameters
        ----------
        walker : :class:`pauxy.walker` object
            Walker object to be updated. On output we have acted on phi by
            B_V(x) and updated the weight appropriately. Updates inplace.
        system : :class:`pauxy.system.System`
            System object.
        trial : :class:`pauxy.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """
        kinetic_real(walker.phi, system, self.bt2)
        delta = self.delta
        nup = system.nup
        for i in range(0, system.nbasis):
            if abs(walker.weight) > 0:
                r = numpy.random.random()
                if r < 0.5:
                    xi = 0
                else:
                    xi = 1
                vtup = walker.phi[i,:nup] * delta[xi, 0]
                vtdown = walker.phi[i,nup:] * delta[xi, 1]
                walker.phi[i,:nup] = walker.phi[i,:nup] + vtup
                walker.phi[i,nup:] = walker.phi[i,nup:] + vtdown
        kinetic_real(walker.phi, system, self.bt2)
        walker.inverse_overlap(trial)
        # Update walker weight
        walker.ot = walker.calc_otrial(trial.psi)

# todo: stucture is the same for all continuous HS transformations.
class HubbardContinuous(object):
    """Propagator for continuous HS transformation, specialised for Hubbard model.

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
        self.hs_type = 'hubbard_continuous'
        self.free_projection = options.get('free_projection', False)
        self.bt2 = numpy.array([scipy.linalg.expm(-0.5*qmc.dt*system.T[0]),
                                scipy.linalg.expm(-0.5*qmc.dt*system.T[1])])
        self.BT_BP = self.bt2
        self.back_propagate = back_propagate
        self.nstblz = qmc.nstblz
        self.btk = numpy.exp(-0.5*qmc.dt*system.eks)
        model = system.__class__.__name__
        self.dt = qmc.dt
        # optimal mean-field shift for the hubbard model
        self.mf_shift = (system.nup+system.ndown) / float(system.nbasis)
        self.iut_fac = 1j*numpy.sqrt((system.U*self.dt))
        self.ut_fac = self.dt*system.U
        # Include factor of M! bad name
        self.mf_nsq = system.nbasis * self.mf_shift**2.0
        self.ebound = (2.0/self.dt)**0.5
        self.mean_local_energy = 0
        if self.free_projection:
            self.propagate_walker = self.propagate_walker_free_continuous
        else:
            self.propagate_walker = self.propagate_walker_constrained_continuous
        if qmc.ffts:
            self.kinetic = kinetic_kspace
        else:
            self.kinetic = kinetic_real
        if verbose:
            print ("# Finished propagator input options.")

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

        mf = self.mf_shift
        ifac = self.iut_fac
        ufac = self.ut_fac
        nsq = self.mf_nsq
        # Normally distrubted auxiliary fields.
        xi = numpy.random.normal(0.0, 1.0, system.nbasis)
        # Optimal field shift for real local energy approximation.
        shift = numpy.diag(walker.G[0])+numpy.diag(walker.G[1]) - mf
        xi_opt = -ifac*shift
        sxf = sum(xi-xi_opt)
        # Propagator for potential term with mean field and auxilary field shift.
        c_xf = cmath.exp(0.5*ufac*nsq-ifac*mf*sxf)
        EXP_VHS = numpy.exp(0.5*ufac*(1-2.0*mf)+ifac*(xi-xi_opt))
        nup = system.nup
        walker.phi[:,:nup] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[:,:nup])
        walker.phi[:,nup:] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[:,nup:])
        return c_xf

    def propagate_walker_free_continuous(self, walker, system, trial):
        r"""Free projection for continuous HS transformation.

        TODO: update if ever adapted to other model types.

        Parameters
        ----------
        walker : :class:`pauxy.walker` object
            Walker object to be updated. On output we have acted on phi by
            B_V(x) and updated the weight appropriately. Updates inplace.
        system : :class:`pauxy.system.System`
            System object.
        trial : :class:`pauxy.trial_wavefunctioin.Trial`
            Trial wavefunction object.
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
        (magn, dtheta) = cmath.polar(c_xf)
        walker.weight *= magn
        walker.phase *= cmath.exp(1j*dtheta)

    def propagate_walker_constrained_continuous(self, walker, system, trial):
        r"""Wrapper function for propagation using continuous transformation.

        This applied the phaseless, local energy approximation and uses importance
        sampling.

        Parameters
        ----------
        walker : :class:`pauxy.walker` object
            Walker object to be updated. On output we have acted on phi by
            B_V(x) and updated the weight appropriately. Updates inplace.
        system : :class:`pauxy.system.System`
            System object.
        trial : :class:`pauxy.trial_wavefunction.Trial`
            Trial wavefunction object.
        """

        # 1. Apply kinetic projector.
        self.kinetic(walker.phi, system, self.bt2)
        # 2. Apply potential projector.
        cxf = self.two_body(walker, system, trial)
        # 3. Apply kinetic projector.
        self.kinetic(walker.phi, system, self.bt2)

        # Now apply phaseless, real local energy approximation
        walker.inverse_overlap(trial.psi)
        walker.greens_function(trial)
        E_L = walker.local_energy(system)[0].real
        # Check for large population fluctuations
        E_L = local_energy_bound(E_L, self.mean_local_energy,
                                 self.ebound)
        ot_new = walker.calc_otrial(trial.psi)
        # Walker's phase.
        dtheta = cmath.phase(cxf*ot_new/walker.ot)
        walker.weight = (walker.weight * math.exp(-0.5*self.dt*(walker.E_L+E_L))
                                       * max(0, math.cos(dtheta)))
        walker.E_L = E_L
        walker.ot = ot_new


def calculate_overlap_ratio_multi_ghf(walker, delta, trial, i):
    """Calculate overlap ratio for single site update with GHF trial.

    Parameters
    ----------
    walker : walker object
        Walker to be updated.
    delta : :class:`numpy.ndarray`
        Delta updates for single spin flip.
    trial : trial wavefunctio object
        Trial wavefunction.
    i : int
        Basis index.
    """
    nbasis = trial.psi.shape[1] // 2
    for (idx, G) in enumerate(walker.Gi):
        guu = G[i,i]
        gdd = G[i+nbasis,i+nbasis]
        gud = G[i,i+nbasis]
        gdu = G[i+nbasis,i]
        walker.R[idx,0] = (
            (1+delta[0,0]*guu)*(1+delta[0,1]*gdd) - delta[0,0]*gud*delta[0,1]*gdu
        )
        walker.R[idx,1] = (
            (1+delta[1,0]*guu)*(1+delta[1,1]*gdd) - delta[1,0]*gud*delta[1,1]*gdu
        )
    R = numpy.einsum('i,ij,i->j',trial.coeffs,walker.R,walker.ots)/walker.ot
    return 0.5 * numpy.array([R[0],R[1]])

def calculate_overlap_ratio_multi_det(walker, delta, trial, i):
    """Calculate overlap ratio for single site update with multi-det trial.

    Parameters
    ----------
    walker : walker object
        Walker to be updated.
    delta : :class:`numpy.ndarray`
        Delta updates for single spin flip.
    trial : trial wavefunctio object
        Trial wavefunction.
    i : int
        Basis index.
    """
    for (idx, G) in enumerate(walker.Gi):
        walker.R[idx,0,0] = (1+delta[0][0]*G[0][i,i])
        walker.R[idx,0,1] = (1+delta[0][1]*G[1][i,i])
        walker.R[idx,1,0] = (1+delta[1][0]*G[0][i,i])
        walker.R[idx,1,1] = (1+delta[1][1]*G[1][i,i])
    spin_prod = numpy.einsum('ikj,ji->ikj',walker.R,walker.ots)
    R = numpy.einsum('i,ij->j',trial.coeffs,spin_prod[:,:,0]*spin_prod[:,:,1])/walker.ot
    return 0.5 * numpy.array([R[0],R[1]])

def calculate_overlap_ratio_single_det(walker, delta, trial, i):
    """Calculate overlap ratio for single site update with UHF trial.

    Parameters
    ----------
    walker : walker object
        Walker to be updated.
    delta : :class:`numpy.ndarray`
        Delta updates for single spin flip.
    trial : trial wavefunctio object
        Trial wavefunction.
    i : int
        Basis index.
    """
    R1 = (1+delta[0][0]*walker.G[0][i,i])*(1+delta[0][1]*walker.G[1][i,i])
    R2 = (1+delta[1][0]*walker.G[0][i,i])*(1+delta[1][1]*walker.G[1][i,i])
    return 0.5 * numpy.array([R1,R2])

def construct_propagator_matrix(system, BT2, config, conjt=False):
    """Construct the full projector from a configuration of auxiliary fields.

    For use with discrete transformation.

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
    bv_up = numpy.diag(numpy.array([system.auxf[xi, 0] for xi in config]))
    bv_down = numpy.diag(numpy.array([system.auxf[xi, 1] for xi in config]))
    Bup = BT2[0].dot(bv_up).dot(BT2[0])
    Bdown = BT2[1].dot(bv_down).dot(BT2[1])

    if conjt:
        return numpy.array([Bup.conj().T, Bdown.conj().T])
    else:
        return numpy.array([Bup, Bdown])


def construct_propagator_matrix_ghf(system, BT2, config, conjt=False):
    """Construct the full projector from a configuration of auxiliary fields.

    For use with GHF trial wavefunction.

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
    bv_up = numpy.diag(numpy.array([system.auxf[xi, 0] for xi in config]))
    bv_down = numpy.diag(numpy.array([system.auxf[xi, 1] for xi in config]))
    BV = scipy.linalg.block_diag(bv_up, bv_down)
    B = BT2.dot(BV).dot(BT2)

    if conjt:
        return B.conj().T
    else:
        return B

def back_propagate(system, psi, trial, nstblz, BT2, dt):
    r"""Perform back propagation for UHF style wavefunction.

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
            B = construct_propagator_matrix(system, BT2,
                                            c, conjt=True)
            psi_bp[iw].phi[:,:nup] = B[0].dot(psi_bp[iw].phi[:,:nup])
            psi_bp[iw].phi[:,nup:] = B[1].dot(psi_bp[iw].phi[:,nup:])
            if i != 0 and i % nstblz == 0:
                psi_bp[iw].reortho(trial)
    return psi_bp

def back_propagate_ghf(system, psi, trial, nstblz, BT2, dt):
    r"""Perform back propagation for GHF style wavefunction.

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
    psi_bp = [MultiGHFWalker({}, system, trial, index=w, weights='ones', wfn0='GHF')
              for w in range(len(psi))]
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, c) in enumerate(w.field_configs.get_block()[0][::-1]):
            B = construct_propagator_matrix_ghf(system, BT2,
                                                c, conjt=True)
            for (idet, psi_i) in enumerate(psi_bp[iw].phi):
                # propagate each component of multi-determinant expansion
                psi_bp[iw].phi[idet] = B.dot(psi_bp[iw].phi[idet])
                if i != 0 and i % nstblz == 0:
                    # implicitly propagating the full GHF wavefunction
                    (psi_bp[iw].phi[idet], detR) = reortho(psi_i)
                    psi_bp[iw].weights[idet] *= detR.conjugate()
    return psi_bp


def back_propagate_single(phi_in, configs, weights,
                          system, nstblz, BT2, store=False):
    r"""Perform back propagation for single walker.

    Parameters
    ---------
    phi_in : :class:`pauxy.walkers.Walker` object
        Walker.
    configs : :class:`numpy.ndarray`
        Auxilliary field configurations.
    weights : :class:`numpy.ndarray`
        Not used. For interface consistency.
    system : system object in general.
        Container for model input options.
    nstblz : int
        Number of steps between GS orthogonalisation.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    store : bool
        If true the the back propagated wavefunctions are stored along the back
        propagation path.

    Returns
    -------
    psi_store : list of :class:`pauxy.walker.Walker` objects
        Back propagated list of walkers.
    """
    nup = system.nup
    psi_store = []
    for (i, c) in enumerate(configs[::-1]):
        B = construct_propagator_matrix(system, BT2, c, conjt=True)
        phi_in[:,:nup] = B[0].dot(phi_in[:,:nup])
        phi_in[:,nup:] = B[1].dot(phi_in[:,nup:])
        if i != 0 and i % nstblz == 0:
            (phi_in[:,:nup], R) = reortho(phi_in[:,:nup])
            (phi_in[:,nup:], R) = reortho(phi_in[:,nup:])
        if store:
            psi_store.append(copy.deepcopy(phi_in))

    return psi_store


def back_propagate_single_ghf(phi, configs, weights, system,
                              nstblz, BT2, store=False):
    r"""Perform back propagation for single walker.

    Parameters
    ---------
    phi : :class:`pauxy.walkers.MultiGHFWalker` object
        Walker.
    configs : :class:`numpy.ndarray`
        Auxilliary field configurations.
    weights : :class:`numpy.ndarray`
        Not used. For interface consistency.
    system : system object in general.
        Container for model input options.
    nstblz : int
        Number of steps between GS orthogonalisation.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    store : bool
        If true the the back propagated wavefunctions are stored along the back
        propagation path.

    Returns
    -------
    psi_store : list of :class:`pauxy.walker.Walker` objects
        Back propagated list of walkers.
    """
    nup = system.nup
    psi_store = []
    for (i, c) in enumerate(configs[::-1]):
        B = construct_propagator_matrix_ghf(system, BT2, c, conjt=True)
        for (idet, psi_i) in enumerate(phi):
            # propagate each component of multi-determinant expansion
            phi[idet] = B.dot(phi[idet])
            if i != 0 and i % nstblz == 0:
                # implicitly propagating the full GHF wavefunction
                (phi[idet], detR) = reortho(psi_i)
                weights[idet] *= detR.conjugate()
        if store:
            psi_store.append(copy.deepcopy(phi))

    return psi_store


def kinetic_kspace(phi, system, btk):
    """Apply the kinetic energy projector in kspace.

    May be faster for very large dilute lattices.

    Parameters
    ---------
    phi : :class:`pauxy.walkers.MultiGHFWalker` object
        Walker.
    system : system object in general.
        Container for model input options.
    B : :class:`numpy.ndarray`
        One body propagator.
    """
    s = system
    # Transform psi to kspace by fft-ing its columns.
    tup = fft_wavefunction(phi[:,:s.nup], s.nx, s.ny,
                           s.nup, phi[:,:s.nup].shape)
    tdown = fft_wavefunction(phi[:,s.nup:], s.nx, s.ny,
                             s.ndown, phi[:,s.nup:].shape)
    # Kinetic enery operator is diagonal in momentum space.
    # Note that multiplying by diagonal btk in this way is faster than using
    # einsum and way faster than using dot using an actual diagonal matrix.
    tup = (btk*tup.T).T
    tdown = (btk*tdown.T).T
    # Transform phi to kspace by fft-ing its columns.
    tup = ifft_wavefunction(tup, s.nx, s.ny, s.nup, tup.shape)
    tdown = ifft_wavefunction(tdown, s.nx, s.ny, s.ndown, tdown.shape)
    if phi.dtype == float:
        phi[:,:s.nup] = tup.astype(float)
        phi[:,s.nup:] = tdown.astype(float)
    else:
        phi[:,:s.nup] = tup
        phi[:,s.nup:] = tdown
