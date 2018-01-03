"""Routines for performing propagation of a walker"""

# TODO: refactor to avoid code repetition between similar routines.

import numpy
import scipy.linalg
import math
import cmath
import copy
import afqmcpy.utils
import afqmcpy.walker as walker


def propagate_walker_discrete(walker, state):
    r"""Wrapper function for propagation using discrete transformation

    The discrete transformation allows us to split the application of the
    projector up a bit more, which allows up to make use of fast matrix update
    routines since only a row might change.

    Parameters
    ----------
    walker : :class:`walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B` and updated the weight
        appropriately. Updates inplace.
    state : :class:`state.State`
        Simulation state.
    """

    if abs(walker.weight) > 0:
        kinetic_importance_sampling(walker, state)
    if abs(walker.weight) > 0:
        state.propagators.potential(walker, state)
    if abs(walker.weight.real) > 0:
        kinetic_importance_sampling(walker, state)


def propagate_walker_discrete_multi_site(walker, state):
    r"""Wrapper function for propagation using discrete transformation

    The discrete transformation allows us to split the application of the
    projector up a bit more, which allows up to make use of fast matrix update
    routines since only a row might change.

    Parameters
    ----------
    walker : :class:`walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B` and updated the weight
        appropriately. Updates inplace.
    state : :class:`state.State`
        Simulation state.
    """

    # 1. Apply kinetic projector.
    state.propagators.kinetic(walker.phi, state)
    # 2. Apply potential projector.
    propagate_potential_auxf(walker, state)
    # 3. Apply kinetic projector.
    state.propagators.kinetic(walker.phi, state)
    walker.inverse_overlap(state.trial.psi, state.system.nup)
    # Calculate new total overlap and update components of overlap
    ot_new = walker.calc_otrial(state.trial.psi)
    # Now apply phaseless approximation
    dtheta = cmath.phase(ot_new/walker.ot)
    walker.weight = walker.weight * max(0, math.cos(dtheta))
    walker.ot = ot_new


def propagate_walker_free(walker, state):
    r"""Propagate walker without imposing constraint.

    Uses single-site updates for potential term.

    Parameters
    ----------
    walker : :class:`walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B` and updated the weight
        appropriately. Updates inplace.
    state : :class:`state.State`
        Simulation state.
"""
    kinetic_real(walker.phi, state)
    delta = state.system.auxf - 1
    nup = state.system.nup
    for i in range(0, state.system.nbasis):
        if abs(walker.weight) > 0:
            r = numpy.random.random()
            # TODO: remove code repition.
            if r > 0.5:
                vtup = walker.phi[i,:nup] * delta[0, 0]
                vtdown = walker.phi[i,nup:] * delta[0, 1]
                walker.phi[i,:nup] = walker.phi[i,:nup] + vtup
                walker.phi[i,nup:] = walker.phi[i,nup:] + vtdown
            else:
                vtup = walker.phi[i,:nup] * delta[1,0]
                vtdown = walker.phi[i,nup:] * delta[1, 1]
                walker.phi[i,:nup] = walker.phi[i,:nup] + vtup
                walker.phi[i,nup:] = walker.phi[i,nup:] + vtdown
    kinetic_real(walker.phi, state)
    walker.inverse_overlap(state.trial.psi, nup)
    # Update walker weight
    walker.ot = walker.calc_otrial(state.trial.psi)
    walker.greens_function(state.trial, nup)


def propagate_walker_free_continuous(walker, state):
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
    nup = state.system.nup
    # 1. Apply kinetic projector.
    kinetic_real(walker.phi, state)
    # Normally distributed random numbers.
    xfields =  numpy.random.normal(0.0, 1.0, state.system.nbasis)
    sxf = sum(xfields)
    # Constant, field dependent term emerging when subtracting mean-field.
    sc = 0.5*state.qmc.ut_fac*state.qmc.mf_nsq-state.qmc.iut_fac*state.qmc.mf_shift*sxf
    c_xf = cmath.exp(sc)
    # Potential propagator.
    s = state.qmc.iut_fac*xfields + 0.5*state.qmc.ut_fac*(1-2*state.qmc.mf_shift)
    bv = numpy.diag(numpy.exp(s))
    # 2. Apply potential projector.
    walker.phi[:,:nup] = bv.dot(walker.phi[:,:nup])
    walker.phi[:,nup:] = bv.dot(walker.phi[:,nup:])
    # 3. Apply kinetic projector.
    kinetic_real(walker.phi, state)
    walker.inverse_overlap(state.trial.psi, nup)
    walker.ot = walker.calc_otrial(state.trial.psi)
    walker.greens_function(state.trial, nup)
    # Constant terms are included in the walker's weight.
    walker.weight = walker.weight * c_xf


def propagate_walker_continuous(walker, state):
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

    # 1. Apply kinetic projector.
    state.propagators.kinetic(walker.phi, state)
    # 2. Apply potential projector.
    cxf = state.propagators.potential(walker, state)
    # 3. Apply kinetic projector.
    state.propagators.kinetic(walker.phi, state)

    # Now apply phaseless, real local energy approximation
    walker.inverse_overlap(state.trial.psi, state.system.nup)
    walker.greens_function(state.trial, state.system.nup)
    E_L = walker.local_energy(state.system)[0].real
    # Check for large population fluctuations
    E_L = local_energy_bound(E_L, state.qmc.mean_local_energy,
                             state.qmc.local_energy_bound)
    ot_new = walker.calc_otrial(state.trial.psi)
    # Walker's phase.
    dtheta = cmath.phase(cxf*ot_new/walker.ot)
    walker.weight = (walker.weight * math.exp(-0.5*state.qmc.dt*(walker.E_L+E_L))
                                   * max(0, math.cos(dtheta)))
    walker.E_L = E_L
    walker.ot = ot_new


def local_energy_bound(local_energy, mean, threshold):
    """Try to suppress rare population events by imposing local energy bound.

    See: Purwanto et al., Phys. Rev. B 80, 214116 (2009).

Parameters
----------
local_energy : float
    Local energy of current walker
mean : float
    Mean value of local energy about which we impose the threshold / bound.
threshold : float
    Amount of lee-way for energy fluctuations about the mean.
"""

    maximum = mean + threshold
    minimum = mean - threshold

    if (local_energy >= maximum):
        local_energy = maximum
    elif (local_energy < minimum):
        local_energy = minimum
    else:
        local_energy = local_energy

    return local_energy

def calculate_overlap_ratio_multi_ghf(walker, delta, trial, i):
    nbasis = trial.psi.shape[1]//2
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
    for (idx, G) in enumerate(walker.Gi):
        walker.R[idx,0,0] = (1+delta[0][0]*G[0][i,i])
        walker.R[idx,0,1] = (1+delta[0][1]*G[1][i,i])
        walker.R[idx,1,0] = (1+delta[1][0]*G[0][i,i])
        walker.R[idx,1,1] = (1+delta[1][1]*G[1][i,i])
    spin_prod = numpy.einsum('ikj,ji->ikj',walker.R,walker.ots)
    R = numpy.einsum('i,ij->j',trial.coeffs,spin_prod[:,:,0]*spin_prod[:,:,1])/walker.ot
    return 0.5 * numpy.array([R[0],R[1]])

def calculate_overlap_ratio_single_det(walker, delta, trial, i):
    R1 = (1+delta[0][0]*walker.G[0][i,i])*(1+delta[0][1]*walker.G[1][i,i])
    R2 = (1+delta[1][0]*walker.G[0][i,i])*(1+delta[1][1]*walker.G[1][i,i])
    return 0.5 * numpy.array([R1,R2])

def discrete_hubbard(walker, state):
    r"""Propagate by potential term using discrete HS transform.

    Parameters
    ----------
    walker : :class:`afqmcpy.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`b_V` and updated the weight appropriately.
        updates inplace.
    state : :class:`afqmcpy.state.State`
        Simulation state.
    """
    # Construct random auxilliary field.
    delta = state.system.auxf - 1
    nup = state.system.nup
    soffset = walker.phi.shape[0] - state.system.nbasis
    for i in range(0, state.system.nbasis):
        # Ratio of determinants for the two choices of auxilliary fields
        probs = state.propagators.calculate_overlap_ratio(walker, delta,
                                                          state.trial, i)
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
            walker.update_overlap(probs, xi, state.trial.coeffs)
            walker.field_config[i] = xi
            walker.update_inverse_overlap(state.trial, vtup, vtdown, nup, i)
            walker.greens_function(state.trial, nup)
        else:
            walker.weight = 0
            return

def dumb_hubbard(walker, state):
    r"""Continuous Hubbard-Statonovich transformation for Hubbard model.

    Only requires M auxiliary fields.

    Parameters
    ----------
    walker : :class:`afqmcpy.walker.Walker`
        walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`b_v` and updated the weight appropriately.
        updates inplace.
    state : :class:`afqmcpy.state.State`
        Simulation state.
    """

    mf = state.qmc.mf_shift
    ifac = state.qmc.iut_fac
    ufac = state.qmc.ut_fac
    nsq = state.qmc.mf_nsq
    # Normally distrubted auxiliary fields.
    xi = numpy.random.normal(0.0, 1.0, state.system.nbasis)
    # Optimal field shift for real local energy approximation.
    shift = numpy.diag(walker.G[0])+numpy.diag(walker.G[1]) - mf
    xi_opt = -ifac*shift
    sxf = sum(xi-xi_opt)
    # Propagator for potential term with mean field and auxilary field shift.
    c_xf = cmath.exp(0.5*ufac*nsq-ifac*mf*sxf)
    EXP_VHS = numpy.exp(0.5*ufac*(1-2.0*mf)+ifac*(xi-xi_opt))
    nup = state.system.nup
    walker.phi[:,:nup] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[:,:nup])
    walker.phi[:,nup:] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[:,nup:])
    return c_xf


def generic_continuous(walker, state):
    r"""Continuous HS transformation

    This form assumes nothing about the form of the two-body Hamiltonian and
    is thus quite slow, particularly if the matrix is M^2xM^2.

    Todo: check if this actually works.

    Parameters
    ----------
    walker : :class:`afqmcpy.walker.Walker`
        walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`b_v` and updated the weight appropriately.
        updates inplace.
    state : :class:`afqmcpy.state.State`
        Simulation state.
    """

    # iterate over spins

    dt = state.dt
    gamma = state.system.gamma
    nup = state.system.nup
    # Generate ~M^2 normally distributed auxiliary fields.
    sigma = dt**0.5 * numpy.random.normal(0.0, 1.0, len(gamma))
    # Construct HS potential, V_HS = sigma dot U
    V_HS = numpy.einsum('ij,j->i', sigma, gamma)
    # Reshape so we can apply to MxN Slater determinant.
    V_HS = numpy.reshape(V_HS, (M,M))
    for n in range(1, nmax_exp+1):
        EXP_V = EXP_V + numpy.dot(V_HS, EXP_V)/numpy.factorial(n)
    walker.phi[:nup] = numpy.dot(EXP_V, walker.phi[:nup])
    walker.phi[nup:] = numpy.dot(EXP_V, walker.phi[:nup])

    # Update inverse and green's function
    walker.inverse_overlap(trial)
    walker.greens_function(trial)
    # Perform importance sampling, phaseless and real local energy approximation and update
    E_L = state.estimators.local_energy(system, walker.G)[0].real
    ot_new = walker.calc_otrial(trial)
    dtheta = cmath.phase(ot_new/walker.ot)
    walker.weight = (walker.weight * exp(-0.5*system.dt*(walker.E_L-E_L))
                                  * max(0, cos(dtheta)))
    walker.E_L = E_L
    walker.ot = ot_new


def kinetic_importance_sampling(walker, state):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    Parameters
    ----------
    walker : :class:`afqmcpy.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`afqmcpy.state.State`
        Simulation state.
    """
    state.propagators.kinetic(walker.phi, state)
    # Update inverse overlap
    walker.inverse_overlap(state.trial.psi, state.system.nup)
    # Update walker weight
    ot_new = walker.calc_otrial(state.trial)
    ratio = (ot_new/walker.ot)
    phase = cmath.phase(ratio)
    if abs(phase) < math.pi/2:
        walker.weight = walker.weight * ratio.real
        walker.ot = ot_new
        # Todo : remove computation of green's function repeatedly.
        walker.greens_function(state.trial, state.system.nup)
    else:
        walker.weight = 0.0


def kinetic_real(phi, state):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    For use with the continuus algorithm and free propagation.

    Parameters
    ----------
    walker : :class:`afqmcpy.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`afqmcpy.state.State`
        Simulation state.
    """
    nup = state.system.nup
    # Assuming that our walker is in UHF form.
    phi[:,:nup] = state.propagators.bt2[0].dot(phi[:,:nup])
    phi[:,nup:] = state.propagators.bt2[1].dot(phi[:,nup:])


def kinetic_ghf(phi, state):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    For use with the GHF algorithm.

    Parameters
    ----------
    walker : :class:`afqmcpy.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`afqmcpy.state.State`
        Simulation state.
    """
    nup = state.system.nup
    nb = state.system.nbasis
    # Assuming that our walker is in GHF form.
    phi[:nb,:nup] = state.propagators.bt2.dot(phi[:nb,:nup])
    phi[nb:,nup:] = state.propagators.bt2.dot(phi[nb:,nup:])


def propagate_potential_auxf(phi, state, field_config):
    """Propagate walker given a fixed set of auxiliary fields.

    Useful for debugging.

    Parameters
    ----------
    phi : :class:`numpy.ndarray`
        Walker's slater determinant to be updated.
    state : :class:`afqmcpy.state.State`
        Simulation state.
    field_config : numpy array
        Auxiliary field configurations to apply to walker.
    """

    bv_up = numpy.array([state.auxf[xi, 0] for xi in field_config])
    bv_down = numpy.array([state.auxf[xi, 1] for xi in field_config])
    phi[:,:nup] = numpy.einsum('i,ij->ij', bv_up, phi[:,:nup])
    phi[:,nup:] = numpy.einsum('i,ij->ij', bv_down, phi[:,nup:])

def construct_propagator_matrix(system, BT2, config, conjt=False):
    """Construct the full projector from a configuration of auxiliary fields.

    Parameters
    ----------
    config : numpy array
        Auxiliary field configuration.

    Returns
    -------
    B : :class:`numpy.ndarray`
        Full projector matrix.
    """
    bv_up = numpy.diag(numpy.array([system.auxf[xi, 0] for xi in config]))
    bv_down = numpy.diag(numpy.array([system.auxf[xi, 1] for xi in config]))
    Bup = BT2.dot(bv_up).dot(BT2)
    Bdown = BT2.dot(bv_down).dot(BT2)

    if conjt:
        return [Bup.conj().T, Bdown.conj().T]
    else:
        return [Bup, Bdown]

def construct_propagator_matrix_ghf(system, BT2, config, conjt=False):
    """Construct the full projector from a configuration of auxiliary fields.

    Parameters
    ----------
    config : numpy array
        Auxiliary field configuration.

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

def back_propagate(system, psi, trial, nstblz, BT2):
    r"""Perform back propagation for UHF style wavefunction.

    todo: Explanation.

    parameters
    ---------
    state : :class:`afqmcpy.state.state`
        state object
    psi_n : list of :class:`afqmcpy.walker.walker` objects
        current distribution of walkers, i.e., :math:`\tau_n'+\tau_{bp}`. on
        output the walker's auxiliary field counter will be set to zero if we
        are not also calculating an itcf.
    step : int
        simulation step (modulo total number of fields to save). this is
        necessary when estimating an itcf for imaginary times >> back
        propagation time.

    returns
    -------
    psi_bp : list of :class:`afqmcpy.walker.walker` objects
        back propagated list of walkers.
    """

    psi_bp = [walker.Walker(1, system, trial, w) for w in range(len(psi))]
    nup = system.nup
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, ws) in enumerate(reversed(list(w))):
            B = construct_propagator_matrix(system, BT2, ws.field_config,
                                            conjt=True)
            psi_bp[iw].phi[:,:nup] = B[0].dot(psi_bp[iw].phi[:,:nup])
            psi_bp[iw].phi[:,nup:] = B[1].dot(psi_bp[iw].phi[:,nup:])
            if i % nstblz == 0:
                psi_bp[iw].reortho(nup)
    return psi_bp

def back_propagate_ghf(system, psi, trial, nstblz, BT2):
    r"""perform backpropagation.

    todo: explanation and disentangle measurement from act.

    parameters
    ---------
    state : :class:`afqmcpy.state.State`
        state object
    psi_n : list of :class:`afqmcpy.walker.Walker` objects
        current distribution of walkers, i.e., :math:`\tau_n'+\tau_{bp}`. on
        output the walker's auxiliary field counter will be set to zero if we
        are not also calculating an itcf.
    step : int
        simulation step (modulo total number of fields to save). this is
        necessary when estimating an itcf for imaginary times >> back
        propagation time.

    returns
    -------
    psi_bp : list of :class:`afqmcpy.walker.Walker` objects
        back propagated list of walkers.
    """

    psi_bp = [walker.MultiGHFWalker(1, system, trial, w, weights='ones',
                                    wfn0='GHF') for w in range(len(psi))]
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, ws) in enumerate(reversed(list(w))):
            B = construct_propagator_matrix_ghf(system, BT2, ws.field_config,
                                                conjt=True)
            for (idet, psi_i) in enumerate(psi_bp[iw].phi):
                # propagate each component of multi-determinant expansion
                psi_i = B.dot(psi_i)
                if i % nstblz == 0:
                    # implicitly propagating the full GHF wavefunction
                    detR = afqmcpy.utils.reortho(psi_i)
                    psi_bp[iw].weights[idet] *= detR
    return psi_bp

def propagate_single(state, psi, B):
    r"""Perform backpropagation for single configuration.

    explanation...

    Parameters
    ---------
    state : :class:`afqmcpy.state.State`
        state object
    psi : list of :class:`afqmcpy.walker.Walker` objects
        Initial states to back propagate.
    B : numpy array
        Propagation matrix.
    """
    nup = state.system.nup
    psi.phi[:,:nup] = B[0].dot(psi.phi[:,:nup])
    psi.phi[:,nup:] = B[1].dot(psi.phi[:,nup:])


def kinetic_kspace(psi, state):
    """Apply the kinetic energy projector in kspace.

    May be faster for very large dilute lattices.
    """
    s = state.system
    # Transform psi to kspace by fft-ing its columns.
    tup = afqmcpy.utils.fft_wavefunction(psi[:,:s.nup], s.nx, s.ny, s.nup, psi[:,:s.nup].shape)
    tdown = afqmcpy.utils.fft_wavefunction(psi[:,s.nup:], s.nx, s.ny, s.ndown, psi[:,s.nup:].shape)
    # Kinetic enery operator is diagonal in momentum space.
    # Note that multiplying by diagonal btk in this way is faster than using
    # einsum and way faster than using dot using an actual diagonal matrix.
    tup = (state.propagators.btk*tup.T).T
    tdown = (state.propagators.btk*tdown.T).T
    # Transform psi to kspace by fft-ing its columns.
    tup = afqmcpy.utils.ifft_wavefunction(tup, s.nx, s.ny, s.nup, tup.shape)
    tdown = afqmcpy.utils.ifft_wavefunction(tdown, s.nx, s.ny, s.ndown, tdown.shape)
    if not state.cplx:
        psi[:,:s.nup] = tup.astype(float)
        psi[:,s.nup:] = tdown.astype(float)
    else:
        psi[:,:s.nup] = tup
        psi[:,s.nup:] = tdown

_projectors = {
    'potential': {
        'Hubbard': {
            'discrete': discrete_hubbard,
            'generic': generic_continuous,
            'continuous': dumb_hubbard,
        }
    }
}

_propagators = {
    'discrete': {
        'free': propagate_walker_free,
        'constrained': propagate_walker_discrete,
    },
    'continuous': {
        'free': propagate_walker_free_continuous,
        'constrained': propagate_walker_continuous,
    }
}

class Propagator:
    '''Base propagator class'''

    def __init__(self, qmc, system, trial):
        if trial.type == 'GHF':
            self.bt2 = scipy.linalg.expm(-0.5*qmc.dt*system.T[0])
        else:
            self.bt2 = numpy.array([scipy.linalg.expm(-0.5*qmc.dt*system.T[0]),
                                    scipy.linalg.expm(-0.5*qmc.dt*system.T[1])])
        if trial.type == 'GHF' and trial.bp_wfn is not None:
            self.BT_BP = scipy.linalg.block_diag(self.bt2, self.bt2)
            self.back_propagate = self.back_propagate_ghf
        else:
            self.BT_BP = self.bt2
            self.back_propagate = self.back_propagate_uhf
        self.nstblz = qmc.nstblz
        self.btk = numpy.exp(-0.5*qmc.dt*system.eks)
        hs_type = qmc.hubbard_stratonovich
        constraint = qmc.constraint
        self.propagate_walker = _propagators[hs_type][constraint]
        model = system.__class__.__name__
        self.potential = _projectors['potential'][model][hs_type]
        if trial.name == 'multi_determinant':
            if trial.type == 'GHF':
                self.calculate_overlap_ratio = calculate_overlap_ratio_multi_ghf
                self.kinetic = kinetic_ghf
            else:
                self.calculate_overlap_ratio = calculate_overlap_ratio_multi_det
                self.kinetic = kinetic_real
        else:
            self.calculate_overlap_ratio = calculate_overlap_ratio_single_det
            if qmc.ffts:
                self.kinetic = kinetic_kspace
            else:
                self.kinetic = kinetic_real

    def back_propagate_uhf(self, system, psi, trial):
        return afqmcpy.propagation.back_propagate(system, psi, trial,
                                                  self.nstblz, self.BT_BP)

    def back_propagate_ghf(self, system, psi, trial):
        return afqmcpy.propagation.back_propagate_ghf(system, psi, trial,
                                                      self.nstblz, self.BT_BP)
