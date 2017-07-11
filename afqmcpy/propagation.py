"""Routines for performing propagation of a walker"""

# TODO: refactor to avoid code repetition between similar routines.

import numpy
import scipy.linalg
import math
import cmath
import copy
import afqmcpy.utils as utils
import afqmcpy.estimators as estimators
import afqmcpy.walker as walker


def propagate_walker_discrete(walker, state):
    """Wrapper function for propagation using discrete transformation

    The discrete transformation allows us to split the application of the
    projector up a bit more, which allows up to make use of fast matrix update
    routines since only a row might change.

Parameters
----------
walker : :class:`walker.Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B(x) and
    updated the weight appropriately. Updates inplace.
state : :class:`state.State`
    Simulation state.
"""

    if abs(walker.weight) > 0:
        state.propagators.kinetic(walker, state)
    if abs(walker.weight) > 0:
        state.propagators.potential(walker, state)
    if abs(walker.weight.real) > 0:
        state.propagators.kinetic(walker, state)


def propagate_walker_free(walker, state):
    """Propagate walker without imposing constraint.

    Uses single-site updates for potential term.

    Parameters
    ----------
    walker : :class:`walker.Walker`
        Walker object to be updated. On output we have acted on |phi_i> by B(x) and
        updated the weight appropriately. Updates inplace.
    state : :class:`state.State`
        Simulation state.
"""
    kinetic_direct(walker.phi, state)
    delta = state.auxf - 1
    for i in range(0, state.system.nbasis):
        if abs(walker.weight) > 0:
            r = numpy.random.random()
            # TODO: remove code repition.
            if r > 0.5:
                vtup = walker.phi[0][i,:] * delta[0, 0]
                vtdown = walker.phi[1][i,:] * delta[0, 1]
                walker.phi[0][i,:] = walker.phi[0][i,:] + vtup
                walker.phi[1][i,:] = walker.phi[1][i,:] + vtdown
            else:
                vtup = walker.phi[0][i,:] * delta[1, 0]
                vtdown = walker.phi[1][i,:] * delta[1, 1]
                walker.phi[0][i,:] = walker.phi[0][i,:] + vtup
                walker.phi[1][i,:] = walker.phi[1][i,:] + vtdown
    kinetic_direct(walker.phi, state)
    walker.inverse_overlap(state.trial.psi)
    # Update walker weight
    walker.ot = walker.calc_otrial(state.trial.psi)
    walker.greens_function(state.trial.psi)


def propagate_walker_free_continuous(walker, state):
    """Free projection for continuous HS transformation.

    TODO: update if ever adapted to other model types.

    Parameters
    ----------
    walker : :class:`walker.Walker`
        Walker object to be updated. On output we have acted on |phi_i> by B(x) and
        updated the weight appropriately. Updates inplace.
    state : :class:`state.State`
        Simulation state.
"""
    # 1. Apply kinetic projector.
    kinetic_direct(walker.phi, state)
    # Normally distributed random numbers.
    xfields =  numpy.random.normal(0.0, 1.0, state.system.nbasis)
    sxf = sum(xfields)
    # Constant, field dependent term emerging when subtracting mean-field.
    c_xf = cmath.exp(0.5*state.ut_fac*state.mf_nsq-state.iut_fac*state.mf_shift*sxf)
    # Potential propagator.
    bv = numpy.diag(numpy.exp(state.iut_fac*xfields+0.5*state.ut_fac*(1-2*state.mf_shift)))
    # 2. Apply potential projector.
    walker.phi[0] = bv.dot(walker.phi[0])
    walker.phi[1] = bv.dot(walker.phi[1])
    # 3. Apply kinetic projector.
    kinetic_direct(walker.phi, state)
    walker.inverse_overlap(state.trial.psi)
    walker.ot = walker.calc_otrial(state.trial.psi)
    walker.greens_function(state.trial.psi)
    # Constant terms are included in the walker's weight.
    walker.weight = walker.weight * c_xf


def propagate_walker_continuous(walker, state):
    """Wrapper function for propagation using continuous transformation.

    This applied the phaseless, local energy approximation and uses importance
    sampling.

Parameters
----------
walker : :class:`walker.Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_V and
    updated the weight appropriately. Updates inplace.
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
    walker.inverse_overlap(state.trial.psi)
    walker.greens_function(state.trial.psi)
    E_L = estimators.local_energy(state.system, walker.G)[0].real
    # Check for large population fluctuations
    E_L = local_energy_bound(E_L, state.mean_local_energy, state.local_energy_bound)
    ot_new = walker.calc_otrial(state.trial.psi)
    # Walker's phase.
    dtheta = cmath.phase(cxf*ot_new/walker.ot)
    walker.weight = (walker.weight * math.exp(-0.5*state.dt*(walker.E_L+E_L))
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

def discrete_hubbard(walker, state):
    """Propagate by potential term using discrete HS transform.

Parameters
----------
walker : :class:`afqmcpy.walker.Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_V and
    updated the weight appropriately. Updates inplace.
state : :class:`afqmcpy.state.State`
    Simulation state.
"""
    # Construct random auxilliary field.
    delta = state.auxf - 1
    for i in range(0, state.system.nbasis):
        # Ratio of determinants for the two choices of auxilliary fields
        probs = 0.5 * numpy.array([(1+delta[0][0]*walker.G[0][i,i])*(1+delta[0][1]*walker.G[1][i,i]),
                                   (1+delta[1][0]*walker.G[0][i,i])*(1+delta[1][1]*walker.G[1][i,i])])
        norm = sum(probs)
        r = numpy.random.random()
        # Is this necessary?
        if norm > 0:
            walker.weight = walker.weight * norm
            if r < probs[0]/norm:
                vtup = walker.phi[0][i,:] * delta[0, 0]
                vtdown = walker.phi[1][i,:] * delta[0, 1]
                walker.phi[0][i,:] = walker.phi[0][i,:] + vtup
                walker.phi[1][i,:] = walker.phi[1][i,:] + vtdown
                walker.ot = 2 * walker.ot * probs[0]
                walker.field_config[i] = 0
            else:
                vtup = walker.phi[0][i,:] * delta[1, 0]
                vtdown = walker.phi[1][i,:] * delta[1, 1]
                walker.phi[0][i,:] = walker.phi[0][i,:] + vtup
                walker.phi[1][i,:] = walker.phi[1][i,:] + vtdown
                walker.ot = 2 * walker.ot * probs[1]
                walker.field_config[i] = 1
        walker.inv_ovlp[0] = utils.sherman_morrison(walker.inv_ovlp[0],
                                                    state.trial.psi[0].T[:,i],
                                                    vtup)
        walker.inv_ovlp[1] = utils.sherman_morrison(walker.inv_ovlp[1],
                                                    state.trial.psi[1].T[:,i],
                                                    vtdown)
        walker.greens_function(state.trial.psi)


def dumb_hubbard(walker, state):
    """Continuous Hubbard-Statonovich transformation for Hubbard model.

    Only requires M auxiliary fields.

Parameters
----------
walker : :class:`afqmcpy.walker.Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_V and
    updated the weight appropriately. Updates inplace.
state : :class:`afqmcpy.state.State`
    Simulation state.
"""

    # Normally distrubted auxiliary fields.
    xi = numpy.random.normal(0.0, 1.0, state.system.nbasis)
    # Optimal field shift for real local energy approximation.
    xi_opt = -state.iut_fac*(numpy.diag(walker.G[0])+numpy.diag(walker.G[1])-state.mf_shift)
    sxf = sum(xi-xi_opt)
    # Propagator for potential term with mean field and auxilary field shift.
    c_xf = cmath.exp(0.5*state.ut_fac*state.mf_nsq-state.iut_fac*state.mf_shift*sxf)
    EXP_VHS = numpy.exp(0.5*state.ut_fac*(1-2.0*state.mf_shift)+state.iut_fac*(xi-xi_opt))
    walker.phi[0] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[0])
    walker.phi[1] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[1])
    return c_xf


def generic_continuous(walker, state):
    """Continuous HS transformation

    This form assumes nothing about the form of the two-body Hamiltonian and
    is thus quite slow, particularly if the matrix is M^2xM^2.

    Todo: check if this actually works.

Parameters
----------
walker : :class:`afqmcpy.walker.Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_V and
    updated the weight appropriately. Updates inplace.
state : :class:`afqmcpy.state.State`
    Simulation state.
"""

    # iterate over spins
    for i in range(0, 2):
        # Generate ~M^2 normally distributed auxiliary fields.
        sigma = state.dt**0.5 * numpy.random.normal(0.0, 1.0, len(state.U))
        # Construct HS potential, V_HS = sigma dot U
        V_HS = numpy.einsum('ij,j->i', sigma, state.U)
        # Reshape so we can apply to MxN Slater determinant.
        V_HS = numpy.reshape(V_HS, (M,M))
        for n in range(1, nmax_exp+1):
            walker.phi[i] = walker.phi[i] + numpy.factorial(n) * np.dot(V_HS, phi)

    # Update inverse and green's function
    walker.inverse_overlap(trial)
    walker.greens_function(trial)
    # Perform importance sampling, phaseless and real local energy approximation and update
    E_L = estimators.local_energy(system, walker.G)[0].real
    ot_new = walker.calc_otrial(trial)
    dtheta = cmath.phase(ot_new/walker.ot)
    walker.weight = (walker.weight * exp(-0.5*system.dt*(walker.E_L-E_L))
                                  * max(0, cos(dtheta)))
    walker.E_L = E_L
    walker.ot = ot_new


def kinetic_importance_sampling(walker, state):
    """Propagate by the kinetic term by direct matrix multiplication.

Parameters
----------
walker : :class:`afqmcpy.walker.Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_V and
    updated the weight appropriately. Updates inplace.
state : :class:`afqmcpy.state.State`
    Simulation state.
"""
    walker.phi[0] = state.propagators.bt2.dot(walker.phi[0])
    walker.phi[1] = state.propagators.bt2.dot(walker.phi[1])
    # Update inverse overlap
    walker.inverse_overlap(state.trial.psi)
    # Update walker weight
    ot_new = walker.calc_otrial(state.trial.psi)
    walker.greens_function(state.trial.psi)
    ratio = ot_new / walker.ot
    if ratio.real > 1e-16 and abs(ratio.imag) < 1e-16:
        walker.weight = walker.weight * (ot_new/walker.ot)
        walker.ot = ot_new
    else:
        walker.weight = 0.0


def kinetic_direct(phi, state):
    """Propagate by the kinetic term by direct matrix multiplication.

    For use with the continuus algorithm and free propagation.

    Parameters
    ----------
    walker : :class:`afqmcpy.walker.Walker`
        Walker object to be updated. On output we have acted on |phi_i> by B_V and
        updated the weight appropriately. Updates inplace.
    state : :class:`afqmcpy.state.State`
        Simulation state.
    """
    phi[0] = state.propagators.bt2.dot(phi[0])
    phi[1] = state.propagators.bt2.dot(phi[1])


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
    phi[0] = numpy.einsum('i,ij->ij', bv_up, phi[0])
    phi[1] = numpy.einsum('i,ij->ij', bv_down, phi[1])

def construct_propagator_matrix(state, config, conjt=False):
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
    BK2 = state.propagators.bt2
    bv_up = numpy.diag(numpy.array([state.auxf[xi, 0] for xi in config]))
    bv_down = numpy.diag(numpy.array([state.auxf[xi, 1] for xi in config]))
    Bup = BK2.dot(bv_up).dot(BK2)
    Bdown = BK2.dot(bv_down).dot(BK2)

    if conjt:
        return [Bup.conj().T, Bdown.conj().T]
    else:
        return [Bup, Bdown]

def back_propagate(state, psi):
    r"""Perform backpropagation.

    TODO: explanation and disentangle measurement from act.

    Parameters
    ---------
    state : :class:`afqmcpy.state.State`
        state object
    psi_n : list of :class:`afqmcpy.walker.Walker` objects
        current distribution of walkers, i.e., :math:`\tau_n'+\tau_{bp}`. On
        output the walker's auxiliary field counter will be set to zero if we
        are not also calculating an ITCF.
    step : int
        Simulation step (modulo total number of fields to save). This is
        necessary when estimating an ITCF for imaginary times >> back
        propagation time.

    Returns
    -------
    psi_bp : list of :class:`afqmcpy.walker.Walker` objects
        Back propagated list of walkers.
    """

    psi_bp = [walker.Walker(1, state.system, state.trial.psi, w)
              for w in range(state.nwalkers)]
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, ws) in enumerate(reversed(list(w))):
            B = construct_propagator_matrix(state, ws.field_config)
            psi_bp[iw].phi[0] = B[0].dot(psi_bp[iw].phi[0])
            psi_bp[iw].phi[1] = B[1].dot(psi_bp[iw].phi[1])
            if i % state.nmeasure == 0:
                psi_bp[iw].reortho()
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
    psi.phi[0] = B[0].dot(psi.phi[0])
    psi.phi[1] = B[1].dot(psi.phi[1])

_projectors = {
    'kinetic': {
        'discrete': kinetic_importance_sampling,
        'continuous': kinetic_direct,
        'opt_continuous': kinetic_direct,
        'dumb_continuous': kinetic_direct,
    },
    'potential': {
        'Hubbard': {
            'discrete': discrete_hubbard,
            'continuous': generic_continuous,
            'opt_continuous': dumb_hubbard,
            'dumb_continuous': dumb_hubbard,
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

# This shouldn't exist, just rename in state at the beginning and rename module
# functions
class Projectors:
    '''Base propagator class'''

    def __init__(self, model, hs_type, dt, T, importance_sampling):
        self.bt2 = scipy.linalg.expm(-0.5*dt*T)
        if 'continuous' in hs_type:
            if importance_sampling:
                self.propagate_walker = _propagators['continuous']['constrained']
            else:
                self.propagate_walker = _propagators['continuous']['free']
        elif importance_sampling:
            self.propagate_walker = _propagators['discrete']['constrained']
        else:
            self.propagate_walker = _propagators['discrete']['free']
        self.kinetic = _projectors['kinetic'][hs_type]
        self.potential = _projectors['potential'][model][hs_type]
