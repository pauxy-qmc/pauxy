'''Routines for performing propagation of a walker'''

import numpy
import scipy.linalg
import afqmcpy.utils as utils
import afqmcpy.estimators as estimators
import math
import cmath


def propagate_walker_discrete(walker, state):
    '''Wrapper function for propagation using discrete transformation

    The discrete transformation allows us to split the application of the
    projector up a bit more, which allows up to make use of fast matrix update
    routines since only a row might change.

    Todo: This about this for continuous transformation.

Parameters
----------
walker : :class:`walker.Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_V and
    updated the weight appropriately. Updates inplace.
state : :class:`state.State`
    Simulation state.
'''

    if abs(walker.weight) > 0:
        state.propagators.kinetic(walker, state)
    if abs(walker.weight) > 0:
        state.propagators.potential(walker, state)
    if abs(walker.weight.real) > 0:
        state.propagators.kinetic(walker, state)


def propagate_walker_free(walker, state):
    '''Free projection without importance sampling.

'''
    walker.phi[0] = state.propagators.bt2.dot(walker.phi[0])
    walker.phi[1] = state.propagators.bt2.dot(walker.phi[1])
    delta = state.auxf - 1
    for i in range(0, state.system.nbasis):
        # Is this necessary?
        if abs(walker.weight) > 0:
            r = numpy.random.random()
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
    walker.phi[0] = state.propagators.bt2.dot(walker.phi[0])
    walker.phi[1] = state.propagators.bt2.dot(walker.phi[1])
    walker.inverse_overlap(state.trial.psi)
    # Update walker weight
    walker.ot = walker.calc_otrial(state.trial.psi)
    walker.greens_function(state.trial.psi)


def propagate_walker_free_continuous(walker, state):
    '''Free projection without importance sampling.

'''
    walker.phi[0] = state.propagators.bt2.dot(walker.phi[0])
    walker.phi[1] = state.propagators.bt2.dot(walker.phi[1])
    xfields =  numpy.random.normal(0.0, 1.0, state.system.nbasis)
    sxf = sum(xfields)
    c_xf = cmath.exp(0.5*state.ut_fac*state.mf_nsq-state.iut_fac*state.mf_shift*sxf)
    bv = numpy.diag(numpy.exp(state.iut_fac*xfields+0.5*state.ut_fac*(1-2*state.mf_shift)))
    walker.phi[0] = bv.dot(walker.phi[0])
    walker.phi[1] = bv.dot(walker.phi[1])
    walker.phi[0] = state.propagators.bt2.dot(walker.phi[0])
    walker.phi[1] = state.propagators.bt2.dot(walker.phi[1])
    walker.inverse_overlap(state.trial.psi)
    walker.ot = walker.calc_otrial(state.trial.psi)
    walker.greens_function(state.trial.psi)
    walker.weight = walker.weight * c_xf


def propagate_walker_continuous(walker, state):
    '''Wrapper function for propagation using continuous transformation

Parameters
----------
walker : :class:`walker.Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_V and
    updated the weight appropriately. Updates inplace.
state : :class:`state.State`
    Simulation state.
'''

    state.propagators.kinetic(walker, state)
    cxf = state.propagators.potential(walker, state)
    state.propagators.kinetic(walker, state)

    # Now apply phaseless, real local energy approximation
    walker.inverse_overlap(state.trial.psi)
    walker.greens_function(state.trial.psi)
    E_L = estimators.local_energy(state.system, walker.G)[0].real
    # Check for large population fluctuations
    E_L = local_energy_bound(E_L, state.mean_local_energy, state.local_energy_bound)
    ot_new = walker.calc_otrial(state.trial.psi)
    dtheta = cmath.phase(cxf*ot_new/walker.ot)
    # print (E_L, walker.weight, walker.vbar, ot_new, walker.ot,
            # math.exp(-0.5*state.dt*(walker.E_L+E_L)), dtheta/math.pi, max(0, math.cos(dtheta)))
    # if (math.cos(dtheta) < 1e-8):
        # print (E_L, walker.vbar, ot_new, walker.ot, dtheta, max(0, math.cos(dtheta)))
        # print (abs(dtheta)/math.pi)
    walker.weight = (walker.weight * math.exp(-0.5*state.dt*(walker.E_L+E_L))
                                   * max(0, math.cos(dtheta)))
    walker.E_L = E_L
    walker.ot = ot_new


def local_energy_bound(local_energy, mean, threshold):
    '''Try to suppress rare population events by imposing local energy bound.

    See: Purwanto et al., Phys. Rev. B 80, 214116 (2009).

Parameters
----------
local_energy : float
    Local energy of current walker
mean : float
    Mean value of local energy about which we impose the threshold / bound.
threshold : float
    Amount of lee-way for energy fluctuations about the mean.
'''

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
    '''Propagate by potential term using discrete HS transform.

Parameters
----------
walker : :class:`Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_V and
    updated the weight appropriately. Updates inplace.
auxf : :class:`numpy.ndarray`
    Possible values of the exponential discrete auxilliary fields.
nbasis : int
    Number of single-particle basis functions (2M for spin).
trial : :class:`numpy.ndarray`
    Trial wavefunction.
'''
    # Construct random auxilliary field.
    delta = state.auxf - 1
    for i in range(0, state.system.nbasis):
        # Ratio of determinants for the two choices of auxilliary fields
        probs = 0.5 * numpy.array([(1+delta[0][0]*walker.G[0][i,i])*(1+delta[0][1]*walker.G[1][i,i]),
                                (1+delta[1][0]*walker.G[0][i,i])*(1+delta[1][1]*walker.G[1][i,i])])
        norm = sum(probs)
        walker.weight = walker.weight * norm
        r = numpy.random.random()
        # Is this necessary?
        if norm > 0:
            if r < probs[0]/norm:
                vtup = walker.phi[0][i,:] * delta[0, 0]
                vtdown = walker.phi[1][i,:] * delta[0, 1]
                walker.phi[0][i,:] = walker.phi[0][i,:] + vtup
                walker.phi[1][i,:] = walker.phi[1][i,:] + vtdown
                walker.ot = 2 * walker.ot * probs[0]
                walker.bp_auxf[self.bp_counter, i] = 0
            else:
                vtup = walker.phi[0][i,:] * delta[1, 0]
                vtdown = walker.phi[1][i,:] * delta[1, 1]
                walker.phi[0][i,:] = walker.phi[0][i,:] + vtup
                walker.phi[1][i,:] = walker.phi[1][i,:] + vtdown
                walker.ot = 2 * walker.ot * probs[1]
                walker.bp_auxf[self.bp_counter, i] = 1
        walker.inv_ovlp[0] = utils.sherman_morrison(walker.inv_ovlp[0],
                                                    state.trial.psi[0].T[:,i],
                                                    vtup)
        walker.inv_ovlp[1] = utils.sherman_morrison(walker.inv_ovlp[1],
                                                    state.trial.psi[1].T[:,i],
                                                    vtdown)
        walker.greens_function(state.trial.psi)
        walker.bp_counter = walker.bp_counter + 1


def dumb_hubbard(walker, state):
    '''Continuous Hubbard-Statonovich transformation for Hubbard model.

    Only requires M auxiliary fields.

Parameters
----------
walker : :class:`walker.Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_V and
    updated the weight appropriately. Updates inplace.
state : :class:`state.State`
    Simulation state.
'''

    # Normally distrubted auxiliary fields.
    xi = numpy.random.normal(0.0, 1.0, state.system.nbasis)
    # Optimal field shift for real local energy approximation.
    xi_opt = -state.iut_fac*(numpy.diag(walker.G[0])+numpy.diag(walker.G[1])-state.mf_shift)
    sxf = sum(xi-xi_opt)
    # Propagator for potential term with mean field and auxilary field shift.
    c_xf = cmath.exp(0.5*state.ut_fac*state.mf_nsq-state.iut_fac*state.mf_shift*sxf)
    # print ((xi-xi_opt)*state.iut_fac)
    EXP_VHS = numpy.exp(0.5*state.ut_fac*(1-2.0*state.mf_shift)+state.iut_fac*(xi-xi_opt))
    walker.phi[0] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[0])
    walker.phi[1] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[1])
    return c_xf


def generic_continuous(walker, state):
    '''Continuous HS transformation

    This form assumes nothing about the form of the two-body Hamiltonian and
    is thus quite slow, particularly if the matrix is M^2xM^2.

Parameters
----------
walker : :class:`Walker`
    Walker object to be updated. On output we have acted on |phi_i> by
    B_V(x-x')/2 and updated the weight appropriately. Updates inplace.
U : :class:`numpy.ndarray`
    Matrix containing eigenvectors of gamma (times :math:`\sqrt{-\lambda}`.
trial : numpy.ndarray
    Trial wavefunction.
nmax_exp : int
    Maximum expansion order of matrix exponential.
'''

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
    E_L = estimators.local_energy(system, walker.G).real
    ot_new = walker.calc_otrial(trial)
    dtheta = cmath.phase(ot_new/walker.ot)
    walker.weight = (walker.weight * exp(-0.5*system.dt*(walker.E_L-E_L))
                                  * max(0, cos(dtheta)))
    walker.E_L = E_L
    walker.ot = ot_new


def kinetic_direct(walker, state):
    '''Propagate by the kinetic term by direct matrix multiplication.

Parameters
----------
walker : :class:`Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_K/2 and
    updated the weight appropriately. Updates inplace.
bk2 : :class:`numpy.ndarray`
    Exponential of the kinetic propagator :math:`e^{-\Delta\tau/2 \hat{K}}`
trial : :class:`numpy.ndarray`
    Trial wavefunction
'''
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


def kinetic_continuous(phi, state):
    '''Propagate by the kinetic term by direct matrix multiplication.

    For use with the continuous algorithm.

Parameters
----------
walker : :class:`Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_K/2 and
    updated the weight appropriately. Updates inplace.
bk2 : :class:`numpy.ndarray`
    Exponential of the kinetic propagator :math:`e^{-\Delta\tau/2 \hat{K}}`
trial : :class:`numpy.ndarray`
    Trial wavefunction
'''
    phi[0] = state.propagators.bt2.dot(phi[0])
    phi[1] = state.propagators.bt2.dot(phi[1])


def propagate_potential_auxf(phi, state, field_config):

    bv_up = numpy.array([state.auxf[0, xi] for xi in field_config])
    bv_down = numpy.array([state.auxf[1, xi] for xi in field_config])
    phi[0] = numpy.einsum('i,ij->ij', bv_up, phi[0])
    phi[1] = numpy.einsum('i,ij->ij', bv_down, phi[0])



def back_propagate(state, psi, psit, psi_bp):

    for (iw, w) in enumerate(psi):
        for (step, field_config) in reversed(list(enumerate(w.bp_auxf))):
            kinetic_continuous(psi_bi[iw].phi, state)
            propagate_potential_auxf(psi_bp[iw].phi, state, field_config)
            kinetic_continuous(psi_bp[iw].phi, state)
            weights[iw] = numpy.conj(((w.weight/w.bp_weight)*w.bp_ot))

        afqmcpy.estimates.back_propagated_observables(state, w, psi[iw])


_projectors = {
    'kinetic': {
        'discrete': kinetic_direct,
        'continuous': kinetic_continuous,
        'opt_continuous': kinetic_continuous,
        'dumb_continuous': kinetic_continuous,
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
