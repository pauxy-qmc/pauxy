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

    if walker.weight.real > 0:
        state.propagators.kinetic(walker, state)
    if walker.weight.real > 0:
        state.propagators.potential(walker, state)
    if walker.weight.real > 0:
        state.propagators.kinetic(walker, state)


def propagate_walker_free(walker, state):
    '''Free projection without importance sampling.

'''
    walker.phi[0] = state.propagators.bt2.dot(walker.phi[0])
    walker.phi[1] = state.propagators.bt2.dot(walker.phi[1])
    delta = state.auxf - 1
    for i in range(0, state.system.nbasis):
        # Is this necessary?
        if walker.weight > 0:
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
    walker.inverse_overlap(state.psi_trial)
    # Update walker weight
    walker.ot = walker.calc_otrial(state.psi_trial)
    walker.greens_function(state.psi_trial)


def propagate_walker_free_continuous(walker, state):
    '''Free projection without importance sampling.

'''
    walker.phi[0] = state.propagators.bt2.dot(walker.phi[0])
    walker.phi[1] = state.propagators.bt2.dot(walker.phi[1])
    x_i = cmath.sqrt((-state.system.U*state.dt))*numpy.random.normal(0.0, 1.0, state.system.nbasis)
    bv = numpy.diag(numpy.exp(x_i))
    walker.phi[0] = bv.dot(walker.phi[0])
    walker.phi[1] = bv.dot(walker.phi[1])
    # delta = state.auxf - 1
    # for i in range(0, state.system.nbasis):
        # # Is this necessary?
        # for i in range(0, state.system.nbasis):
            # # For convenience..
            # # Need shift here
            # x_i = cmath.sqrt((-state.system.U*state.dt))*numpy.random.normal(0.0, 1.0)
            # delta = cmath.exp(x_i) - 1
            # # Check speed here with numpy (restructure array)
            # vtup = walker.phi[0][i,:] * delta
            # vtdown = walker.phi[1][i,:] * delta
            # walker.phi[0][i,:] = walker.phi[0][i,:] + vtup
            # walker.phi[1][i,:] = walker.phi[1][i,:] + vtdown
    walker.phi[0] = state.propagators.bt2.dot(walker.phi[0])
    walker.phi[1] = state.propagators.bt2.dot(walker.phi[1])
    walker.inverse_overlap(state.psi_trial)
    # Update walker weight
    walker.ot = walker.calc_otrial(state.psi_trial)
    # print (walker.ot)
    walker.greens_function(state.psi_trial)


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
    state.propagators.potential(walker, state)
    state.propagators.kinetic(walker, state)

    # Phaseless approximation
    walker.inverse_overlap(state.psi_trial)
    walker.greens_function(state.psi_trial)
    (E_L, walker.vbar) = estimators.local_energy(state.system, walker.G)
    av = -3.481
    if (E_L >= av + state.local_energy_bound):
        E_L = av + state.local_energy_bound
    elif (E_L <= av - state.local_energy_bound):
        E_L <= av - state.local_energy_bound
    else:
        E_L = E_L
    ot_new = walker.calc_otrial(state.psi_trial)
    dtheta = cmath.phase(ot_new/walker.ot)
    # print (dtheta/(math.pi))
    print (E_L, walker.weight, walker.vbar, ot_new, walker.ot,
            math.exp(-0.5*state.dt*(walker.E_L+E_L)), dtheta/math.pi, max(0, math.cos(dtheta)))
    # if (math.cos(dtheta) < 1e-8):
        # print (E_L, walker.vbar, ot_new, walker.ot, dtheta, max(0, math.cos(dtheta)))
        # print (abs(dtheta)/math.pi)
    walker.weight = (walker.weight * math.exp(-0.5*state.dt*(walker.E_L+E_L))
                                   * max(0, math.cos(dtheta)))
    walker.E_L = E_L
    walker.ot = ot_new


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
            else:
                vtup = walker.phi[0][i,:] * delta[1, 0]
                vtdown = walker.phi[1][i,:] * delta[1, 1]
                walker.phi[0][i,:] = walker.phi[0][i,:] + vtup
                walker.phi[1][i,:] = walker.phi[1][i,:] + vtdown
                walker.ot = 2 * walker.ot * probs[1]
        walker.inv_ovlp[0] = utils.sherman_morrison(walker.inv_ovlp[0],
                                                    state.psi_trial[0].T[:,i],
                                                    vtup)
        walker.inv_ovlp[1] = utils.sherman_morrison(walker.inv_ovlp[1],
                                                    state.psi_trial[1].T[:,i],
                                                    vtdown)
        walker.greens_function(state.psi_trial)


def continuous_hubbard(walker, state):
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

    x_i = cmath.sqrt((-state.system.U*state.dt))*numpy.random.normal(0.0, 1.0, state.system.nbasis) + state.dt**0.5*walker.vbar
    bv = numpy.diag(numpy.exp(x_i))
    walker.phi[0] = bv.dot(walker.phi[0])
    walker.phi[1] = bv.dot(walker.phi[1])
    # for i in range(0, state.system.nbasis):
        # # For convenience..
        # # Need shift here
        # x_i = cmath.sqrt((-2.0*state.system.U*state.dt)) * (
                # numpy.random.normal(0.0, 1.0) - state.dt**0.5*walker.vbar)
        # delta = cmath.exp(x_i) - 1
        # # Check speed here with numpy (restructure array)
        # vtup = walker.phi[0][i,:] * delta
        # vtdown = walker.phi[1][i,:] * delta
        # walker.phi[0][i,:] = walker.phi[0][i,:] + vtup
        # walker.phi[1][i,:] = walker.phi[1][i,:] + vtdown


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

    # For convenience..
    # Need shift here
    x_i = numpy.random.normal(0.0, 1.0, state.system.nbasis)
    gterm = (1j)*(state.dt*state.system.U)**0.5*(numpy.diag(walker.G[0])+numpy.diag(walker.G[1]))
    xmxb = (1j)*(state.dt*state.system.U)**0.5 * (x_i+gterm)
    gterm = numpy.diag(walker.G[0]) + numpy.diag(walker.G[1])
    # print gterm
    xmxb = (1j)*(state.dt*state.system.U)**0.5 * (x_i+(1j)*(state.dt*state.system.U)**0.5*gterm)
    EXP_VHS = numpy.exp(xmxb)
    # walker.phi[0] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[0])
    # walker.phi[1] = numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[1])
    bv = numpy.diag(numpy.exp(xmxb))
    # print (bv.dot(walker.phi[0]))
    # print (numpy.einsum('i,ij->ij', EXP_VHS, walker.phi[0]))
    walker.phi[0] = bv.dot(walker.phi[0])
    walker.phi[1] = bv.dot(walker.phi[1])


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
            walker.phi[i] += numpy.factorial(n) * np.dot(V_HS, phi)

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
    walker.inverse_overlap(state.psi_trial)
    # Update walker weight
    ot_new = walker.calc_otrial(state.psi_trial)
    walker.greens_function(state.psi_trial)
    ratio = ot_new / walker.ot
    if ratio.real > 1e-16 and abs(ratio.imag) < 1e-16:
        walker.weight = walker.weight * (ot_new/walker.ot)
        walker.ot = ot_new
    else:
        walker.weight = 0.0


def kinetic_continuous(walker, state):
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
    walker.phi[0] = state.propagators.bt2.dot(walker.phi[0])
    walker.phi[1] = state.propagators.bt2.dot(walker.phi[1])


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
            'opt_continuous': continuous_hubbard,
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
