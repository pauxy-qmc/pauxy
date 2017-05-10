'''Routines for performing propagation of a walker'''

import random
import numpy
import afqmcpy.utils as utils
from math import exp, phase


def kinetic_direct(walker, bt2, trial):
    '''Propagate by the kinetic term by direct matrix multiplication.

Parameters
----------
walker : :class:`Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_K/2 and
    updated the weight appropriately. Updates inplace.
bk2 : numpy.ndarray
    Exponential of the kinetic propagator :math:`e^{-\Delta\tau/2 \hat{K}}`
trial : numpy.ndarray
    Trial wavefunction
'''
    walker.phi[0] = bt2.dot(walker.phi[0])
    walker.phi[1] = bt2.dot(walker.phi[1])
    # Update inverse overlap
    walker.inverse_overlap(trial)
    # Update walker weight
    ot_new = walker.calc_otrial(trial)
    walker.greens_function(trial)
    if ot_new/walker.ot > 1e-16:
        walker.weight = walker.weight * (ot_new/walker.ot)
        walker.ot = ot_new
    else:
        walker.weight = 0.0


def discrete_hubbard(walker, auxf, nbasis, trial):
    '''Propagate by potential term using discrete HS transform.

Parameters
----------
walker : :class:`Walker`
    Walker object to be updated. On output we have acted on |phi_i> by B_V and
    updated the weight appropriately. Updates inplace.
auxf : numpy.ndarray
    Possible values of the exponential discrete auxilliary fields.
nbasis : int
    Number of single-particle basis functions (2M for spin).
trial : numpy.ndarray
    Trial wavefunction.
'''
    # Construct random auxilliary field.
    delta = auxf - 1
    for i in range(0, nbasis):
        # Ratio of determinants for the two choices of auxilliary fields
        probs = 0.5 * numpy.array([(1+delta[0][0]*walker.G[0][i,i])*(1+delta[0][1]*walker.G[1][i,i]),
                                (1+delta[1][0]*walker.G[0][i,i])*(1+delta[1][1]*walker.G[1][i,i])])
        norm = sum(probs)
        walker.weight = walker.weight * norm
        r = random.random()
        if walker.weight > 0:
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
        walker.inv_ovlp[0] = utils.sherman_morrison(walker.inv_ovlp[0], trial[0].T[:,i],
                                            vtup)
        walker.inv_ovlp[1] = utils.sherman_morrison(walker.inv_ovlp[1], trial[1].T[:,i],
                                            vtdown)
        walker.greens_function(trial)


def generic_continuous(walker, U, trial, nmax_exp=4):
    '''Continuous HS transformation

    This form assumes nothing about the form of the two-body Hamiltonian and
    is thus quite slow, particularly if the matrix is M^2xM^2.

Parameters
----------
walker : :class:`Walker`
    Walker object to be updated. On output we have acted on |phi_i> by
    B_V(x-x')/2 and updated the weight appropriately. Updates inplace.
U : numpy.ndarray
    Matrix containing eigenvectors of gamma (times :math:`\sqrt{-\lambda}`.
trial : numpy.ndarray
    Trial wavefunction.
nmax_exp : int
    Maximum expansion order of matrix exponential.
'''

    # iterate over spins
    for i in range(0, 2):
        # Generate ~M^2 normally distributed auxiliary fields.
        sigma = numpy.random.normal(0.0, 1.0, len(U))
        # Construct HS potential, V_HS = sigma dot U
        V_HS = numpy.einsum('ij,j->i', sigma, U)
        # Reshape so we can apply to MxN Slater determinant.
        V_HS = numpy.reshape(V_HS, (M,M))
        for n in range(1, nmax_exp+1):
            walker.phi[i] += numpy.factorial(n) * np.dot(V_HS, phi)

    # Update inverse and green's function
    walker.inverse_overlap(trial)
    walker.greens_function(trial)
    # Perform importance sampling, phaseless and real local energy approximation and update
    E_L = estimators.local_energy(system, walker).real
    ot_new = walker.calc_otrial(trial)
    dtheta = phase(ot_new/walker.ot)
    walker.weight = walker.weight * exp(-0.5*system.dt*(walker.E_L-E_L))
                                  * max(0, dtheta)
    walker.E_L = E_L
    walker.ot_new = ot_new
