import scipy.linalg
import numpy


def diag_sorted(H):

    (eigs, eigv) = scipy.linalg.eigh(H)
    idx = eigs.argsort()
    eigs = eigs[idx]
    eigv = eigv[:,idx]

    return (eigs, eigv)


def free_electron(system, cplx):

    (eigs, eigv) = diag_sorted(system.T)
    if cplx:
        trial_type = complex
    else:
        trial_type = float
    psi_trial = numpy.array([eigv[:,:system.nup], eigv[:,:system.ndown]],
                            dtype=trial_type)

    return (psi_trial, eigs)


def uhf(system, cplx, ueff, nit_max=100):

    nmf = (system.nup + system.ndown) / system.nbasis
    # initial guess for the site occupancy
    ni_up = numpy.array(nmf, system.nbasis)
    ni_down = numpy.array(nmf, system.nbasis)
    kp = kpoints(system)
    mu = chemical_potential(system, ni_up, ni_down)
    e_up = numpy.zeros(system.nbasis**2)
    e_down = numpy.zeros(system.nbasis**2)
    ev_up = numpy.zeros(shape=(system.nbasis, system.nbasis**2))
    ev_down = numpy.zeros(shape=(system.nbasis, system.nbasis))

    for it in range(0, nit_max):
        start = 0
        end = system.nbasis
        for ik, k in enumerate(kp):
            (e_up[start:end], ev_up[:,start:end]) = diagonalise_mean_field(system, k, ni_down)
            (e_down[start:end], ev_down[:,start:end]) = diagonalise_mean_field(system, k, ni_up)
            start += system.nbasis
            end += system.nbasis
        # We have that < n_{i\sigma} > = \sum_l^{Nbands} \sum_{k<k_F}
        # |c_{il\sigma}(k)|^2 so we update these here by summing over
        # kpoints and bands we need to self consistently determine n_i and
        # the chemical potential such that \sum_i n_{i\sigma} = n_\sigma.
        # First we need to determined the chemical potential.
        ni_up_new += nav_eigv(e_up, ev_up, kpoints)
        ni_down_new += nav_eigv(e_up, ev_up, kpoints)
        mu_new = chem_pot(ni_up_new, ni_down_new, mu)
        if check_self_consistency(ni_up_new, ni_down_new, ni_up, ni_down, mu_new, mu):
            break
        else:
            ni_up = ni_up_new
            ni_down = ni_down_new
            mu = mu_new

    psi_trial = numpy.array([eigv[:,:system.nup], eigv[:,:system.ndown]],
                            dtype=trial_type)


def diagonalise_mean_field(system, k, ni, U):

    H = numpy.zeros(shape=(system.nbasis, system.nbasis))
    for (ix, i) in enumerate(system.nbasis):
        for (jx, j) in enumerate(system.nbasis):
            if (ix == jx):
                H[i, j] = U*ni[i]
            xy1 = decode_basis(nx, ny, i)
            xy2 = decode_basis(nx, ny, j)
            dij = xy1-xy2
            if sum(abs(dij)) == 1:
                H[i, j] = -t * cmath.exp(1j*numpy.dot(k, dij))
            # Take care of periodic boundary conditions
            if ((abs(dij)==[nx-1,0]).all() or (abs(dij)==[0,ny-1]).all()):
                T[i, j] += -t

    return (diag_sorted(H))


def check_self_consistency(ni_up_new, ni_down_new, ni_up_old, ni_down_old, mu_new, mu_old):

    delta_ni = sum(ni_up_new-ni_up_old+ni_down_new-ni_down_old)
    delta_mu = mu_new - mu_old
    print ("#", delta_ni, delta_mu)
    if abs(delta_ni) < 1e-8 and abs(delta_mu) < 1e-8:
        return True
    else:
        return False


def chemical_potential(n, ni_up, ni_down):

    return (
        sc.optimize.fsolve(nav-n, 0, args=(occ_up, occ_down,
                           eup, edown))[0]
    )


def nav_eigv(e, ev, kpoints, mu):

    ni = numpy.zeros(len(kpoints))
    for i in range(0, len(kpoints)):
        for ejk, evjk in zip(e, ev[i, :]):
            if ejk < mu:
                ni[i] += conj(evjk)*evjk

    return ni


def nav(mu, occ_up, occ_down, eup, edown):
    n = 0.0
    for (ni, ei) in zip(occ_op, eup):
        if ei < mu:
            n += ni
    for (ni, ei) in zip(occ_down, edown):
        if ei < mu:
            n += ni
    return n

def kpoints(system):

    kpoints = []
    kfac = 2.0*math.pi / system.nx
    for nx in range(-system.nx//2, system.nx//2):
        for ny in range(-system.ny//2, system.ny//2):
            kpoints.append([kfac*nx, kfac*ny])

    return kpoints

def calc_ni(psi, nx, ny):
    '''Calculate average occupancies'''
