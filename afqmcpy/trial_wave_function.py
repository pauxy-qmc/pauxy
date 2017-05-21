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

    for it in range(0, nit_max):
        for ik, k in enumerate(kp):
            (e_up, ev_up) = diagonalise_mean_field(system, k, ni_down):
            (e_down, ev_down) = diagonalise_mean_field(system, k, ni_up):
            # We have that < n_{i\sigma} > = \sum_l^{Nbands} \sum_{k<k_F}
            # |c_{il\sigma}(k)|^2 so we update these here by summing over
            # kpoints and bands we need to self consistently determine n_i and
            # the chemical potential such that \sum_i n_{i\sigma} = n_\sigma.
            # First we need to determined the chemical potential.
            mu = chemical_potential(system, ni_up, ni_down)
            (ni_up_new, nav_up_new) += nav(ev_up, mu)
            (ni_down_new, nav_down_new) += nav(ev_down, mu)
        if check_self_consistency(ni_up_new, ni_down_new):
            break
        else:
            ni_up = ni_up_new
            ni_down = ni_down_new

    psi_trial = numpy.array([eigv[:,:system.nup], eigv[:,:system.ndown]],
                            dtype=trial_type)


def construct_hamiltonian(system, k, ni):
def kpoints(system):

    kpoints = []
    kfac = 2.0*math.pi / system.nx
    for nx in range(-system.nx//2, system.nx//2):
        for ny in range(-system.ny//2, system.ny//2):
            kpoints.append([kfac*nx, kfac*ny])

    return kpoints

def calc_ni(psi, nx, ny):
    '''Calculate average occupancies'''
