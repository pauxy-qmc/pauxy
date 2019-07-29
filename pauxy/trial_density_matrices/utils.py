from pauxy.trial_density_matrices.onebody import OneBody

def get_trial_density_matrices(comm, options, system, cplx, parallel, beta, dt, verbose=False):
    """Wrapper to select trial wavefunction class.

    Parameters
    ----------
    options : dict
        Trial wavefunction input options.
    system : class
        System class.
    cplx : bool
        If true then trial wavefunction will be complex.
    parallel : bool
        If true then running in parallel.

    Returns
    -------
    trial : class or None
        Trial wavfunction class.
    """
    trial_type = options.get('name', 'one_body')
    if trial_type == 'one_body_mod':
        trial = OneBody(comm, options, system, beta, dt, H1=system.h1e_mod, verbose=verbose)
    elif trial_type == 'one_body':
        trial = OneBody(comm, options, system, beta, dt, verbose=verbose)
    else:
        trial = None

    return trial
