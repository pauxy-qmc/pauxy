from pauxy.trial_density_matrices.onebody import OneBody

def get_trial_density_matrices(options, system, cplx, parallel, beta, dt, verbose=False):
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
    if options['name'] == 'one_body':
        trial = OneBody(options, system, beta, dt, verbose=verbose)
    if options['name'] == 'one_body_mod':
        trial = OneBody(options, system, beta, dt, system.h1e_mod, verbose=verbose)
    else:
        trial = OneBody(options, system, beta, dt, verbose=verbose)
        # trial = None

    return trial
