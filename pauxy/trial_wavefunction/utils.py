from pauxy.trial_wavefunction.free_electron import FreeElectron
from pauxy.trial_wavefunction.uhf  import UHF
from pauxy.trial_wavefunction.hartree_fock import HartreeFock
from pauxy.trial_wavefunction.multi_determinant import MultiDeterminant

def get_trial_wavefunction(options, system, cplx, parallel, verbose=False):
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
    if options['name'] == 'free_electron':
        trial = FreeElectron(system, cplx, options, parallel, verbose)
    elif options['name'] == 'UHF':
        trial = UHF(system, cplx, options, parallel, verbose)
    elif options['name'] == 'multi_determinant':
        trial = MultiDeterminant(system, cplx, options, parallel, verbose)
    elif options['name'] == 'hartree_fock':
        trial = HartreeFock(system, cplx, options, parallel, verbose)
    else:
        trial = None

    return trial
