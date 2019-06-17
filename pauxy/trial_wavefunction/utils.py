from pauxy.trial_wavefunction.free_electron import FreeElectron
from pauxy.trial_wavefunction.uhf  import UHF
from pauxy.trial_wavefunction.hartree_fock import HartreeFock
from pauxy.trial_wavefunction.multi_determinant import MultiDeterminant
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.utils.from_pyscf import get_pyscf_wfn

def get_trial_wavefunction(system, options={}, mf=None, parallel=False, verbose=False):
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
    if mf is not None:
        wfn, coeffs = get_pyscf_wfn(system, mf)
        trial = MultiSlater(system, wfn, coeffs)
    elif options['name'] == 'free_electron':
        trial = FreeElectron(system, True, options, parallel, verbose)
    elif options['name'] == 'UHF':
        trial = UHF(system, True, options, parallel, verbose)
    elif options['name'] == 'multi_slater':
        trial = HartreeFock(system, True, options, parallel, verbose)

    return trial
