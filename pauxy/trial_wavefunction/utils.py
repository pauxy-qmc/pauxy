import numpy
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
        trial = MultiSlater(system, wfn, coeffs,
                            parallel=parallel, verbose=verbose)
    elif options['name'] == 'hartree_fock':
        na = system.nup
        nb = system.ndown
        wfn = numpy.zeros((system.nbasis,system.nup+system.ndown),
                          dtype=numpy.complex128)
        coeffs = numpy.array([1.0+0j])
        I = numpy.identity(system.nbasis, dtype=numpy.complex128)
        wfn[:,:na] = I[:,:na]
        wfn[:,na:] = I[:,:nb]
        trial = MultiSlater(system, wfn, coeffs, options=options,
                            parallel=parallel, verbose=verbose)
    elif options['name'] == 'free_electron':
        trial = FreeElectron(system, True, options, parallel, verbose)
    elif options['name'] == 'UHF':
        trial = UHF(system, True, options, parallel, verbose)
    else:
        trial = None

    return trial
