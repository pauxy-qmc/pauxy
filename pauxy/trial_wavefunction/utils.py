import numpy
import sys
from pauxy.trial_wavefunction.free_electron import FreeElectron
from pauxy.trial_wavefunction.uhf  import UHF
from pauxy.trial_wavefunction.lang_firsov  import LangFirsov
from pauxy.trial_wavefunction.coherent_state  import CoherentState
from pauxy.trial_wavefunction.hartree_fock import HartreeFock
from pauxy.trial_wavefunction.multi_determinant import MultiDeterminant
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.utils.io import read_qmcpack_wfn_hdf, get_input_value

def get_trial_wavefunction(system, options={}, mf=None,
                           parallel=False, verbose=0):
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
    wfn_file = get_input_value(options, 'filename', default=None,
                               alias=['wavefunction_file'], verbose=verbose)
    if wfn_file is not None:
        if verbose:
            print("# Reading wavefunction from {}.".format(wfn_file))
        read, psi0 = read_qmcpack_wfn_hdf(wfn_file)
        ndets = options.get('ndets', None)
        if ndets is not None:
            wfn = []
            for x in read:
                wfn.append(x[:ndets])
        else:
            wfn = read
        trial = MultiSlater(system, wfn, options=options,
                            parallel=parallel, verbose=verbose,
                            init=psi0)
    elif options['name'] == 'MultiSlater':
        if verbose:
            print("# Guessing RHF trial wavefunction.")
        na = system.nup
        nb = system.ndown
        wfn = numpy.zeros((1,system.nbasis,system.nup+system.ndown),
                          dtype=numpy.complex128)
        coeffs = numpy.array([1.0+0j])
        I = numpy.identity(system.nbasis, dtype=numpy.complex128)
        wfn[0,:,:na] = I[:,:na]
        wfn[0,:,na:] = I[:,:nb]
        trial = MultiSlater(system, (coeffs,wfn), options=options,
                            parallel=parallel, verbose=verbose)
    elif options['name'] == 'hartree_fock':
        trial = HartreeFock(system, True, options,
                            parallel=parallel, verbose=verbose)
    elif options['name'] == 'free_electron':
        trial = FreeElectron(system, True, options, parallel, verbose)
    elif options['name'] == 'lang_firsov':
        trial = LangFirsov(system, True, options, parallel, verbose)
    elif options['name'] == 'coherent_state':
        trial = CoherentState(system, True, options, parallel, verbose)
    elif options['name'] == 'UHF':
        trial = UHF(system, True, options, parallel, verbose)
    else:
        print("Unknown trial wavefunction type.")
        sys.exit()

    return trial
