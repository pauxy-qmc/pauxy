import numpy
import sys
from pauxy.trial_wavefunction.free_electron import FreeElectron
from pauxy.trial_wavefunction.uhf  import UHF
from pauxy.trial_wavefunction.hartree_fock import HartreeFock
from pauxy.trial_wavefunction.multi_determinant import MultiDeterminant
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.utils.io import read_qmcpack_wfn_hdf, get_input_value

def get_trial_wavefunction(system, options={}, mf=None,
                           comm=None, verbose=0):
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
    wfn_type = options.get('name', 'MultiSlater')
    if wfn_type == 'MultiSlater':
        if wfn_file is not None:
            if verbose:
                print("# Reading wavefunction from {}.".format(wfn_file))
            read, psi0 = read_qmcpack_wfn_hdf(wfn_file)
            thresh = options.get('threshold', None)
            if thresh is not None:
                coeff = read[0]
                ndets = len(coeff[abs(coeff)>thresh])
                if verbose:
                    print("# Discarding determinants with weight "
                          "  below {}.".format(thresh))
            else:
                ndets = options.get('ndets', None)
                if ndets is None:
                    ndets = len(read[0])
            if verbose:
                print("# Number of determinants in trial wavefunction: {}"
                      .format(ndets))
            if ndets is not None:
                wfn = []
                # Wavefunction is a tuple, immutable so have to iterate through
                for x in read:
                    wfn.append(x[:ndets])
        else:
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
            wfn = (coeffs, wfn)
        trial = MultiSlater(system, wfn, init=psi0, options=options, verbose=verbose)
        if system.name == 'Generic':
            trial.half_rotate(system, comm)
        rediag = options.get('recompute_ci', False)
        if rediag:
            if comm.rank == 0:
                if verbose:
                    print("# Recomputing trial wavefunction ci coeffs.")
                coeffs = trial.recompute_ci_coeffs(system)
            else:
                coeffs = None
            coeffs = comm.bcast(coeffs, root=0)
            trial.coeffs = coeffs
    elif wfn_type == 'hartree_fock':
        trial = HartreeFock(system, options, verbose=verbose)
    elif wfn_type == 'free_electron':
        trial = FreeElectron(system, options, verbose)
    elif wfn_type == 'UHF':
        if comm.rank == 0:
            wfn = UHF(system, options, verbose)
            psi = wfn.psi
        else:
            psi = None
        psi = comm.bcast(psi)
        nmo = psi.shape[0]
        nel = psi.shape[1]
        trial = MultiSlater(system, (numpy.array([1.0]), psi.reshape(1,nmo,nel)), options=options, verbose=verbose)
    else:
        print("Unknown trial wavefunction type.")
        sys.exit()

    return trial
