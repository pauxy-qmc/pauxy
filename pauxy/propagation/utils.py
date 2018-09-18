"""Routines for performing propagation of a walker"""

from pauxy.propagation.hubbard import HirschSpin, HubbardContinuous
from pauxy.propagation.planewave import PlaneWave
from pauxy.propagation.generic import Generic


def get_continuous_propagator(options, qmc, system, trial, verbose=False):
    """Wrapper to select propagator class.

    Parameters
    ----------
    options : dict
        Propagator input options.
    qmc : :class:`pauxy.qmc.QMCOpts` class
        Trial wavefunction input options.
    system : class
        System class.
    trial : class
        Trial wavefunction object.

    Returns
    -------
    propagator : class or None
        Propagator object.
    """
    if system.name == "UEG":
        propagator = PlaneWave(options, qmc, system, trial, verbose)
    elif system.name == "Hubbard":
        propagator = HubbardContinuous(options, qmc, system, trial, verbose)
    elif system.name == "Generic":
        propagator = GenericContinuous(options, qmc, system, trial, verbose)
    else:
        propagator = None

    return propagator

def get_discrete_propagator(options, qmc, system, trial, verbose=False):
    """Wrapper to select propagator class.

    Parameters
    ----------
    options : dict
        Propagator input options.
    qmc : :class:`pauxy.qmc.QMCOpts` class
        Trial wavefunction input options.
    system : class
        System class.
    trial : class
        Trial wavefunction object.

    Returns
    -------
    propagator : class or None
        Propagator object.
    """
    hs_type = options.get('hubbard_stratonovich', 'discrete')
    if system.name == "Hubbard":
        propagator = HirschSpin(options, qmc, system, trial, verbose)
    else:
        propagator = None

    return propagator
