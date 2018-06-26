"""Routines for performing propagation of a walker"""

from pauxy.propagation.hubbard import Discrete, Continuous
from pauxy.propagation.planewave import PlaneWave
from pauxy.propagation.generic import GenericContinuous


def get_propagator(options, qmc, system, trial, verbose=False):
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
    print ("XXX")
    hs_type = options.get('hubbard_stratonovich', 'discrete')
    if hs_type == 'discrete':
        propagator = Discrete(options, qmc, system, trial, verbose)
    elif hs_type == "hubbard_continuous":
        propagator = Continuous(options, qmc, system, trial, verbose)
    elif hs_type == "continuous":
        propagator = GenericContinuous(options, qmc, system, trial, verbose)
    elif hs_type == "plane_wave":
        propagator = PlaneWave(options, qmc, system, trial, verbose)
    else:
        propagator = None

    return propagator
