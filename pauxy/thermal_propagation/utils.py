"""Routines for performing propagation of a walker"""

from pauxy.thermal_propagation.generic import GenericContinuous
from pauxy.thermal_propagation.hubbard import ThermalDiscrete
from pauxy.thermal_propagation.planewave import PlaneWave


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
    if system.name == "Hubbard":
        hs_type = options.get('hubbard_stratonovich', 'discrete')
    elif system.name == "UEG":
        hs_type = options.get('hubbard_stratonovich', 'plane_wave')
    elif system.name == "Generic":
        hs_type = options.get('hubbard_stratonovich', 'continuous')
    if hs_type == 'discrete':
        propagator = ThermalDiscrete(options, qmc, system, trial, verbose)
    elif hs_type == "plane_wave":
        propagator = PlaneWave(options, qmc, system, trial, verbose)
    elif hs_type == "continuous":
        propagator = GenericContinuous(options, qmc, system, trial, verbose)
    else:
        propagator = None

    return propagator
