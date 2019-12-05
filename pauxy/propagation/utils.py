"""Routines for performing propagation of a walker"""

from pauxy.propagation.continuous import Continuous
from pauxy.propagation.hubbard import HirschSpin
from pauxy.propagation.hubbard_holstein import HirschSpinDMC

# TODO: Fix for discrete transformation.
def get_propagator_driver(system, trial, qmc, options={}, verbose=False):
    hs = options.get('hubbard_stratonovich', 'continuous')
    if 'discrete' in hs:
        return get_discrete_propagator(options, qmc, system, trial, verbose)
    else:
        return Continuous(system, trial, qmc, options=options, verbose=verbose)

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
        propagator = HirschSpin(system, trial, qmc,
                                options=options, verbose=verbose)
    elif system.name == "HubbardHolstein":
        propagator = HirschSpinDMC(system, trial, qmc,
                                options=options, verbose=verbose)
    else:
        propagator = None

    return propagator
