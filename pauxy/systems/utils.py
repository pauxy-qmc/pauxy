from pauxy.systems.hubbard import Hubbard
from pauxy.systems.hubbard_holstein import HubbardHolstein
from pauxy.systems.generic import Generic
from pauxy.systems.ueg import UEG

def get_system(sys_opts=None, verbose=0, chol_cut=1e-5):
    """Wrapper to select system class

    Parameters
    ----------
    sys_opts : dict
        System input options.
    verbose : bool
        Output verbosity.

    Returns
    -------
    system : object
        System class.
    """
    if sys_opts['name'] == 'Hubbard':
        system = Hubbard(sys_opts, verbose)
    elif sys_opts['name'] == 'HubbardHolstein':
        system = HubbardHolstein(sys_opts, verbose)
    elif sys_opts['name'] == 'Generic':
        system = Generic(inputs=sys_opts, verbose=verbose)
    elif sys_opts['name'] == 'UEG':
        system = UEG(sys_opts, verbose)
    else:
        system = None

    return system
