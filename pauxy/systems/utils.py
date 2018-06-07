from pauxy.systems.hubbard import Hubbard
from pauxy.systems.generic import Generic


def get_system(model, dt, verbose):
    """Wrapper to select system class

    Parameters
    ----------
    model : dict
        Model input options.
    dt : float
        Timestep.

    Returns
    -------
    system : class
        System class. See :ref:`pauxy.systems.generic` or
        :ref:`pauxy.system.hubbard`.
    """
    if model['name'] == 'Hubbard':
        system = Hubbard(model, dt, verbose)
    elif model['name'] == 'Generic':
        system = Generic(model, dt, verbose)
    else:
        system = None

    return system
