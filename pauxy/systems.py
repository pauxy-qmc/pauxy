import pauxy.hubbard
import pauxy.generic

def get_system(model, dt):
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
        System class. See :ref:`pauxy.generic` or :ref:`pauxy.hubbard`.
    """
    if model['name'] == 'Hubbard':
        # sytem packages all generic information + model specific information.
        system = pauxy.hubbard.Hubbard(model, dt)
    elif model['name'] == 'Generic':
        system = pauxy.generic.Generic(model, dt)
    else:
        system = None

    return system
