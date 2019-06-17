from pauxy.systems.hubbard import Hubbard
from pauxy.systems.generic import Generic
from pauxy.systems.ueg import UEG
from pauxy.utils.from_pyscf import integrals_from_scf


def get_system(sys_opts=None, mf=None, verbose=0, chol_cut=1e-5):
    """Wrapper to select system class

    Parameters
    ----------
    sys_opts : dict
        System input options.
    mf : object
        pyscf object. Optional. Default: None.
    verbose : bool
        Output verbosity.

    Returns
    -------
    system : object
        System class.
    """
    if mf is not None:
        h1e, chol, ecore, oao = (
                integrals_from_scf(mf, verbose=verbose, chol_cut=chol_cut)
                )
        nb = h1e.shape[0]
        system = Generic(nelec=mf.mol.nelec, h1e=h1e,
                         chol=chol.reshape((-1,nb,nb)),
                         ecore=ecore, verbose=verbose)
        system.oao = oao
    elif sys_opts.get('pyscf_chk', None) is not None:
        if sys_opts['name'] == 'Hubbard':
            system = Hubbard(sys_opts, verbose)
        elif sys_opts['name'] == 'Generic':
            system = Generic(sys_opts, verbose)
        elif sys_opts['name'] == 'UEG':
            system = UEG(sys_opts, verbose)
        else:
            system = None

    return system
