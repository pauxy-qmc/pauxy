from pauxy.systems.hubbard import Hubbard
from pauxy.systems.generic import Generic, read_integrals
from pauxy.systems.ueg import UEG
from pauxy.utils.mpi import get_shared_array, have_shared_mem

def get_system(sys_opts=None, verbose=0, chol_cut=1e-5, comm=None):
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
    elif sys_opts['name'] == 'Generic':
        filename = sys_opts.get('integrals', False)
        hcore, chol, enuc = get_generic_integrals(filename, comm=comm)
        system = Generic(inputs=sys_opts, h1e=hcore, chol=chol, ecore=enuc,
                         verbose=verbose)
        # system = None
    elif sys_opts['name'] == 'UEG':
        system = UEG(sys_opts, verbose)
    else:
        system = None

    return system


def get_generic_integrals(filename, comm=None):
    shmem = have_shared_mem(comm)
    if shmem:
        if comm.rank == 0:
            hcore, chol, enuc, nelec = read_integrals(filename)
            hc_shape = hcore.shape
            ch_shape = chol.shape
            dtype = chol.dtype
        else:
            hc_shape = None
            ch_shape = None
            dtype = None
            enuc = None
        shape = comm.bcast(hc_shape, root=0)
        dtype = comm.bcast(dtype, root=0)
        enuc = comm.bcast(enuc, root=0)
        hcore_shmem = get_shared_array(comm, shape, dtype)
        if comm.rank == 0:
            hcore_shmem[:] = hcore[:]
        comm.Barrier()
        shape = comm.bcast(ch_shape, root=0)
        chol_shmem = get_shared_array(comm, shape, dtype)
        if comm.rank == 0:
            chol_shmem[:] = chol[:]
        comm.Barrier()
        return hcore_shmem, chol_shmem, enuc
    else:
        hcore, chol, enuc, nelec = read_integrals(filename, sparse)
        return hcore, chol, enuc
