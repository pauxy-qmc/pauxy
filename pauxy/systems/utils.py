import numpy
import sys
from pauxy.systems.hubbard import Hubbard
from pauxy.systems.generic import Generic, read_integrals, construct_h1e_mod
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
        filename = sys_opts.get('integrals', None)
        if filename is None:
            if comm.rank == 0:
                print("# Error: integrals not specfied.")
                sys.exit()
        nup, ndown = sys_opts.get('nup'), sys_opts.get('ndown')
        if nup is None or ndown is None:
            if comm.rank == 0:
                print("# Error: Number of electrons not specified.")
                sys.exit()
        nelec = (nup, ndown)
        hcore, chol, h1e_mod, enuc = get_generic_integrals(filename,
                                                           comm=comm,
                                                           verbose=verbose)
        system = Generic(h1e=hcore, chol=chol, ecore=enuc,
                         h1e_mod=h1e_mod, nelec=nelec,
                         verbose=verbose)
    elif sys_opts['name'] == 'UEG':
        system = UEG(sys_opts, verbose)
    else:
        system = None

    return system

def get_generic_integrals(filename, comm=None, verbose=False):
    shmem = have_shared_mem(comm)
    if verbose:
        print("# Have shared memory: {}".format(shmem))
    if shmem:
        if comm.rank == 0:
            hcore, chol, enuc = read_integrals(filename)
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
        hcore_shmem = get_shared_array(comm, (2,)+shape, dtype)
        if comm.rank == 0:
            hcore_shmem[0] = hcore[:]
            hcore_shmem[1] = hcore[:]
        comm.Barrier()
        shape = comm.bcast(ch_shape, root=0)
        chol_shmem = get_shared_array(comm, shape, dtype)
        if comm.rank == 0:
            chol_shmem[:] = chol[:]
        comm.Barrier()
        h1e_mod_shmem = get_shared_array(comm, hcore_shmem.shape, dtype)
        if comm.rank == 0:
            construct_h1e_mod(chol_shmem, hcore_shmem, h1e_mod_shmem)
        comm.Barrier()
        return hcore_shmem, chol_shmem, h1e_mod_shmem, enuc
    else:
        hcore, chol, enuc = read_integrals(filename)
        h1 = numpy.array([hcore, hcore])
        h1e_mod = numpy.zeros(h1.shape, dtype=h1.dtype)
        construct_h1e_mod(chol, h1, h1e_mod)
        return h1, chol, h1e_mod, enuc
