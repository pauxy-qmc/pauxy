"""Helper Routines for setting up a calculation"""
# todo : handle more gracefully.
import time
import numpy
import json
import warnings
import h5py
import sys
try:
    from mpi4py import MPI
    parallel = True
except ImportError:
    warnings.warn('mpi4py not found.')
    parallel = False
import pauxy.cpmc
import pauxy.utils


def init(input_file, verbose=False):
    if parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
    else:
        comm = FakeComm()
        rank = 0
        nprocs = 1
    if rank == 0:
        if verbose:
            print('# Initialising PAUXY simulation from %s'%input_file)
        with open(input_file) as inp:
            options = json.load(inp)
        inp.close()
        # sometimes python is beautiful
        if verbose:
            print('# Running on %s core%s'%(nprocs, 's' if nprocs > 1 else ''))
    else:
        options = None
    options = comm.bcast(options, root=0)
    seed = options['qmc_options'].get('rng_seed', None)
    if seed is None:
        # only set "random" part of seed on parent processor so we can reproduce
        # results in when running in parallel.
        if rank == 0:
            seed = numpy.array([numpy.random.randint(0, 1e8)], dtype='i4')
            # Can't directly json serialise numpy arrays
            options['qmc_options']['rng_seed'] = seed[0].item()
        else:
            seed = numpy.empty(1, dtype='i4')
        if comm is not None:
            comm.Bcast(seed, root=0)
        seed = seed[0]
    seed = seed + rank
    numpy.random.seed(seed)

    return (options, comm)


def setup_parallel(options, comm=None, verbose=False):
    """Wrapper routine for initialising simulation

    Parameters
    ----------
    input_file : json file.
        Simulation input file.

    Returns
    -------
    state : :class:`pauxy.state.State`
        Simulation state.
    """
    if comm.Get_rank() == 0:
        cpmc = pauxy.cpmc.CPMC(options.get('model'),
                               options.get('qmc_options'),
                               options.get('estimates'),
                               options.get('trial_wavefunction'),
                               options.get('propagator', {}),
                               parallel=True,
                               verbose=verbose)
    else:
        cpmc = None
    cpmc = comm.bcast(cpmc, root=0)
    cpmc.init_time = time.time()
    if cpmc.trial.error:
        warnings.warn('Error in constructing trial wavefunction. Exiting')
        sys.exit()
    cpmc.rank = comm.Get_rank()
    cpmc.nprocs = comm.Get_size()
    cpmc.root = cpmc.rank == 0
    # We can't serialise '_io.BufferWriter' object, so just delay initialisation
    # of estimators object to after MPI communication.
    # TODO: Do this more gracefully.
    cpmc.qmc.nwalkers = int(cpmc.qmc.nwalkers/cpmc.nprocs)
    if cpmc.qmc.nwalkers == 0:
        # This should occur on all processors so we don't need to worry about
        # race conditions / mpi4py hanging.
        if cpmc.root:
            warnings.warn('Not enough walkers for selected core count. There '
                          'must be at least one walker per core set in the '
                          'input file. Exiting.')
        sys.exit()

    # TODO: Return cpmc and psi and run from another routine.
    cpmc.estimators = (
        pauxy.estimators.Estimators(options.get('estimates'),
                                    cpmc.root,
                                    cpmc.qmc,
                                    cpmc.system,
                                    cpmc.trial,
                                    cpmc.propagators.BT_BP)
    )
    cpmc.psi = pauxy.walker.Walkers(cpmc.system,
                                    cpmc.trial,
                                    cpmc.qmc.nwalkers,
                                    cpmc.estimators.nprop_tot,
                                    cpmc.estimators.nbp)
    if comm.Get_rank() == 0:
        json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')
        json_string = json.dumps(pauxy.utils.serialise(cpmc, verbose=1),
                                 sort_keys=False, indent=4)
        cpmc.estimators.h5f.create_dataset('metadata',
                                           data=numpy.array([json_string],
                                                            dtype=object),
                                           dtype=h5py.special_dtype(vlen=str))

    return cpmc

class FakeComm:
    """Fake MPI communicator class to reduce logic."""

    def __init__(self):
        pass

    def Barrier(self):
        pass
    def Get_rank(self):
        return 0
    def Get_size(self):
        return 1
    def Gather(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf
    def Bcast(self, sendbuf, root=0):
        return sendbuf
    def bcast(self, sendbuf, root=0):
        return sendbuf
    def isend(self, sendbuf, dest=None, tag=None):
        return FakeReq()
    def recv(self, sendbuf, root=0):
        pass
    def Reduce(self, sendbuf, recvbuf, op=None):
        recvbuf[:] = sendbuf

class FakeReq:

    def __init__(self):
        pass
    def wait(self):
        pass
