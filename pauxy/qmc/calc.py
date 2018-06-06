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
    parallel = False
from pauxy.qmc.afqmc import AFQMC
import pauxy.utils


def init_communicator():
    if parallel:
        comm = MPI.COMM_WORLD
    else:
        comm = FakeComm()
    return comm


def read_input(input_file, comm, verbose=False):
    """Helper function to parse input file and setup parallel calculation.

    Parameters
    ----------
    input_file : string
        Input filename.
    verbose : bool
        If true print out set up information.

    Returns
    -------
    options : dict
        Python dict of input options.
    comm : MPI communicator
        Communicator object. If mpi4py is not installed then we return a fake
        communicator.
    """
    if comm.rank == 0:
        if verbose:
            print('# Initialising PAUXY simulation from %s'%input_file)
        with open(input_file) as inp:
            options = json.load(inp)
        inp.close()
        # sometimes python is beautiful
        if verbose:
            print('# Running on %s core%s.'%(comm.size, 's' if comm.size > 1 else ''))
    else:
        options = None
    options = comm.bcast(options, root=0)

    return options


def set_rng_seed(options, comm):
    seed = options['qmc_options'].get('rng_seed', None)
    if seed is None:
        # only set "random" part of seed on parent processor so we can reproduce
        # results in when running in parallel.
        if comm.rank == 0:
            seed = numpy.array([numpy.random.randint(0, 1e8)], dtype='i4')
            # Can't directly json serialise numpy arrays
            options['qmc_options']['rng_seed'] = seed[0].item()
        else:
            seed = numpy.empty(1, dtype='i4')
        comm.Bcast(seed, root=0)
        seed = seed[0]
    seed = seed + comm.rank
    numpy.random.seed(seed)


def setup_parallel(options, comm=None, verbose=False):
    """Wrapper routine for initialising simulation

    Parameters
    ----------
    options : dict
        Input options.
    comm : MPI communicator
        MPI communicator object.
    verbose : bool
        If true print out set up information.

    Returns
    -------
    afqmc : :class:`pauxy.afqmc.CPMC`
        CPMC driver.
    """
    if comm.Get_rank() == 0:
        afqmc = pauxy.qmc.afqmc.AFQMC(options.get('model'),
                                      options.get('qmc_options'),
                                      options.get('estimates'),
                                      options.get('trial_wavefunction'),
                                      options.get('propagator', {}),
                                      parallel=True,
                                      verbose=verbose)
    else:
        afqmc = None
    afqmc = comm.bcast(afqmc, root=0)
    afqmc.init_time = time.time()
    if afqmc.trial.error:
        warnings.warn('Error in constructing trial wavefunction. Exiting')
        sys.exit()
    afqmc.rank = comm.Get_rank()
    afqmc.nprocs = comm.Get_size()
    afqmc.root = afqmc.rank == 0
    # We can't serialise '_io.BufferWriter' object, so just delay initialisation
    # of estimators object to after MPI communication.
    # Simpler to just ensure a fixed number of walkers per core.
    afqmc.qmc.nwalkers = int(afqmc.qmc.nwalkers/afqmc.nprocs)
    if afqmc.qmc.nwalkers == 0:
        # This should occur on all processors so we don't need to worry about
        # race conditions / mpi4py hanging.
        if afqmc.root:
            warnings.warn('Not enough walkers for selected core count. There '
                          'must be at least one walker per core set in the '
                          'input file. Exiting.')
        sys.exit()

    afqmc.estimators = (
        pauxy.estimators.Estimators(options.get('estimates'),
                                    afqmc.root,
                                    afqmc.qmc,
                                    afqmc.system,
                                    afqmc.trial,
                                    afqmc.propagators.BT_BP)
    )
    afqmc.psi = pauxy.walker.Walkers(afqmc.system,
                                    afqmc.trial,
                                    afqmc.qmc.nwalkers,
                                    afqmc.estimators.nprop_tot,
                                    afqmc.estimators.nbp)
    if comm.Get_rank() == 0:
        json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')
        json_string = json.dumps(pauxy.utils.serialise(afqmc, verbose=1),
                                 sort_keys=False, indent=4)
        afqmc.estimators.h5f.create_dataset('metadata',
                                           data=numpy.array([json_string],
                                                            dtype=object),
                                           dtype=h5py.special_dtype(vlen=str))

    return afqmc

class FakeComm:
    """Fake MPI communicator class to reduce logic."""

    def __init__(self):
        self.rank = 0
        self.size = 1

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
