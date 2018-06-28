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
from pauxy.qmc.thermal_afqmc import ThermalAFQMC
from pauxy.estimators.handler import Estimators
from pauxy.utils.io import  to_json
from pauxy.utils.misc import serialise
from pauxy.walkers.handler import Walkers
from pauxy.qmc.comm import FakeComm


def init_communicator():
    if parallel:
        comm = MPI.COMM_WORLD
    else:
        comm = FakeComm()
    return comm

def setup_calculation(input_options):
    comm = init_communicator()
    if (isinstance(input_options, str)):
        options = read_input(input_options, comm, verbose=True)
    else:
        options = input_options
    set_rng_seed(options, comm)
    if comm.size > 1:
        afqmc = setup_parallel(options, comm, verbose=True)
    else:
        afqmc = get_driver(options, comm)
    return (afqmc, comm)

def get_driver(options, comm):
    beta = options.get('qmc_options').get('beta', None)
    if beta is not None:
        afqmc = ThermalAFQMC(options.get('model'),
                             options.get('qmc_options'),
                             options.get('estimates', {}),
                             options.get('trial', {}),
                             options.get('propagator', {}),
                             parallel=comm.size>1,
                             verbose=True)
    else:
        afqmc = AFQMC(options.get('model'),
                      options.get('qmc_options'),
                      options.get('estimates', {}),
                      options.get('trial_wavefunction', {}),
                      options.get('propagator', {}),
                      parallel=comm.size>1,
                      verbose=True)
    return afqmc

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
        afqmc = get_driver(options, comm)
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
        Estimators(options.get('estimates', {}),
                   afqmc.root,
                   afqmc.qmc,
                   afqmc.system,
                   afqmc.trial,
                   afqmc.propagators.BT_BP)
    )
    afqmc.psi = Walkers(afqmc.system,
                        afqmc.trial,
                        afqmc.qmc.nwalkers,
                        afqmc.estimators.nprop_tot,
                        afqmc.estimators.nbp)
    if comm.rank() == 0:
        json_string = to_json(afqmc)
        afqmc.estimators.json_string = json_string
        afqmc.estimators.dump_metadata()

    return afqmc
