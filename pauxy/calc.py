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
    warnings.warn('No MPI library found')
    parallel = False
import pauxy.cpmc
import pauxy.utils

def init(input_file, verbose=True):
    if parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
    else:
        comm = None
        rank = 0
        nprocs = 1
    if rank == 0:
        if verbose:
            print('# Initialising AFQMCPY simulation from %s'%input_file)
        with open(input_file) as inp:
            options = json.load(inp)
        inp.close()
        # sometimes python is beautiful
        if verbose:
            print('# Running on %s core%s'%(nprocs, 's' if nprocs > 1 else ''))
    else:
        options = None
    if comm is not None:
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

def setup_parallel(options, comm=None):
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
                                  parallel=True)
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
    cpmc.psi = pauxy.walker.Walkers(cpmc.system, cpmc.trial,
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
        print ('# Input options:')
        print (json.dumps(pauxy.utils.serialise(cpmc, verbose=0),
                          sort_keys=False, indent=4))
        print('# End of input options.')
        cpmc.estimators.estimators['mixed'].print_key()
        cpmc.estimators.estimators['mixed'].print_header()

    return cpmc
