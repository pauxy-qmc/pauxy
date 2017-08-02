import sys
import json
import time
import numpy
from mpi4py import MPI
import warnings
import afqmcpy.state
import afqmcpy.qmc
import afqmcpy.walker
import afqmcpy.estimators

# TODO: change module name
def initialise(input_file):
    """Wrapper routine for initialising simulation

    Parameters
    ----------
    input_file : json file.
        Simulation input file.

    Returns
    -------
    state : :class:afqmcpy.state.State`
        Simulation state.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    if rank == 0:
        with open(input_file) as inp:
            options = json.load(inp)
        print('# Initialising AFQMCPY simulation from %s'%input_file)
        # sometimes python is beautiful
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
            # Can't serialise numpy arrays
            options['qmc_options']['rng_seed'] = seed[0].item()
        else:
            seed = numpy.empty(1, dtype='i4')
        comm.Bcast(seed, root=0)
        seed = seed[0]
    seed = seed + rank
    numpy.random.seed(seed)
    if rank == 0:
        state = afqmcpy.state.State(options.get('model'),
                                    options.get('qmc_options'),
                                    options.get('estimates'),
                                    options.get('trial_wavefunction'))
    else:
        state = None
    state = comm.bcast(state, root=0)
    state.rank = rank
    state.nprocs = nprocs
    state.root = state.rank == 0
    # We can't serialise '_io.BufferWriter' object, so just delay initialisation
    # of estimators object to after MPI communication.
    # TODO: Do this more gracefully.
    state.qmc.nwalkers = int(state.qmc.nwalkers/nprocs)
    if state.qmc.nwalkers == 0:
        # This should occur on all processors so we don't need to worry about
        # race conditions / mpi4py hanging.
        if state.root:
            warnings.warn('Not enough walkers for selected core count. There '
                          'must be at least one walker per core set in the '
                          'input file. Exiting.')
        sys.exit()
    state.estimators = afqmcpy.estimators.Estimators(options.get('estimates'),
                                                     state.root,
                                                     state.uuid,
                                                     state.qmc.dt,
                                                     state.system.nbasis,
                                                     state.qmc.nwalkers,
                                                     state.json_string)
    if state.trial.__class__.__name__ == 'MultiDeterminant':
        psi0 = [afqmcpy.walker.MultiDetWalker(1, state.system, state.trial)
                for w in range(state.qmc.nwalkers)]
    else:
        psi0 = [afqmcpy.walker.Walker(1, state.system, state.trial.psi, w)
                for w in range(state.qmc.nwalkers)]
    (state, psi) = afqmcpy.qmc.do_qmc(state, psi0, comm)
    # TODO: Return state and psi and run from another routine.
    return state

def finalise(state, init_time):

    if state.root:
        print ("# End Time: %s"%time.asctime())
        print ("# Running time : %.6f seconds"%(time.time()-init_time))
