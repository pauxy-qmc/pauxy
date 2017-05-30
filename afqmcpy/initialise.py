import sys
import json
import time
from mpi4py import MPI
import afqmcpy.state
import afqmcpy.qmc
import afqmcpy.walker

def initialise(input_file):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    if rank == 0:
        with open(input_file) as inp:
            options = json.load(inp)
        print('# Initialising AFQMCPY simulation from %s'%input_file)
        # sometimes python is beautiful
        print('# Running on %s core%s'%(nprocs, 's' if nprocs > 1 else ''))
    state = afqmcpy.state.State(options['model'], options['qmc_options'])
    state.rank = rank
    state.nprocs = nprocs
    state.root = state.rank == 0
    if state.root:
        state.write_json()
    state.nwalkers = int(state.nwalkers/nprocs)
    psi0 = [afqmcpy.walker.Walker(1, state.system, state.trial.psi, w) for w in
            range(state.nwalkers)]
    afqmcpy.qmc.do_qmc(state, psi0, comm)

def finalise():

    print ("# End Time: %s"%time.asctime())
