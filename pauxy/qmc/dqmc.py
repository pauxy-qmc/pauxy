import sys
import json
import time
import numpy
import uuid
from pauxy.utils.io import  to_json, serialise, get_input_value
from pauxy.qmc.options import QMCOpts
from pauxy.qmc.utils import set_rng_seed
from pauxy.systems.utils import get_system
from pauxy.utils.mpi import get_shared_comm
from pauxy.dqmc_updates.utils import get_update_driver
from pauxy.estimators.simple import Energy
from pauxy.estimators.thermal import greens_function_qr_strat


class DQMC(object):
    """DQMC driver.

    Parameters
    ----------

    Attributes
    ----------
    """
    def __init__(self, comm, options=None, system=None, verbose=False):
        # 1. Environment attributes
        if verbose is not None:
            self.verbosity = verbose
            if comm.rank != 0:
                self.verbosity = 0
            verbose = verbose > 0 and comm.rank == 0
        if comm.rank == 0:
            self.uuid = str(uuid.uuid1())
            get_sha1 = options.get('get_sha1', True)
            if get_sha1:
                try:
                    self.sha1, self.branch = get_git_revision_hash()
                except:
                    self.sha1 = 'None'
                    self.branch = 'None'
            else:
                self.sha1 = 'None'
                self.branch = 'None'
            if verbose:
                self.sys_info = get_sys_info(self.sha1, self.branch, self.uuid, comm.size)
        self.root = comm.rank == 0
        self.rank = comm.rank
        self._init_time = time.time()
        self.run_time = time.asctime()
        self.shared_comm = get_shared_comm(comm, verbose=verbose)
        # 2. Calculation objects.
        if system is not None:
            self.system = system
        else:
            sys_opts = get_input_value(options, 'model',
                                       default={},
                                       alias=['system'],
                                       verbose=self.verbosity>1)
            self.system = get_system(sys_opts, verbose=verbose,
                                     comm=self.shared_comm)

        qmc_opt = get_input_value(options, 'qmc', default={},
                                  alias=['qmc_options'],
                                  verbose=self.verbosity>1)
        self.qmc = QMCOpts(qmc_opt, self.system,
                           verbose=self.verbosity>1)
        self.qmc.rng_seed = set_rng_seed(self.qmc.rng_seed, comm)
        self.qmc.ntime_slices = int(round(self.qmc.beta/self.qmc.dt))
        prop_opt = get_input_value(options, 'propagator', default={},
                                   alias=['update','prop'],
                                   verbose=self.verbosity>1)
        self.propagators = get_update_driver(self.system,
                                             self.qmc.dt,
                                             self.qmc.ntime_slices,
                                             options=prop_opt,
                                             verbose=verbose)
        self.qmc.nstblz = min(self.qmc.nstblz, self.propagators.stack.stack_size)
        self.tsetup = time.time() - self._init_time
        est_opts = get_input_value(options, 'estimators', default={},
                                   alias=['estimates','estimator'],
                                   verbose=self.verbosity>1)
        self.estimators = Energy(self.qmc.ntime_slices,
                                 self.qmc.nsteps,
                                 self.root,
                                 options=est_opts,
                                 verbose=verbose)
        if comm.rank == 0:
            json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')
            json_string = to_json(self)
            self.estimators.json_string = json_string
            self.estimators.dump_metadata()
            if verbose:
                self.estimators.estimators['mixed'].print_key()
                self.estimators.estimators['mixed'].print_header()

    def run(self, comm):
        nslice = self.qmc.ntime_slices
        blocks = self.qmc.nblocks
        nsteps = self.qmc.nsteps
        nstblz = self.qmc.nstblz
        prop = self.propagators
        G = greens_function_qr_strat(prop.stack, slice_ix=nslice)
        estimators = self.estimators
        for block in range(blocks):
            for step in range(nsteps):
                for islice in range(nslice):
                    G = prop.propagate_greens_function(G, islice)
                    # print(G[0,0,0])
                    phase = prop.update(G, islice)
                    # print(G[0,1,1])
                    if islice % nstblz == nstblz -1 and islice != 0:
                        G = greens_function_qr_strat(prop.stack,
                                                     slice_ix=islice)
                    # print(G[0,0,0])
                    estimators.update_step(self.system, G, 1.0, phase)
                    # assert False
                estimators.update_block()
                # assert False
            estimators.write(comm, block)
