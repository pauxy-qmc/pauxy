import copy
import cmath
import h5py
import math
import numpy
import scipy.linalg
import time
from pauxy.walkers.multi_ghf import MultiGHFWalker
from pauxy.walkers.single_det import SingleDetWalker
from pauxy.walkers.multi_det import MultiDetWalker
from pauxy.walkers.thermal import ThermalWalker
from pauxy.walkers.stack import FieldConfig
from pauxy.qmc.comm import FakeComm
from pauxy.utils.io import get_input_value


class Walkers(object):
    """Container for groups of walkers which make up a wavefunction.

    Parameters
    ----------
    system : object
        System object.
    trial : object
        Trial wavefunction object.
    nwalkers : int
        Number of walkers to initialise.
    nprop_tot : int
        Total number of propagators to store for back propagation + itcf.
    nbp : int
        Number of back propagation steps.
    """

    def __init__(self, walker_opts, system, trial, qmc, verbose=False,
                comm=None):
        self.nwalkers = qmc.nwalkers
        self.ntot_walkers = qmc.ntot_walkers
        self.write_freq = walker_opts.get('write_freq', 0)
        self.write_file = walker_opts.get('write_file', 'restart.h5')
        self.read_file = walker_opts.get('read_file', None)
        if comm is None:
            rank = 0
        else:
            rank = comm.rank
        if verbose:
            print ("# Setting up wavefunction object.")
        if trial.name == 'MultiSlater':
            self.walker_type = 'MSD'
            self.walkers = [
                    MultiDetWalker(walker_opts, system, trial,
                                   verbose=(verbose and w == 0))
                    for w in range(qmc.nwalkers)
                    ]
        elif trial.name == 'thermal':
            self.walker_type = 'thermal'
            self.walkers = [ThermalWalker(walker_opts, system, trial, verbose and w==0)
                            for w in range(qmc.nwalkers)]
        else:
            self.walker_type = 'SD'
            self.walkers = [SingleDetWalker(walker_opts, system, trial, w)
                            for w in range(qmc.nwalkers)]
        if system.name == "Generic" or system.name == "UEG":
            dtype = complex
        else:
            dtype = int
        pcont_method = get_input_value(walker_opts, 'population_control',
                                       default='comb')
        if pcont_method == 'comb':
            self.pop_control = self.comb
        elif pcont_method == 'pair_branch':
            self.pop_control = self.pair_branch
        if verbose:
            print("# Using {} population control "
                  "algorithm.".format(pcont_method))
        self.stack_size = walker_opts.get('stack_size', 1)
        walker_size = 3 + self.walkers[0].phi.size
        self.min_weight = walker_opts.get('min_weight', 0.1)
        self.max_weight = walker_opts.get('min_weight', 4.0)
        if self.write_freq > 0:
            self.write_restart = True
            self.dsets = []
            with h5py.File(self.write_file,'w',driver='mpio',comm=comm) as fh5:
                for i in range(self.ntot_walkers):
                    fh5.create_dataset('walker_%d'%i, (walker_size,),
                                       dtype=numpy.complex128)

        else:
            self.write_restart = False
        if self.read_file is not None:
            if verbose:
                print("# Reading walkers from %s file series."%self.read_file)
            self.read_walkers(comm)
        self.calculate_nwalkers()
        self.set_total_weight(qmc.ntot_walkers)

    def calculate_nwalkers(self):
        self.nw = sum(w.alive for w in self.walkers)

    def orthogonalise(self, trial, free_projection):
        """Orthogonalise all walkers.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        free_projection : bool
            True if doing free projection.
        """
        for w in self.walkers:
            detR = w.reortho(trial)
            if free_projection:
                (magn, dtheta) = cmath.polar(detR)
                w.weight *= magn
                w.phase *= cmath.exp(1j*dtheta)

    def add_field_config(self, nprop_tot, nbp, system, dtype):
        """Add FieldConfig object to walker object.

        Parameters
        ----------
        nprop_tot : int
            Total number of propagators to store for back propagation + itcf.
        nbp : int
            Number of back propagation steps.
        nfields : int
            Number of fields to store for each back propagation step.
        dtype : type
            Field configuration type.
        """
        for w in self.walkers:
            w.field_configs = FieldConfig(system.nfields, nprop_tot, nbp, dtype)

    def copy_historic_wfn(self):
        """Copy current wavefunction to psi_n for next back propagation step."""
        for (i,w) in enumerate(self.walkers):
            numpy.copyto(self.walkers[i].phi_old, self.walkers[i].phi)

    def copy_bp_wfn(self, phi_bp):
        """Copy back propagated wavefunction.

        Parameters
        ----------
        phi_bp : object
            list of walker objects containing back propagated walkers.
        """
        for (i, (w,wbp)) in enumerate(zip(self.walkers, phi_bp)):
            numpy.copyto(self.walkers[i].phi_bp, wbp.phi)

    def copy_init_wfn(self):
        """Copy current wavefunction to initial wavefunction.

        The definition of the initial wavefunction depends on whether we are
        calculating an ITCF or not.
        """
        for (i,w) in enumerate(self.walkers):
            numpy.copyto(self.walkers[i].phi_right, self.walkers[i].phi)

    def comb(self, comm):
        """Apply the comb method of population control / branching.

        See Booth & Gubernatis PRE 80, 046704 (2009).

        Parameters
        ----------
        comm : MPI communicator
        """
        # Need make a copy to since the elements in psi are only references to
        # walker objects in memory. We don't want future changes in a given
        # element of psi having unintended consequences.
        # todo : add phase to walker for free projection
        weights = numpy.array([abs(w.weight) for w in self.walkers])
        global_weights = None
        if self.ntot_walkers == 1:
            self.walkers[0].weight = 1
            return
        if comm.rank == 0:
            global_weights = numpy.empty(len(weights)*comm.size)
            parent_ix = numpy.zeros(len(global_weights), dtype='i')
        else:
            global_weights = numpy.empty(len(weights)*comm.size)
            parent_ix = numpy.empty(len(global_weights), dtype='i')
        comm.Gather(weights, global_weights, root=0)
        if comm.rank == 0:
            total_weight = sum(global_weights)
            cprobs = numpy.cumsum(global_weights)
            ntarget = self.nw * comm.size
            r = numpy.random.random()
            comb = [(i+r) * (total_weight/(ntarget)) for i in range(ntarget)]
            iw = 0
            ic = 0
            while ic < len(comb):
                if comb[ic] < cprobs[iw]:
                    parent_ix[iw] += 1
                    ic += 1
                else:
                    iw += 1
        else:
            total_weight = None

        comm.Bcast(parent_ix, root=0)
        # Keep total weight saved for capping purposes.
        total_weight = comm.bcast(total_weight, root=0)
        self.set_total_weight(total_weight)
        # where returns a tuple (array,), selecting first element.
        kill = numpy.where(parent_ix == 0)[0]
        clone = numpy.where(parent_ix > 1)[0]
        reqs = []
        walker_buffers = []
        # First initiate non-blocking sends of walkers.
        for i, (c, k) in enumerate(zip(clone, kill)):
            # Sending from current processor?
            if c // self.nw == comm.rank:
                # Location of walker to clone in local list.
                clone_pos = c % self.nw
                # copying walker data to intermediate buffer to avoid issues
                # with accessing walker data during send. Might not be
                # necessary.
                walker_buffers.append(self.walkers[clone_pos].__dict__)
                dest_proc = k // self.nw
                reqs.append(comm.isend(walker_buffers[-1],
                            dest=dest_proc, tag=i))
        # Now receive walkers on processors where walkers are to be killed.
        for i, (c, k) in enumerate(zip(clone, kill)):
            # Receiving to current processor?
            if k // self.nw == comm.rank:
                # Processor we are receiving from.
                source_proc = c // self.nw
                # Location of walker to kill in local list of walkers.
                kill_pos = k % self.nw
                walker_buffer = comm.recv(source=source_proc, tag=i)
                self.walkers[kill_pos].__dict__ = walker_buffer
        # Complete non-blocking send.
        for rs in reqs:
            rs.wait()
        # Necessary?
        comm.Barrier()
        # Reset walker weight.
        # TODO: check this.
        for w in self.walkers:
            w.weight = 1.0

    def pair_branch(self, comm):
        walker_info = [[w.weight,1,comm.rank,comm.rank] for w in self.walkers]
        glob_inf = comm.allgather(walker_info)
        # Unpack lists
        glob_inf = numpy.array([item for sub in glob_inf for item in sub])
        # glob_inf.sort(key=lambda x: x[0])
        sort = numpy.argsort(glob_inf[:,0])
        isort = numpy.argsort(sort)
        glob_inf = glob_inf[sort]
        s = 0
        e = len(glob_inf) - 1
        tags = []
        isend = 0
        while s < e:
            if glob_inf[s][0] < self.min_weight or glob_inf[e][0] > self.max_weight:
                # sum of paired walker weights
                wab = glob_inf[s][0] + glob_inf[e][0]
                r = numpy.random.rand()
                if r < glob_inf[e][0] / wab:
                    # clone large weight walker
                    glob_inf[e][0] = 0.5 * wab
                    glob_inf[e][1] = 2
                    # Processor we will send duplicated walker to
                    glob_inf[e][3] = glob_inf[s][2]
                    send = glob_inf[s][2]
                    # Kill small weight walker
                    glob_inf[s][0] = 0.0
                    glob_inf[s][1] = 0
                    glob_inf[s][3] = glob_inf[e][2]
                else:
                    # clone small weight walker
                    glob_inf[s][0] = 0.5 * wab
                    glob_inf[s][1] = 2
                    # Processor we will send duplicated walker to
                    glob_inf[s][3] = glob_inf[e][2]
                    send = glob_inf[e][2]
                    # Kill small weight walker
                    glob_inf[e][0] = 0.0
                    glob_inf[e][1] = 0
                    glob_inf[e][3] = glob_inf[s][2]
                tags.append([send])
                s += 1
                e -= 1
            else:
                break
        glob_inf = glob_inf[isort]
        reqs = []
        nw = self.nwalkers
        walker_buffers = []
        for iw, walker in enumerate(glob_inf[comm.rank*nw:(comm.rank+1)*nw]):
            if walker[1] > 1:
                tag = comm.rank*len(walker_info) + walker[3]
                walker_buffers.append(self.walkers[iw].__dict__)
                reqs.append(comm.isend(walker_buffers[-1],
                            dest=int(round(walker[3])), tag=tag))
        for iw, walker in enumerate(glob_inf[comm.rank*nw:(comm.rank+1)*nw]):
            if walker[1] == 0:
                tag = walker[3]*len(walker_info) + comm.rank
                buff = comm.recv(source=int(round(walker[3])), tag=tag)
                self.walkers[iw].__dict__ = walker_buffer
        for r in reqs:
            r.wait()


    def recompute_greens_function(self, trial, time_slice=None):
        for w in self.walkers:
            w.greens_function(trial, time_slice)

    def set_total_weight(self, total_weight):
        for w in self.walkers:
            w.total_weight = total_weight

    def reset(self, trial):
        for w in self.walkers:
            w.stack.reset()
            w.stack.set_all(trial.dmat)
            w.greens_function(trial)
            w.weight = 1.0
            w.phase = 1.0 + 0.0j

    def get_write_buffer(self, i):
        w = self.walkers[i]
        buff = numpy.concatenate([[w.weight], [w.phase], [w.ot], w.phi.ravel()])
        return buff

    def set_walker_from_buffer(self, i, buff):
        w = self.walkers[i]
        w.weight = buff[0]
        w.phase = buff[1]
        w.ot = buff[2]
        w.phi = buff[3:].reshape(self.walkers[i].phi.shape)

    def write_walkers(self, comm):
        start = time.time()
        with h5py.File(self.write_file,'r+',driver='mpio',comm=comm) as fh5:
            for (i,w) in enumerate(self.walkers):
                ix = i + self.nwalkers*comm.rank
                buff = self.get_write_buffer(i)
                fh5['walker_%d'%ix][:] = self.get_write_buffer(i)
        if comm.rank == 0:
            print(" # Writing walkers to file.")
            print(" # Time to write restart: {:13.8e} s"
                  .format(time.time()-start))

    def read_walkers(self, comm):
        with h5py.File(self.read_file, 'r') as fh5:
            for (i,w) in enumerate(self.walkers):
                try:
                    ix = i + self.nwalkers*comm.rank
                    self.set_walker_from_buffer(i, fh5['walker_%d'%ix][:])
                except KeyError:
                    print(" # Could not read walker data from:"
                          " %s"%(self.read_file))
