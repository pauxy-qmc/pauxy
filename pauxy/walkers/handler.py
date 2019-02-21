import copy
import numpy
import math
import cmath
import scipy.linalg
from pauxy.walkers.multi_ghf import MultiGHFWalker
from pauxy.walkers.single_det import SingleDetWalker
from pauxy.walkers.multi_det import MultiDetWalker
from pauxy.walkers.thermal import ThermalWalker, PropagatorStack
from pauxy.qmc.comm import FakeComm


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

    def __init__(self, walker_opts, system, trial, qmc,
                 nprop_tot, nbp, verbose=False):
        self.nwalkers = qmc.nwalkers
        self.ntot_walkers = qmc.ntot_walkers
        if verbose:
            print ("# Setting up wavefunction object.")
        if trial.name == 'multi_determinant':
            if trial.type == 'GHF':
                self.walkers = [MultiGHFWalker(walker_opts, system, trial)
                                for w in range(qmc.nwalkers)]
            else:
                self.walkers = [MultiDetWalker(walker_opts, system, trial)
                                for w in range(qmc.nwalkers)]
        elif trial.name == 'thermal':
            self.walkers = [ThermalWalker(walker_opts, system, trial, verbose and w==0)
                            for w in range(qmc.nwalkers)]
        else:
            self.walkers = [SingleDetWalker(walker_opts, system, trial, w)
                            for w in range(qmc.nwalkers)]
        if system.name == "Generic" or system.name == "UEG":
            dtype = complex
        else:
            dtype = int
        self.pop_control = self.comb
        self.stack_size = walker_opts.get('stack_size', 1)
        self.add_field_config(nprop_tot, nbp, system, dtype)
        self.calculate_total_weight()
        self.calculate_nwalkers()

    def calculate_total_weight(self):
        self.total_weight = sum(w.weight for w in self.walkers if w.alive)

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
            w.stack = PropagatorStack(self.stack_size, nprop_tot, system.nbasis,
                                      dtype, None, None, False)

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
        iproc : int
            Current processor index
        nprocs : int
            Total number of mpi processors
        """
        # Need make a copy to since the elements in psi are only references to
        # walker objects in memory. We don't want future changes in a given
        # element of psi having unintended consequences.
        new_psi = copy.deepcopy(self.walkers)
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

        # Wait for master
        comm.Bcast(parent_ix, root=0)
        # Copy back new information
        send = []
        recv = []
        for (i, w) in enumerate(parent_ix):
            # processor index of killed walker
            if w == 0:
                recv.append([i//self.nw, i%self.nw])
            elif w > 1:
                for ns in range(0, w-1):
                    send.append([i//self.nw, i%self.nw])
        # Send / Receive walkers.
        reqs = []
        reqr = []
        walker_buffers = []
        for i, (s,r) in enumerate(zip(send, recv)):
            # don't want to access buffer during non-blocking send.
            if (comm.rank == s[0]):
                # Sending duplicated walker
                walker_buffers.append(new_psi[s[1]].get_buffer())
                reqs.append(comm.isend(walker_buffers[-1],
                            dest=r[0], tag=i))
        for i, (s,r) in enumerate(zip(send, recv)):
            if isinstance(comm, FakeComm):
                # no mpi4py
                walker_buffer = walker_buffers[i]
                self.walkers[r[1]].set_buffer(walker_buffer)
            else:
                if (comm.rank == r[0]):
                    walker_buffer = comm.recv(source=s[0], tag=i)
                    self.walkers[r[1]].set_buffer(walker_buffer)
        for rs in reqs:
            rs.wait()
        comm.Barrier()
        # Reset walker weight.
        for w in self.walkers:
            w.weight = 1.0

    def recompute_greens_function(self, trial, time_slice=None):
        for w in self.walkers:
            w.greens_function(trial, time_slice)

    def reset(self, trial):
        for w in self.walkers:
            w.stack.reset()
            w.stack.set_all(trial.dmat)
            w.greens_function(trial)
            w.weight = 1.0
            w.phase = 1.0 + 0.0j

class FieldConfig(object):
    """Object for managing stored auxilliary field.

    Parameters
    ----------
    nfields : int
        Number of fields to store for each back propagation step.
    nprop_tot : int
        Total number of propagators to store for back propagation + itcf.
    nbp : int
        Number of back propagation steps.
    dtype : type
        Field configuration type.
    """
    def __init__(self, nfields, nprop_tot, nbp, dtype):
        self.configs = numpy.zeros(shape=(nprop_tot, nfields), dtype=dtype)
        self.cos_fac = numpy.zeros(shape=(nprop_tot, 1), dtype=float)
        self.weight_fac = numpy.zeros(shape=(nprop_tot, 1), dtype=complex)
        self.tot_wfac = 1.0 + 0j
        self.step = 0
        # need to account for first iteration and how we iterate
        self.block = -1
        self.ib = 0
        self.nfields = nfields
        self.nbp = nbp
        self.nprop_tot = nprop_tot
        self.nblock = nprop_tot // nbp

    def push(self, config):
        """Add field configuration to buffer.

        Parameters
        ----------
        config : int
            Auxilliary field configuration.
        """
        self.configs[self.step, self.ib] = config
        self.ib = (self.ib + 1) % self.nfields
        # Completed field configuration for this walker?
        if self.ib == 0:
            self.step = (self.step + 1) % self.nprop_tot
            # Completed this block of back propagation steps?
            if self.step % self.nbp == 0:
                self.block = (self.block + 1) % self.nblock

    def push_full(self, config, cfac, wfac):
        """Add full field configuration for walker to buffer.

        Parameters
        ----------
        config : :class:`numpy.ndarray`
            Auxilliary field configuration.
        cfac : float
            Cosine factor if using phaseless approximation.
        wfac : complex
            Weight factor to restore full walker weight following phaseless
            approximation.
        """
        self.configs[self.step] = config
        self.cos_fac[self.step] = cfac
        self.weight_fac[self.step] = wfac
        try:
            self.tot_wfac *= wfac/cfac
        except ZeroDivisionError:
            self.tot_wfac = 0.0
        # Completed field configuration for this walker?
        self.step = (self.step + 1) % self.nprop_tot
        # Completed this block of back propagation steps?
        if self.step % self.nbp == 0:
            self.block = (self.block + 1) % self.nblock

    def get_block(self):
        """Return a view to current block for back propagation."""
        start = self.block * self.nbp
        end = (self.block + 1) * self.nbp
        return (self.configs[start:end], self.cos_fac[start:end],
                self.weight_fac[start:end])

    def get_superblock(self):
        """Return a view to current super block for ITCF."""
        end = self.nprop_tot - self.nbp
        return (self.configs[:end], self.cos_fac[:end], self.weight_fac[:end])
