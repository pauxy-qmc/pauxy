import numpy
import scipy.linalg
import copy
import pauxy.estimators
from pauxy.trial_wavefunction import FreeElectron, read_fortran_complex_numbers
import math


class Walkers:
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

    def __init__(self, system, trial, nwalkers, nprop_tot, nbp):
        if trial.name == 'multi_determinant':
            if trial.type == 'GHF':
                self.walkers = [MultiGHFWalker(1, system, trial)
                                for w in range(nwalkers)]
            else:
                self.walkers = [MultiDetWalker(1, system, trial)
                                for w in range(nwalkers)]
        else:
            self.walkers = [Walker(1, system, trial, w)
                            for w in range(nwalkers)]
        if system.name == "Generic":
            dtype = complex
        else:
            dtype = int
        self.pop_control = self.comb
        self.add_field_config(nprop_tot, nbp, system.nfields, dtype)
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
                w.weight = detR * w.weight

    def add_field_config(self, nprop_tot, nbp, nfields, dtype):
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
            w.field_configs = FieldConfig(nfields, nprop_tot, nbp, dtype)

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
            numpy.copyto(self.walkers[i].phi_init, self.walkers[i].phi)

    def comb(self, comm, iproc, nprocs):
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
        global_weights = numpy.zeros(len(weights)*nprocs)
        if iproc == 0:
            parent_ix = numpy.arange(len(global_weights), dtype='i')
        else:
            parent_ix = numpy.empty(len(global_weights), dtype='i')

        if comm is not None:
            comm.Gather(weights, global_weights, root=0)
        if (comm is None) or iproc == 0:
            total_weight = sum(global_weights)
            cprobs = numpy.cumsum(global_weights)
            ntarget = self.nw * nprocs

            r = numpy.random.random()
            comb = [(i+r) * (total_weight/(ntarget)) for i in range(ntarget)]
            for (ic, c) in enumerate(comb):
                for (iw, w) in enumerate(cprobs):
                    if c < w:
                        parent_ix[ic] = iw
                        break

        # Wait for master
        if comm is not None:
            comm.Bcast(parent_ix, root=0)
        # Copy back new information
        send = []
        recv = []
        for (i,p) in enumerate(parent_ix):
            loc_ix = i % self.nw
            new_ix = p % self.nw
            proc_ix = i // self.nw
            new_proc_ix = p // self.nw
            if proc_ix == iproc and new_ix != loc_ix:
                # Walker on current processor has been killed and replaced with
                # another.
                # [location on current proc's, proc id of new walker]
                recv.append([loc_ix, new_proc_ix, i])
                if new_proc_ix == iproc:
                    send.append([new_ix, proc_ix, i])
            elif new_proc_ix == iproc and new_ix != loc_ix:
                # We need to send a walker somewhere else.
                # [location on current proc's, proc id of new walker]
                send.append([new_ix, proc_ix, i])
        # Send / Receive walkers.
        reqs = []
        reqr = []
        walker_buffers = []
        for i, s in enumerate(send):
            walker_buffers.append(new_psi[s[0]].get_buffer())
            reqs.append(comm.isend(walker_buffers[i], dest=s[1], tag=s[2]))
        for rc in recv:
            walker_buffer = comm.recv(source=rc[1], tag=rc[2])
            self.walkers[rc[0]].set_buffer(walker_buffer)
        for rs in reqs:
            rs.wait()
        comm.Barrier()
        # Reset walker weight.
        for w in self.walkers:
            w.weight = 1.0

class Walker:
    """UHF style walker.

    Parameters
    ----------
    weight : int
        Walker weight.
    system : object
        System object.
    trial : object
        Trial wavefunction object.
    index : int
        Element of trial wavefunction to initalise walker to.
    """

    def __init__(self, weight, system, trial, index=0):
        self.weight = weight
        self.alive = 1
        if trial.initial_wavefunction == 'free_electron':
            self.phi = numpy.zeros(shape=(system.nbasis,system.ne),
                                   dtype=trial.psi.dtype)
            tmp = FreeElectron(system, system.ktwist.all() != None, {})
            self.phi[:,:system.nup] = tmp.psi[:,:system.nup]
            self.phi[:,system.nup:] = tmp.psi[:,system.nup:]
        else:
            self.phi = copy.deepcopy(trial.psi)
        self.inv_ovlp = [0, 0]
        self.nup = system.nup
        self.inverse_overlap(trial.psi)
        self.G = numpy.zeros(shape=(2, system.nbasis, system.nbasis),
                             dtype=trial.psi.dtype)
        self.Gmod = numpy.zeros(shape=(2, system.nbasis, system.nup),
                                dtype=trial.psi.dtype)
        self.greens_function(trial)
        self.ot = 1.0
        # interface consistency
        self.ots = numpy.zeros(1)
        self.E_L = pauxy.estimators.local_energy(system, self.G)[0].real
        # walkers overlap at time tau before backpropagation occurs
        self.ot_bp = 1.0
        # walkers weight at time tau before backpropagation occurs
        self.weight_bp = weight
        # Historic wavefunction for back propagation.
        self.phi_old = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        self.phi_init = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        self.phi_bp = copy.deepcopy(self.phi)
        self.weights = numpy.array([1])

    def inverse_overlap(self, trial):
        """Compute inverse overlap matrix from scratch.

        Parameters
        ----------
        trial : :class:`numpy.ndarray`
            Trial wavefunction.
        """
        nup = self.nup
        self.inv_ovlp[0] = (
            scipy.linalg.inv((trial[:,:nup].conj()).T.dot(self.phi[:,:nup]))
        )
        self.inv_ovlp[1] = (
            scipy.linalg.inv((trial[:,nup:].conj()).T.dot(self.phi[:,nup:]))
        )

    def update_inverse_overlap(self, trial, vtup, vtdown, i):
        """Update inverse overlap matrix given a single row update of walker.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        vtup : :class:`numpy.ndarray`
            Update vector for spin up sector.
        vtdown : :class:`numpy.ndarray`
            Update vector for spin down sector.
        i : int
            Basis index.
        """
        nup = self.nup
        self.inv_ovlp[0] = (
            pauxy.utils.sherman_morrison(self.inv_ovlp[0],
                                         trial.psi[i,:nup].conj(),
                                         vtup)
        )
        self.inv_ovlp[1] = (
            pauxy.utils.sherman_morrison(self.inv_ovlp[1],
                                         trial.psi[i,nup:].conj(),
                                         vtdown)
        )

    def calc_otrial(self, trial):
        """Caculate overlap with trial wavefunction.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.

        Returns
        -------
        ot : float / complex
            Overlap.
        """
        dup = scipy.linalg.det(self.inv_ovlp[0])
        ddn = scipy.linalg.det(self.inv_ovlp[1])
        return 1.0 / (dup*ddn)

    def update_overlap(self, probs, xi, coeffs):
        """Update overlap.

        Parameters
        ----------
        probs : :class:`numpy.ndarray`
            Probabilities for chosing particular field configuration.
        xi : int
            Chosen field configuration.
        coeffs : :class:`numpy.ndarray`
            Trial wavefunction coefficients. For interface consistency.
        """
        self.ot = 2 * self.ot * probs[xi]

    def reortho(self, trial):
        """reorthogonalise walker.

        parameters
        ----------
        trial : object
            trial wavefunction object. for interface consistency.
        """
        nup = self.nup
        (self.phi[:,:nup], Rup) = scipy.linalg.qr(self.phi[:,:nup],
                                                  mode='economic')
        (self.phi[:,nup:], Rdown) = scipy.linalg.qr(self.phi[:,nup:],
                                                    mode='economic')
        signs_up = numpy.diag(numpy.sign(numpy.diag(Rup)))
        signs_down = numpy.diag(numpy.sign(numpy.diag(Rdown)))
        self.phi[:,:nup] = self.phi[:,:nup].dot(signs_up)
        self.phi[:,nup:] = self.phi[:,nup:].dot(signs_down)
        drup = scipy.linalg.det(signs_up.dot(Rup))
        drdn = scipy.linalg.det(signs_down.dot(Rdown))
        detR = drup * drdn
        self.ot = self.ot / detR
        return detR

    def greens_function(self, trial):
        """Compute walker's green's function.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        """
        nup = self.nup
        t = trial.psi
        self.G[0] = (
            (self.phi[:,:nup].dot(self.inv_ovlp[0]).dot(t[:,:nup].conj().T)).T
        )
        self.G[1] = (
            (self.phi[:,nup:].dot(self.inv_ovlp[1]).dot(t[:,nup:].conj().T)).T
        )

    def rotated_greens_function(self):
        """Compute "rotated" walker's green's function.

        Green's function without trial wavefunction multiplication.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        """
        nup = self.nup
        self.Gmod[0] = self.phi[:,:nup].dot(self.inv_ovlp[0])
        self.Gmod[1] = self.phi[:,nup:].dot(self.inv_ovlp[1])

    def local_energy(self, system):
        """Compute walkers local energy

        Parameters
        ----------
        system : object
            System object.

        Returns
        -------
        (E, T, V) : tuple
            Mixed estimates for walker's energy components.
        """
        return pauxy.estimators.local_energy(system, self.G)

    def get_buffer(self):
        """Get walker buffer for MPI communication

        Returns
        -------
        buff : dict
            Relevant walker information for population control.
        """
        buff = {
            'phi': self.phi,
            'phi_old': self.phi_old,
            'phi_init': self.phi_init,
            'phi_bp': self.phi_bp,
            'weight': self.weight,
            'inv_ovlp': self.inv_ovlp,
            'G': self.G,
            'overlap': self.ot,
            'overlaps': self.ots,
            'fields': self.field_configs.configs,
            'cfacs': self.field_configs.cos_fac,
            'E_L': self.E_L,
            'weight_fac': self.field_configs.weight_fac
        }
        return buff

    def set_buffer(self, buff):
        """Set walker buffer following MPI communication

        Parameters
        -------
        buff : dict
            Relevant walker information for population control.
        """
        self.phi = numpy.copy(buff['phi'])
        self.phi_old = numpy.copy(buff['phi_old'])
        self.phi_init = numpy.copy(buff['phi_init'])
        self.phi_bp = numpy.copy(buff['phi_bp'])
        self.inv_ovlp = numpy.copy(buff['inv_ovlp'])
        self.G = numpy.copy(buff['G'])
        self.weight = buff['weight']
        self.ot = buff['overlap']
        self.E_L = buff['E_L']
        self.ots = numpy.copy(buff['overlaps'])
        self.field_configs.configs = numpy.copy(buff['fields'])
        self.field_configs.cos_fac = numpy.copy(buff['cfacs'])
        self.field_configs.weight_fac = numpy.copy(buff['weight_fac'])

class MultiDetWalker:
    """Multi-UHF style walker.

    Parameters
    ----------
    weight : int
        Walker weight.
    system : object
        System object.
    trial : object
        Trial wavefunction object.
    index : int
        Element of trial wavefunction to initalise walker to.
    """

    def __init__(self, weight, system, trial, index=0):
        self.weight = nw
        self.alive = 1
        self.phi = copy.deepcopy(trial.psi[index])
        # This stores an array of overlap matrices with the various elements of
        # the trial wavefunction.
        up_shape = (trial.ndets, system.nup, system.nup)
        down_shape = (trial.ndets, system.ndown, system.ndown)
        self.inv_ovlp = [numpy.zeros(shape=(up_shape)),
                         numpy.zeros(shape=(down_shape))]
        self.inverse_overlap(trial.psi)
        # Green's functions for various elements of the trial wavefunction.
        self.Gi = numpy.zeros(shape=(trial.ndets, 2, system.nbasis,
                              system.nbasis))
        # Should be nfields per basis * ndets.
        # Todo: update this for the continuous HS trasnform case.
        self.R = numpy.zeros(shape=(trial.ndets, 2, 2))
        # Actual green's function contracted over determinant index in Gi above.
        # i.e., <psi_T|c_i^d c_j|phi>
        self.G = numpy.zeros(shape=(2, system.nbasis, system.nbasis))
        self.ots = numpy.zeros(shape=(2,trial.ndets))
        # Contains overlaps of the current walker with the trial wavefunction.
        self.ot = self.calc_otrial(trial)
        self.greens_function(trial, system.nup)
        self.E_L = pauxy.estimators.local_energy(system, self.G)[0].real
        G2 = pauxy.estimators.gab(trial.psi[0][:,:system.nup],
                                    trial.psi[0][:,:system.nup])
        self.index = index
        self.nup = system.nup
        self.field_config = numpy.zeros(shape=(system.nbasis), dtype=int)

    def inverse_overlap(self, trial):
        """Compute inverse overlap matrix from scratch.

        Parameters
        ----------
        trial : :class:`numpy.ndarray`
            Trial wavefunction.
        """
        nup = self.nup
        for (indx, t) in enumerate(trial):
            self.inv_ovlp[0][indx,:,:] = (
                scipy.linalg.inv((t[:,:nup].conj()).T.dot(self.phi[:,:nup]))
            )
            self.inv_ovlp[1][indx,:,:] = (
                scipy.linalg.inv((t[:,nup:].conj()).T.dot(self.phi[:,nup:]))
            )

    def calc_otrial(self, trial):
        """Caculate overlap with trial wavefunction.

        The trial wavefunctions coefficients should be complex conjugated!

        Parameters
        ----------
        trial : object
            Trial wavefunction object.

        Returns
        -------
        ot : float / complex
            Overlap.
        """
        ot = 0.0
        for (ix, c) in enumerate(trial.coeffs):
            deto_up = 1.0 / scipy.linalg.det(self.inv_ovlp[0][ix,:,:])
            deto_down = 1.0 / scipy.linalg.det(self.inv_ovlp[1][ix,:,:])
            self.ots[0, ix] = deto_up
            self.ots[1, ix] = deto_down
            ot += c * deto_up * deto_down
        return ot

    def update_overlap(self, probs, xi, coeffs):
        """Update overlap.

        Parameters
        ----------
        probs : :class:`numpy.ndarray`
            Probabilities for chosing particular field configuration.
        xi : int
            Chosen field configuration.
        coeffs : :class:`numpy.ndarray`
            Trial wavefunction coefficients. For interface consistency.
        """
        # Update each component's overlap and the total overlap.
        # The trial wavefunctions coeficients should be included in ots?
        self.ots = numpy.einsum('ji,ij->ij', self.R[:, xi, :], self.ots)
        self.ot = sum(coeffs * self.ots[0, :] * self.ots[1, :])

    def reortho(self, trial):
        """Reorthogonalise walker.

        Parameters
        ----------
        trial : object
            Trial wavefunction object. For interface consistency.
        """
        nup = self.nup
        # We assume that our walker is still block diagonal in the spin basis.
        (self.phi[:, :nup], Rup) = scipy.linalg.qr(self.phi[:, :nup],
                                                   mode='economic')
        (self.phi[:, nup:], Rdown) = scipy.linalg.qr(self.phi[:, nup:],
                                                     mode='economic')
        # Enforce a positive diagonal for the overlap.
        signs_up = numpy.diag(numpy.sign(numpy.diag(Rup)))
        signs_down = numpy.diag(numpy.sign(numpy.diag(Rdown)))
        self.phi[:, :nup] = self.phi[:, :nup].dot(signs_up)
        self.phi[:, nup:] = self.phi[:, nup:].dot(signs_down)
        # Todo: R is upper triangular.
        detR_up = scipy.linalg.det(signs_up.dot(Rup))
        detR_down = scipy.linalg.det(signs_down.dot(Rdown))
        self.ots[0] = self.ots[0] / detR_up
        self.ots[1] = self.ots[1] / detR_down
        self.ot = self.ot / (detR_up * detR_down)

    def greens_function(self, trial):
        """Compute walker's green's function.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        """
        nup = self.nup
        for (ix, t) in enumerate(trial.psi):
            # construct "local" green's functions for each component of psi_T
            self.Gi[ix, 0, :, :] = (
                (self.phi[:, :nup].dot(self.inv_ovlp[0][ix]).dot(
                    t[:, :nup].conj().T)).T
            )
            self.Gi[ix, 1, :, :] = (
                (self.phi[:, nup:].dot(self.inv_ovlp[1][ix]).dot(
                    t[:, nup:].conj().T)).T
            )
        denom = numpy.einsum('j,ij->i', trial.coeffs, self.ots)
        self.G = numpy.einsum('i,ijkl,ji->jkl', trial.coeffs,
                              self.Gi, self.ots)
        self.G[0] = self.G[0] / denom[0]
        self.G[1] = self.G[1] / denom[1]

    def update_inverse_overlap(self, trial, vtup, vtdown, i):
        """Update inverse overlap matrix given a single row update of walker.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        vtup : :class:`numpy.ndarray`
            Update vector for spin up sector.
        vtdown : :class:`numpy.ndarray`
            Update vector for spin down sector.
        i : int
            Basis index.
        """
        nup = self.nup
        for (ix, t) in enumerate(trial.psi):
            self.inv_ovlp[0][ix] = (
                pauxy.utils.sherman_morrison(self.inv_ovlp[0][ix],
                                             t[i, :nup].conj(), vtup)
            )
            self.inv_ovlp[1][ix] = (
                pauxy.utils.sherman_morrison(self.inv_ovlp[1][ix],
                                             t[i, nup:].conj(), vtdown)
            )

    def local_energy(self, system):
        """Compute walkers local energy

        Parameters
        ----------
        system : object
            System object.

        Returns
        -------
        (E, T, V) : tuple
            Mixed estimates for walker's energy components.
        """
        return pauxy.estimators.local_energy_multi_det(system, self.Gi,
                                                       self.weights)


class MultiGHFWalker:
    """Multi-GHF style walker.

    Parameters
    ----------
    weight : int
        Walker weight.
    system : object
        System object.
    trial : object
        Trial wavefunction object.
    index : int
        Element of trial wavefunction to initalise walker to.
    weights : string
        Initialise weights to zeros or ones.
    wfn0 : string
        Initial wavefunction.
    """

    def __init__(self, weight, system, trial, index=0,
                 weights='zeros', wfn0='init'):
        self.weight = nw
        self.alive = 1
        # Initialise to a particular free electron slater determinant rather
        # than GHF. Can actually initialise to GHF by passing single GHF with
        # initial_wavefunction. The distinction is really for back propagation
        # when we may want to use the full expansion.
        self.nup = system.nup
        if wfn0 == 'init':
            # Initialise walker with single determinant.
            if trial.initial_wavefunction != 'free_electron':
                orbs = read_fortran_complex_numbers(trial.read_init)
                self.phi = orbs.reshape((2*system.nbasis, system.ne), order='F')
            else:
                self.phi = numpy.zeros(shape=(2*system.nbasis,system.ne),
                                    dtype=trial.psi.dtype)
                tmp = FreeElectron(system, trial.psi.dtype==complex, {})
                self.phi[:system.nbasis,:system.nup] = tmp.psi[:,:system.nup]
                self.phi[system.nbasis:,system.nup:] = tmp.psi[:,system.nup:]
        else:
            self.phi = copy.deepcopy(trial.psi)
        # This stores an array of overlap matrices with the various elements of
        # the trial wavefunction.
        self.inv_ovlp = numpy.zeros(shape=(trial.ndets, system.ne, system.ne),
                                    dtype=self.phi.dtype)
        if weights == 'zeros':
            self.weights = numpy.zeros(trial.ndets, dtype=trial.psi.dtype)
        else:
            self.weights = numpy.ones(trial.ndets, dtype=trial.psi.dtype)
        if wfn0 != 'GHF':
            self.inverse_overlap(trial.psi)
        # Green's functions for various elements of the trial wavefunction.
        self.Gi = numpy.zeros(shape=(trial.ndets, 2*system.nbasis,
                                     2*system.nbasis), dtype=self.phi.dtype)
        # Should be nfields per basis * ndets.
        # Todo: update this for the continuous HS trasnform case.
        self.R = numpy.zeros(shape=(trial.ndets, 2), dtype=self.phi.dtype)
        # Actual green's function contracted over determinant index in Gi above.
        # i.e., <psi_T|c_i^d c_j|phi>
        self.G = numpy.zeros(shape=(2*system.nbasis, 2*system.nbasis),
                             dtype=self.phi.dtype)
        self.ots = numpy.zeros(trial.ndets, dtype=self.phi.dtype)
        # Contains overlaps of the current walker with the trial wavefunction.
        if wfn0 != 'GHF':
            self.ot = self.calc_otrial(trial)
            self.greens_function(trial)
            self.E_L = pauxy.estimators.local_energy_ghf(system, self.Gi,
                                                         self.weights,
                                                         sum(self.weights))[0].real
        self.nb = system.nbasis
        # Historic wavefunction for back propagation.
        self.phi_old = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        self.phi_init = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        self.phi_bp = copy.deepcopy(trial.psi)

    def inverse_overlap(self, trial):
        """Compute inverse overlap matrix from scratch.

        Parameters
        ----------
        trial : :class:`numpy.ndarray`
            Trial wavefunction.
        """
        nup = self.nup
        for (indx, t) in enumerate(trial):
            self.inv_ovlp[indx,:,:] = (
                scipy.linalg.inv((t.conj()).T.dot(self.phi))
            )

    def calc_otrial(self, trial):
        """Caculate overlap with trial wavefunction.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.

        Returns
        -------
        ot : float / complex
            Overlap.
        """
        # The trial wavefunctions coefficients should be complex conjugated
        # on initialisation!
        for (ix, inv) in enumerate(self.inv_ovlp):
            self.ots[ix] = 1.0 / scipy.linalg.det(inv)
            self.weights[ix] = trial.coeffs[ix] * self.ots[ix]
        return sum(self.weights)

    def update_overlap(self, probs, xi, coeffs):
        """Update overlap.

        Parameters
        ----------
        probs : :class:`numpy.ndarray`
            Probabilities for chosing particular field configuration.
        xi : int
            Chosen field configuration.
        coeffs : :class:`numpy.ndarray`
            Trial wavefunction coefficients. For interface consistency.
        """
        # Update each component's overlap and the total overlap.
        # The trial wavefunctions coeficients should be included in ots?
        self.ots = self.R[:,xi] * self.ots
        self.weights = coeffs * self.ots
        self.ot = 2.0 * self.ot * probs[xi]

    def reortho(self, trial):
        """Update overlap.

        Parameters
        ----------
        probs : :class:`numpy.ndarray`
            Probabilities for chosing particular field configuration.
        xi : int
            Chosen field configuration.
        coeffs : :class:`numpy.ndarray`
            Trial wavefunction coefficients. For interface consistency.
        """
        nup = self.nup
        # We assume that our walker is still block diagonal in the spin basis.
        (self.phi[:self.nb,:nup], Rup) = (
            scipy.linalg.qr(self.phi[:self.nb,:nup], mode='economic')
        )
        (self.phi[self.nb:,nup:], Rdown) = (
            scipy.linalg.qr(self.phi[self.nb:,nup:], mode='economic')
        )
        # Enforce a positive diagonal for the overlap.
        signs_up = numpy.diag(numpy.sign(numpy.diag(Rup)))
        signs_down = numpy.diag(numpy.sign(numpy.diag(Rdown)))
        self.phi[:self.nb,:nup] = self.phi[:self.nb,:nup].dot(signs_up)
        self.phi[self.nb:,nup:] = self.phi[self.nb:,nup:].dot(signs_down)
        # Todo: R is upper triangular.
        drup = scipy.linalg.det(signs_up.dot(Rup))
        drdn = scipy.linalg.det(signs_down.dot(Rdown))
        detR = drup * drdn
        self.inverse_overlap(trial.psi)
        self.ot = self.calc_otrial(trial)

    def greens_function(self, trial):
        """Compute walker's green's function.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        """
        nup = self.nup
        for (ix, t) in enumerate(trial.psi):
            # construct "local" green's functions for each component of psi_T
            self.Gi[ix,:,:] = (
                (self.phi.dot(self.inv_ovlp[ix]).dot(t.conj().T)).T
            )
        denom = sum(self.weights)
        self.G = numpy.einsum('i,ijk->jk', self.weights, self.Gi) / denom

    def update_inverse_overlap(self, trial, vtup, vtdown, i):
        """Update inverse overlap matrix given a single row update of walker.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        vtup : :class:`numpy.ndarray`
            Update vector for spin up sector.
        vtdown : :class:`numpy.ndarray`
            Update vector for spin down sector.
        i : int
            Basis index.
        """
        nup = self.nup
        for (indx, t) in enumerate(trial.psi):
            self.inv_ovlp[indx,:,:] = (
                scipy.linalg.inv((t.conj()).T.dot(self.phi))
            )

    def local_energy(self, system):
        """Compute walkers local energy

        Parameters
        ----------
        system : object
            System object.

        Returns
        -------
        (E, T, V) : tuple
            Mixed estimates for walker's energy components.
        """
        return pauxy.estimators.local_energy_ghf(system, self.Gi,
                                                   self.weights, self.ot)
    # def update_inverse_overlap(self, trial, vtup, vtdown, nup, i):
        # for (ix, t) in enumerate(trial.psi):
            # self.inv_ovlp[ix] = (
                # pauxy.utils.sherman_morrison(self.inv_ovlp[ix],
                                               # t[:,:nup].T[:,i], vtup)
            # )

class FieldConfig:
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
