"""Routines and classes for estimation of observables."""

from __future__ import print_function

import numpy
import time
import copy
from mpi4py import MPI
import scipy.linalg
import afqmcpy.utils


class Estimators():
    """Container for qmc estimates of observables.

    Attributes
    ----------
    header : list of strings
        Default estimates and simulation information.
    funit : file unit
        File to write back-propagated estimates to.
    bp_header : list of strings
        Back-propagated estimates.
    nestimators : int
        Number of estimators.
    names : :class:`afqmcpy.estimators.EstimatorEnum`
        Enum type object to allow for clearer (maybe) indexing.
    estimates : :class:`numpy.ndarray`
        Array containing accumulated estimates.
        See afqmcpy.estimators.Estimates.print_key for description.
    """

    def __init__(self, estimates, root, uuid, dt, nbasis, nwalkers, json_string):
        self.header = ['iteration', 'Weight', 'E_num', 'E_denom', 'E', 'time']
        self.key = {
            'iteration': "Simulation iteration. iteration*dt = tau.",
            'Weight': "Total walker weight.",
            'E_num': "Numerator for projected energy estimator.",
            'E_denom': "Denominator for projected energy estimator.",
            'E': "Projected energy estimator.",
            'time': "Time per processor to complete one iteration.",
        }
        print_key(self.key)
        print_header(self.header)
        self.nestimators = len(self.header[1:])
        # Sub-members:
        # 1. Back-propagation
        bp = estimates.get('back_propagation', None)
        self.back_propagation = bp is not None
        if self.back_propagation:
            self.back_prop = BackPropagation(bp, root, uuid, json_string)
            self.nestimators +=  len(self.back_prop.header[1:])
            self.nprop_tot = self.back_prop.nmax
        else:
            self.nprop_tot = 1
        # 2. Imaginary time correlation functions.
        itcf = estimates.get('itcf', None)
        self.calc_itcf = itcf is not None
        self.estimates = numpy.zeros(self.nestimators)
        if self.calc_itcf:
            self.itcf = ITCF(itcf, dt, root, uuid, json_string, nbasis)
            self.estimates = numpy.zeros(self.nestimators +
                                         len(self.itcf.spgf.flatten()))
            self.nprop_tot += self.itcf.nmax
        if self.calc_itcf or self.back_propagation:
            # Store for historic wavefunctions/walkers along back propagation
            # path.
            self.psi_hist = numpy.zeros(shape=(nwalkers, self.nprop_tot+1),
                                        dtype=object)
        self.names = EstimatorEnum(self.nestimators)
        # only store up component for the moment.
        self.zero(nbasis)


    def zero(self, nbasis):
        """Zero estimates.

        On return self.estimates is zerod and the timers are reset.

        """
        self.estimates[:] = 0
        self.estimates[self.names.time] = time.time()
        if self.back_propagation:
            self.back_prop.estimates[:] = 0
        if self.calc_itcf:
            self.itcf.spgf = numpy.zeros(shape=(self.itcf.nmax+1,
                                                2, 2,
                                                nbasis,
                                                nbasis))

    def print_step(self, state, comm, step, print_bp=True, print_itcf=True):
        """Print QMC estimates.

        Parameters
        ----------
        state : :class:`afqmcpy.state.State`
            Simulation state.
        comm :
            MPI communicator.
        step : int
            Current iteration number.
        """
        es = self.estimates
        ns = self.names
        es[ns.eproj] = (state.qmc.nmeasure*es[ns.enumer]/(state.nprocs*es[ns.edenom])).real
        es[ns.weight:ns.enumer] = es[ns.weight:ns.enumer].real
        # Back propagated estimates
        if self.back_propagation:
            es[ns.evar:ns.pot+1] = self.back_prop.estimates / state.nprocs
        es[ns.time] = (time.time()-es[ns.time]) / state.nprocs
        if self.calc_itcf:
            es[ns.pot+1:] = self.itcf.spgf.flatten() / state.nprocs
        global_estimates = numpy.zeros(len(self.estimates))
        comm.Reduce(es, global_estimates, op=MPI.SUM)
        global_estimates[:ns.time] = (
            global_estimates[:ns.time] / state.qmc.nmeasure
        )
        if state.root:
            print(afqmcpy.utils.format_fixed_width_floats([step]+
                                list(global_estimates[:ns.evar])))
            if self.back_propagation and print_bp:
                ff = (
                    afqmcpy.utils.format_fixed_width_floats([step]+
                        list(global_estimates[ns.evar:ns.pot+1]))
                )
                self.back_prop.funit.write((ff+'\n').encode('utf-8'))

        print_now = (
            state.root and step%self.nprop_tot == 0 and
            self.calc_itcf and print_itcf
        )
        if print_now:
            spgf = global_estimates[ns.pot+1:].reshape(self.itcf.spgf.shape)
            for (i,s) in enumerate(self.itcf.keys[0]):
                for (j,t) in enumerate(self.itcf.keys[1]):
                    self.itcf.to_file(spgf[:,i,j,:,:],
                                      self.itcf.rspace_units[i,j],
                                      state.qmc.dt)
            if self.itcf.kspace:
                M = state.system.nbasis
                # FFT the real space Green's function.
                # Todo : could just use numpy.fft.fft....
                spgf_k = numpy.einsum('ik,rqpkl,lj->rqpij', state.system.P,
                                      spgf, state.system.P.conj().T).real/M
                for (i,t) in enumerate(self.itcf.keys[0]):
                    for (j,s) in enumerate(self.itcf.keys[1]):
                        self.itcf.to_file(spgf_k[:,i,j,:,:],
                                          self.itcf.kspace_units[i,j],
                                          state.qmc.dt)

        self.zero(state.system.nbasis)

    def update(self, w, state):
        """Update regular estimates for walker w.

        Parameters
        ----------
        w : :class:`afqmcpy.walker.Walker`
            current walker
        state : :class:`afqmcpy.state.State`
            system parameters as well as current 'state' of the simulation.
        """
        if state.qmc.importance_sampling:
            # When using importance sampling we only need to know the current
            # walkers weight as well as the local energy, the walker's overlap
            # with the trial wavefunction is not needed.
            if 'continuous' in state.qmc.hubbard_stratonovich:
                self.estimates[self.names.enumer] += w.weight * w.E_L.real
            else:
                self.estimates[self.names.enumer] += w.weight*local_energy(state.system, w.G)[0].real
            self.estimates[self.names.weight] += w.weight
            self.estimates[self.names.edenom] += w.weight
        else:
            self.estimates[self.names.enumer] += (w.weight*local_energy(state.system, w.G)[0]*w.ot).real
            self.estimates[self.names.weight] += w.weight.real
            self.estimates[self.names.edenom] += (w.weight*w.ot).real

class EstimatorEnum:
    """Enum structure for help with indexing estimators array.

    python's support for enums doesn't help as it indexes from 1.
    """
    def __init__(self, nestimators):
        # Exception for alignment of equal sign.
        self.weight = 0
        self.enumer = 1
        self.edenom = 2
        self.eproj  = 3
        self.time   = 4
        self.evar   = 5
        self.kin    = 6
        self.pot    = 7


class BackPropagation:

    def __init__(self, bp, root, uuid, json_string):
        self.nmax = bp.get('nback_prop', 0)
        self.header = ['iteration', 'E', 'T', 'V']
        self.estimates = numpy.zeros(len(self.header[1:]))
        self.key = {
            'iteration': "Simulation iteration when back-propagation "
                         "measurement occured.",
            'E_var': "BP estimate for internal energy.",
            'T': "BP estimate for kinetic energy.",
            'V': "BP estimate for potential energy."
        }
        if root:
            file_name = 'back_propagated_estimates_%s.out'%uuid[:8]
            self.funit = open(file_name, 'ab')
            self.funit.write(json_string.encode('utf-8'))
            print_key(self.key, self.funit.write, eol='\n', encode=True)
            print_header(self.header, self.funit.write, eol='\n', encode=True)

    def update(self, system, psi_nm, psi_n, psi_bp):
        """Calculate back-propagated "local" energy for given walker/determinant.

        Parameters
        ----------
        psi : list of :class:`afqmcpy.walker.Walker` objects
            current distribution of walkers, i.e., at the current iteration in the
            simulation corresponding to :math:`\tau'=\tau+\tau_{bp}`.
        psit : list of :class:`afqmcpy.walker.Walker` objects
            previous distribution of walkers, i.e., at the current iteration in the
            simulation corresponding to :math:`\tau`.
        psi_bp : list of :class:`afqmcpy.walker.Walker` objects
            backpropagated walkers at time :math:`\tau_{bp}`.
        """
        denominator = sum(wnm.weight for wnm in psi_nm)
        current = numpy.zeros(3)
        GTB = [0, 0]
        nup = system.nup
        for i, (wnm, wn, wb) in enumerate(zip(psi_nm, psi_n, psi_bp)):
            GTB[0] = gab(wb.phi[:,:nup], wn.phi[:,:nup]).T
            GTB[1] = gab(wb.phi[:,nup:], wn.phi[:,nup:]).T
            current = current + wnm.weight*numpy.array(list(local_energy(system, GTB)))
        self.estimates = self.estimates + current.real / denominator


class ITCF:

    def __init__(self, itcf, dt, root, uuid, json_string, nbasis):
        self.stable = itcf.get('stable', True)
        self.tmax = itcf.get('tmax', 0.0)
        self.mode = itcf.get('mode', 'full')
        self.nmax = int(self.tmax/dt)
        self.kspace = itcf.get('kspace', False)
        # self.spgf(i,j,k,l,m) gives the (l,m)th element of the spin-j(=0 for up
        # and 1 for down) k-ordered(0=greater,1=lesser) imaginary time green's
        # function at time i.
        # +1 in the first dimension is for the green's function at time tau = 0.
        self.spgf = numpy.zeros(shape=(self.nmax+1, 2, 2,
                                       nbasis,
                                       nbasis))
        self.keys = [['up', 'down'], ['greater', 'lesser']]
        # I don't like list indexing so stick with numpy.
        if root:
            self.rspace_units = numpy.empty(shape=(2,2), dtype=object)
            self.kspace_units = numpy.empty(shape=(2,2), dtype=object)
            base = '_greens_function_%s.out'%uuid[:8]
            for (i, s) in enumerate(self.keys[0]):
                for (j, t) in enumerate(self.keys[1]):
                    name = 'spin_%s_%s'%(s,t) + base
                    self.rspace_units[i,j] = open(name, 'ab')
                    self.rspace_units[i,j].write(json_string.encode('utf-8'))
                    if self.kspace:
                        self.kspace_units[i,j] = open('kspace_'+name, 'ab')
                        self.kspace_units[i,j].write(json_string.encode('utf-8'))

    def calculate_spgf_unstable(self, state, psi_hist, psi_left):
        """Calculate imaginary time single-particle green's function.

        This uses the naive unstable algorithm.

        Parameters
        ----------
        state : :class:`afqmcpy.state.State`
            state object
        psi : list of :class:`afqmcpy.walker.Walker` objects
            current distribution of walkers, i.e., at the current iteration in the
            simulation corresponding to :math:`\tau_r'=\tau_n+\tau+\tau_{bp}`.
        psi_right : list of :class:`afqmcpy.walker.Walker` objects
            previous distribution of walkers, i.e., at :math:`\tau_n`.
        psi_left : list of :class:`afqmcpy.walker.Walker` objects
            backpropagated walkers projected to :math:`\tau_{bp}`.

        On return the spgf estimator array will have been updated.
        """

        I = numpy.identity(state.system.nbasis)
        nup = state.system.nup
        denom = sum(w.weight for w in psi_hist[:,-1])
        for ix, (w, wr, wl) in enumerate(zip(psi_hist[:,-1], psi_hist[:,0], psi_left)):
            # Initialise time-displaced GF for current walker.
            Ggr = [I, I]
            Gls = [I, I]
            # 1. Construct psi_left for first step in algorithm by back
            # propagating the input back propagated left hand wfn.
            # Note we use the first itcf_nmax fields for estimating the ITCF.
            for (ic, c) in reversed(list(enumerate(psi_hist[ix,1:self.nmax+1]))):
                # propagators should be applied in reverse order
                B = afqmcpy.propagation.construct_propagator_matrix(state,
                                                                    c.field_config,
                                                                    conjt=True)
                afqmcpy.propagation.propagate_single(state, wl, B)
            # 2. Calculate G(n,n). This is the equal time Green's function at
            # the step where we began saving auxilary fields (constructed with
            # psi_left back propagated along this path.)
            Ggr[0] = I - gab(wl.phi[:,:nup], wr.phi[:,:nup])
            Ggr[1] = I - gab(wl.phi[:,nup:], wr.phi[:,nup:])
            Gls[0] = I - Ggr[0]
            Gls[1] = I - Ggr[1]
            self.spgf[0,0,0] = self.spgf[0,0,0] + w.weight*Ggr[0].real
            self.spgf[0,1,0] = self.spgf[0,1,0] + w.weight*Ggr[1].real
            self.spgf[0,0,1] = self.spgf[0,0,1] + w.weight*Gls[0].real
            self.spgf[0,1,1] = self.spgf[0,1,1] + w.weight*Gls[1].real
            # 3. Construct ITCF by moving forwards in imaginary time from time
            # slice n along our auxiliary field path.
            for (ic, c) in enumerate(psi_hist[ix,1:self.nmax+1]):
                # B takes the state from time n to time n+1.
                B = afqmcpy.propagation.construct_propagator_matrix(state,
                                                                c.field_config)
                Ggr[0] = B[0].dot(Ggr[0])
                Ggr[1] = B[1].dot(Ggr[1])
                Gls[0] = Gls[0].dot(scipy.linalg.inv(B[0]))
                Gls[1] = Gls[1].dot(scipy.linalg.inv(B[1]))
                self.spgf[ic+1,0,0] = self.spgf[ic+1,0,0] + w.weight*Ggr[0].real
                self.spgf[ic+1,1,0] = self.spgf[ic+1,1,0] + w.weight*Ggr[1].real
                self.spgf[ic+1,0,1] = self.spgf[ic+1,0,1] + w.weight*Gls[0].real
                self.spgf[ic+1,1,1] = self.spgf[ic+1,1,1] + w.weight*Gls[1].real
            # zero the counter to start accumulating fields again in the
            # following iteration.
            w.bp_counter = 0
        self.spgf = self.spgf / denom

    def calculate_spgf(self, state, psi_hist, psi_left):
        """Calculate imaginary time single-particle green's function.

        This uses the stable algorithm as outlined in: Feldbacher and Assad,
        Phys. Rev. B 63, 073105.

        Parameters
        ----------
        state : :class:`afqmcpy.state.State`
            state object
        psi : list of :class:`afqmcpy.walker.Walker` objects
            current distribution of walkers, i.e., at the current iteration in the
            simulation corresponding to :math:`\tau_r'=\tau_n+\tau+\tau_{bp}`.
        psi_right : list of :class:`afqmcpy.walker.Walker` objects
            previous distribution of walkers, i.e., at :math:`\tau_n`.
        psi_left : list of :class:`afqmcpy.walker.Walker` objects
            backpropagated walkers projected to :math:`\tau_{bp}`.

        On return the spgf estimator array will have been updated.
        """

        I = numpy.identity(state.system.nbasis)
        Gnn = [I, I]
        Bi = [I, I]
        # Be careful not to modify right hand wavefunctions field
        # configurations.
        nup = state.system.nup
        denom = sum(w.weight for w in psi_hist[:,-1])
        for ix, (w, wr, wl) in enumerate(zip(psi_hist[:,-1], psi_hist[:,0], psi_left)):
            # Initialise time-displaced less and greater GF for current walker.
            Gls = [I, I]
            Ggr = [I, I]
            # Store for intermediate back propagated left-hand wavefunctions.
            # This leads to more stable equal time green's functions compared to
            # by multiplying psi_L^n by B^{-1}(x^(n)) factors.
            psi_Ls = []
            # 1. Construct psi_L for first step in algorithm by back
            # propagating the input back propagated left hand wfn.
            # Note we use the first itcf_nmax fields for estimating the ITCF.
            for (ic, c) in reversed(list(enumerate(psi_hist[ix,1:self.nmax+1]))):
                # propagators should be applied in reverse order
                B = afqmcpy.propagation.construct_propagator_matrix(state,
                                                                    c.field_config,
                                                                    conjt=True)
                afqmcpy.propagation.propagate_single(state, wl, B)
                if ic % state.qmc.nstblz == 0:
                    wl.reortho(nup)
                psi_Ls.append(copy.deepcopy(wl))
            # 2. Calculate G(n,n). This is the equal time Green's function at
            # the step where we began saving auxilary fields (constructed with
            # psi_L back propagated along this path.)
            Gnn[0] = I - gab(wl.phi[:,:nup], wr.phi[:,:nup])
            Gnn[1] = I - gab(wl.phi[:,nup:], wr.phi[:,nup:])
            self.spgf[0,0,0] = self.spgf[0,0,0] + w.weight*Gnn[0].real
            self.spgf[0,1,0] = self.spgf[0,1,0] + w.weight*Gnn[1].real
            self.spgf[0,0,1] = self.spgf[0,0,1] + w.weight*(I-Gnn[0]).real
            self.spgf[0,1,1] = self.spgf[0,1,1] + w.weight*(I-Gnn[1]).real
            # 3. Construct ITCF by moving forwards in imaginary time from time
            # slice n along our auxiliary field path.
            for (ic, c) in enumerate(psi_hist[ix,1:self.nmax+1]):
                # B takes the state from time n to time n+1.
                B = afqmcpy.propagation.construct_propagator_matrix(state,
                                                                    c.field_config)
                Bi[0] = scipy.linalg.inv(B[0])
                Bi[1] = scipy.linalg.inv(B[1])
                # G is the cumulative product of stabilised short-time ITCFs.
                # The first term in brackets is the G(n+1,n) which should be
                # well conditioned.
                Ggr[0] = (B[0].dot(Gnn[0])).dot(Ggr[0])
                Ggr[1] = (B[1].dot(Gnn[1])).dot(Ggr[1])
                Gls[0] = ((I-Gnn[0]).dot(Bi[0])).dot(Gls[0])
                Gls[1] = ((I-Gnn[1]).dot(Bi[1])).dot(Gls[1])
                self.spgf[ic+1,0,0] = self.spgf[ic+1,0,0] + w.weight*Ggr[0].real
                self.spgf[ic+1,1,0] = self.spgf[ic+1,1,0] + w.weight*Ggr[1].real
                self.spgf[ic+1,0,1] = self.spgf[ic+1,0,1] + w.weight*Gls[0].real
                self.spgf[ic+1,1,1] = self.spgf[ic+1,1,1] + w.weight*Gls[1].real
                # Construct equal-time green's function shifted forwards along
                # the imaginary time interval. We need to update |psi_L> =
                # (B(c)^{dagger})^{-1}|psi_L> and |psi_R> = B(c)|psi_L>, where c
                # is the current configution in this loop. Note that we store
                # |psi_L> along the path, so we don't need to remove the
                # propagator matrices.
                L = psi_Ls[len(psi_Ls)-ic-1]
                afqmcpy.propagation.propagate_single(state, wr, B)
                if ic % state.qmc.nstblz == 0:
                    wr.reortho(nup)
                Gnn[0] = I - gab(L.phi[:,:nup], wr.phi[:,:nup])
                Gnn[1] = I - gab(L.phi[:,nup:], wr.phi[:,nup:])
        self.spgf = self.spgf / denom

    def to_file(self, spgf, funit, dt):
        """Save ITCF to file.

        This appends to any previous estimates from the same simulation.

        Stolen from https://stackoverflow.com/a/3685339

        Parameters
        ----------
        itcf : numpy array
            Flattened ITCF.
        dt : float
            Time step
        gf_shape: tuple
            Actual shape of ITCF Green's function matrix.
        funit : file
            Output file for ITCF.
        mode : string or list
            if mode == 'full' we print the full green's function else we'll
            print some elements of G.
        """
        for (ic, g) in enumerate(spgf):
            funit.write(('# tau = %4.2f\n'%(ic*dt)).encode('utf-8'))
            # Maybe look at binary / hdf5 format if things get out of hand.
            if self.mode == 'full':
                numpy.savetxt(self.funit, g)
            elif self.mode == 'diagonal':
                output = afqmcpy.utils.format_fixed_width_floats(numpy.diag(g))
                funit.write((output+'\n').encode('utf-8'))
            else:
                output = afqmcpy.utils.format_fixed_width_floats(g[self.mode])
                funit.write((output+'\n').encode('utf-8'))

def local_energy(system, G):
    '''Calculate local energy of walker for the Hubbard model.

Parameters
----------
system : :class:`Hubbard`
    System information for the Hubbard model.
G : :class:`numpy.ndarray`
    Greens function for given walker phi, i.e.,
    :math:`G=\langle \phi_T| c_i^{\dagger}c_j | \phi\rangle`.

Returns
-------
E_L(phi) : float
    Local energy of given walker phi.
'''

    ke = numpy.sum(system.T * (G[0] + G[1]))
    pe = sum(system.U*G[0][i][i]*G[1][i][i] for i in range(0, system.nbasis))

    return (ke + pe, ke, pe)


def gab(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,
    .. math::
        \langle phi_A|c_i^{\dagger}c_j|phi_B\rangle = [B(A^{*T}B)^{-1}A^{*T}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    # Todo: check energy evaluation at later point, i.e., if this needs to be
    # transposed. Shouldn't matter for Hubbard model.
    inv_O = scipy.linalg.inv((A.conj().T).dot(B))
    GAB = B.dot(inv_O.dot(A.conj().T))
    return GAB


def print_key(key, print_function=print, eol='', encode=False):
    """Print out information about what the estimates are.

    Parameters
    ----------
    back_propagation : bool, optional
        True if doing back propagation. Default : False.
    print_function : method, optional
        How to print state information, e.g. to std out or file. Default : print.
    eol : string, optional
        String to append to output, e.g., '\n', Default : ''.
    """
    header = (
        eol + '# Explanation of output column headers:\n' +
        '# -------------------------------------' + eol
    )
    if encode:
        header = header.encode('utf-8')
    print_function(header)
    for (k, v) in key.items():
        s = '# %s : %s'%(k, v) + eol
        if encode:
            s = s.encode('utf-8')
        print_function(s)


def print_header(header, print_function=print, eol='', encode=False):
    """Print out header for estimators

    Parameters
    ----------
    back_propagation : bool, optional
        True if doing back propagation. Default : False.
    print_function : method, optional
        How to print state information, e.g. to std out or file. Default : print.
    eol : string, optional
        String to append to output, e.g., '\n', Default : ''.
    """
    s = afqmcpy.utils.format_fixed_width_strings(header) + eol
    if encode:
        s = s.encode('utf-8')
    print_function(s)

def eproj(estimates, enum):
    """Real projected energy.

    Parameters
    ----------
    estimates : numpy.array
        Array containing estimates averaged over all processors.
    enum : :class:`afqmcpy.estimators.EstimatorEnum` object
        Enumerator class outlining indices of estimates array elements.

    Returns
    -------
    eproj : float
        Projected energy from current estimates array.
    """

    numerator = estimates[enum.enumer]
    denominator = estimates[enum.edenom]
    return (numerator/denominator).real
