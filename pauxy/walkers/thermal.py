import copy
import cmath
import numpy
import scipy.linalg
from pauxy.utils.linalg import regularise_matrix_inverse
from pauxy.estimators.thermal import greens_function, one_rdm_from_G, particle_number
from pauxy.estimators.mixed import local_energy
from pauxy.walkers.stack import PropagatorStack

class ThermalWalker(object):

    def __init__(self, walker_opts, system, trial, verbose=False):
        self.weight = walker_opts.get('weight', 1.0)
        self.phase = 1.0 + 0.0j
        self.alive = True
        self.num_slices = trial.num_slices
        if system.name == "UEG" or system.name == "Generic":
            dtype = numpy.complex128
        else:
            dtype = numpy.float64
        self.G = numpy.zeros(trial.dmat.shape, dtype=dtype)
        self.nbasis = trial.dmat[0].shape[0]
        self.total_weight = 0
        self.stack_size = walker_opts.get('stack_size', None)
        max_diff_diag = numpy.linalg.norm((numpy.diag(trial.dmat[0].diagonal())-trial.dmat[0]))
        if max_diff_diag < 1e-10:
            self.diagonal_trial = True
        else:
            self.diagonal_trial = False

        if self.stack_size == None:
            self.stack_size = trial.stack_size
        elif self.stack_size > trial.stack_size:
            if verbose:
                print("# Walker stack size differs from that estimated from "
                      "trial density matrix. Be careful.")
        self.stack_length = self.num_slices // self.stack_size

        if verbose and self.diagonal_trial:
            print("# Trial density matrix is diagonal.")
        self.stack = PropagatorStack(self.stack_size, trial.num_slices,
                                     trial.dmat.shape[-1], dtype,
                                     trial.dmat, trial.dmat_inv,
                                     diagonal=self.diagonal_trial)

        # Initialise all propagators to the trial density matrix.
        self.stack.set_all(trial.dmat)
        self.greens_function(trial)
        self.ot = 1.0

        # temporary storage for stacks...
        self.Tl = [numpy.identity(trial.dmat[0].shape[0]), numpy.identity(trial.dmat[1].shape[0])]
        self.Ql = [numpy.identity(trial.dmat[0].shape[0]), numpy.identity(trial.dmat[1].shape[0])]
        self.Dl = [numpy.identity(trial.dmat[0].shape[0]), numpy.identity(trial.dmat[1].shape[0])]
        self.Tr = [numpy.identity(trial.dmat[0].shape[0]), numpy.identity(trial.dmat[1].shape[0])]
        self.Qr = [numpy.identity(trial.dmat[0].shape[0]), numpy.identity(trial.dmat[1].shape[0])]
        self.Dr = [numpy.identity(trial.dmat[0].shape[0]), numpy.identity(trial.dmat[1].shape[0])]

    def greens_function(self, trial, slice_ix=None, inplace=True):
        return self.greens_function_qr_strat(trial, slice_ix=slice_ix,
                                             inplace=inplace)

    def greens_function_svd(self, trial, slice_ix=None, inplace=True):
        if slice_ix == None:
            slice_ix = self.stack.time_slice
        bin_ix = slice_ix // self.stack.stack_size
        # For final time slice want first block to be the rightmost (for energy
        # evaluation).
        if bin_ix == self.stack.nbins:
            bin_ix = -1
        if inplace:
            G = None
        else:
            G = numpy.zeros(self.G.shape, self.G.dtype)
        for spin in [0, 1]:
            # Need to construct the product A(l) = B_l B_{l-1}..B_L...B_{l+1}
            # in stable way. Iteratively construct SVD decompositions starting
            # from the rightmost (product of) propagator(s).
            B = self.stack.get((bin_ix+1)%self.stack.nbins)
            (U1, S1, V1) = scipy.linalg.svd(B[spin])
            for i in range(2, self.stack.nbins+1):
                ix = (bin_ix + i) % self.stack.nbins
                B = self.stack.get(ix)
                T1 = numpy.dot(B[spin], U1)
                # todo optimise
                T2 = numpy.dot(T1, numpy.diag(S1))
                (U1, S1, V) = scipy.linalg.svd(T2)
                V1 = numpy.dot(V, V1)
            A = numpy.dot(U1.dot(numpy.diag(S1)), V1)
            # Final SVD decomposition to construct G(l) = [I + A(l)]^{-1}.
            # Care needs to be taken when adding the identity matrix.
            T3 = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
            (U2, S2, V2) = scipy.linalg.svd(T3)
            U3 = numpy.dot(U1, U2)
            D3 = numpy.diag(1.0/S2)
            V3 = numpy.dot(V2, V1)
            # G(l) = (U3 S2 V3)^{-1}
            #      = V3^{\dagger} D3 U3^{\dagger}
            if inplace:
                # self.G[spin] = (V3inv).dot(U3.conj().T)
                self.G[spin] = (V3.conj().T).dot(D3).dot(U3.conj().T)
            else:
                # G[spin] = (V3inv).dot(U3.conj().T)
                G[spin] = (V3.conj().T).dot(D3).dot(U3.conj().T)
        return G

    def greens_function_qr(self, trial, slice_ix=None, inplace=True):
        if (slice_ix == None):
            slice_ix = self.stack.time_slice

        bin_ix = slice_ix // self.stack.stack_size
        # For final time slice want first block to be the rightmost (for energy
        # evaluation).
        if bin_ix == self.stack.nbins:
            bin_ix = -1
        if not inplace:
            G = numpy.zeros(self.G.shape, self.G.dtype)
        else:
            G = None
        for spin in [0, 1]:
            # Need to construct the product A(l) = B_l B_{l-1}..B_L...B_{l+1}
            # in stable way. Iteratively construct SVD decompositions starting
            # from the rightmost (product of) propagator(s).
            B = self.stack.get((bin_ix+1)%self.stack.nbins)
            (U1, V1) = scipy.linalg.qr(B[spin], pivoting = False, check_finite = False)

            for i in range(2, self.stack.nbins+1):
                ix = (bin_ix + i) % self.stack.nbins
                B = self.stack.get(ix)
                T1 = numpy.dot(B[spin], U1)
                (U1, V) = scipy.linalg.qr(T1, pivoting = False, check_finite = False)
                V1 = numpy.dot(V, V1)

            # Final SVD decomposition to construct G(l) = [I + A(l)]^{-1}.
            # Care needs to be taken when adding the identity matrix.
            V1inv = scipy.linalg.solve_triangular(V1, numpy.identity(V1.shape[0]))

            T3 = numpy.dot(U1.conj().T, V1inv) + numpy.identity(V1.shape[0])
            (U2, V2) = scipy.linalg.qr(T3, pivoting = False, check_finite = False)

            U3 = numpy.dot(U1, U2)
            V3 = numpy.dot(V2, V1)
            V3inv = scipy.linalg.solve_triangular(V3, numpy.identity(V3.shape[0]))
            # G(l) = (U3 S2 V3)^{-1}
            #      = V3^{\dagger} D3 U3^{\dagger}
            if inplace:
                self.G[spin] = (V3inv).dot(U3.conj().T)
            else:
                G[spin] = (V3inv).dot(U3.conj().T)
        return G

    def compute_left_right(self, center_ix):
        # Use Stratification method (DOI 10.1109/IPDPS.2012.37)
        # B(L) .... B(1)
        for spin in [0, 1]:
            # right bit
            # B(right) ... B(1)
            if (center_ix > 0):
                # print ("center_ix > 0")
                B = self.stack.get(0)
                (self.Qr[spin], R1, P1) = scipy.linalg.qr(B[spin], pivoting=True, check_finite=False)
                # Form D matrices
                self.Dr[spin] = numpy.diag(R1.diagonal())
                D1inv = numpy.diag(1.0/R1.diagonal())
                self.Tr[spin] = numpy.einsum('ii,ij->ij',D1inv, R1)
                # now permute them
                self.Tr[spin][:,P1] = self.Tr[spin] [:,range(self.nbasis)]

                for ix in range(1, center_ix):
                    B = self.stack.get(ix)
                    C2 = numpy.einsum('ij,jj->ij',
                        numpy.dot(B[spin], self.Qr[spin]),
                        self.Dr[spin])
                    (self.Qr[spin], R1, P1) = scipy.linalg.qr(C2, pivoting=True, check_finite=False)
                    # Compute D matrices
                    D1inv = numpy.diag(1.0/R1.diagonal())
                    self.Dr[spin] = numpy.diag(R1.diagonal())
                    # smarter permutation
                    # D^{-1} * R
                    tmp = numpy.einsum('ii,ij->ij',D1inv, R1)
                    # D^{-1} * R * P^T
                    tmp[:,P1] = tmp[:,range(self.nbasis)]
                    # D^{-1} * R * P^T * T
                    self.Tr[spin] = numpy.dot(tmp, self.Tr[spin])

            # left bit
            # B(l) ... B(left)
            if (center_ix < self.stack.nbins-1):
                # print("center_ix < self.stack.nbins-1 first")
                # We will assume that B matrices are all diagonal for left....
                B = self.stack.get(center_ix+1)
                self.Dl[spin] = numpy.diag(B[spin].diagonal())
                D1inv = numpy.diag(1.0/B[spin].diagonal())
                self.Ql[spin] = numpy.identity(B[spin].shape[0])
                self.Tl[spin] = numpy.identity(B[spin].shape[0])

                for ix in range(center_ix+2, self.stack.nbins):
                    # print("center_ix < self.stack.nbins-1 first inner loop")
                    B = self.stack.get(ix)
                    C2 = numpy.diag(numpy.einsum('ii,ii->i',B[spin],self.Dl[spin]))
                    R1 = numpy.diag(C2.diagonal())
                    D1inv = numpy.diag(1.0/R1.diagonal())
                    self.Dl[spin] = numpy.diag(R1.diagonal())

    def compute_right(self, center_ix):
        # Use Stratification method (DOI 10.1109/IPDPS.2012.37)
        # B(L) .... B(1)
        for spin in [0, 1]:
            # right bit
            # B(right) ... B(1)
            if (center_ix > 0):
                # print ("center_ix > 0")
                B = self.stack.get(0)
                (self.Qr[spin], R1, P1) = scipy.linalg.qr(B[spin], pivoting=True, check_finite=False)
                # Form D matrices
                self.Dr[spin] = numpy.diag(R1.diagonal())
                D1inv = numpy.diag(1.0/R1.diagonal())
                self.Tr[spin] = numpy.einsum('ii,ij->ij',D1inv, R1)
                # now permute them
                self.Tr[spin][:,P1] = self.Tr[spin] [:,range(self.nbasis)]

                for ix in range(1, center_ix):
                    B = self.stack.get(ix)
                    C2 = numpy.einsum('ij,jj->ij',
                        numpy.dot(B[spin], self.Qr[spin]),
                        self.Dr[spin])
                    (self.Qr[spin], R1, P1) = scipy.linalg.qr(C2, pivoting=True, check_finite=False)
                    # Compute D matrices
                    D1inv = numpy.diag(1.0/R1.diagonal())
                    self.Dr[spin] = numpy.diag(R1.diagonal())
                    # smarter permutation
                    # D^{-1} * R
                    tmp = numpy.einsum('ii,ij->ij',D1inv, R1)
                    # D^{-1} * R * P^T
                    tmp[:,P1] = tmp[:,range(self.nbasis)]
                    # D^{-1} * R * P^T * T
                    self.Tr[spin] = numpy.dot(tmp, self.Tr[spin])

    def compute_left(self, center_ix):
        # Use Stratification method (DOI 10.1109/IPDPS.2012.37)
        # B(L) .... B(1)
        for spin in [0, 1]:
            # left bit
            # B(l) ... B(left)
            if (center_ix < self.stack.nbins-1):
                # print("center_ix < self.stack.nbins-1 first")
                # We will assume that B matrices are all diagonal for left....
                B = self.stack.get(center_ix+1)
                self.Dl[spin] = numpy.diag(B[spin].diagonal())
                D1inv = numpy.diag(1.0/B[spin].diagonal())
                self.Ql[spin] = numpy.identity(B[spin].shape[0])
                self.Tl[spin] = numpy.identity(B[spin].shape[0])

                for ix in range(center_ix+2, self.stack.nbins):
                    # print("center_ix < self.stack.nbins-1 first inner loop")
                    B = self.stack.get(ix)
                    C2 = numpy.diag(numpy.einsum('ii,ii->i',B[spin],self.Dl[spin]))
                    R1 = numpy.diag(C2.diagonal())
                    D1inv = numpy.diag(1.0/R1.diagonal())
                    self.Dl[spin] = numpy.diag(R1.diagonal())

    def greens_function_left_right(self, center_ix, inplace=False):
        if not inplace:
            G = numpy.zeros(self.G.shape, self.G.dtype)
        else:
            G = None

        Bc = self.stack.get(center_ix)
        for spin in [0,1]:
            if (center_ix > 0): # there exists right bit
                # print("center_ix > 0 second")
                Ccr = numpy.einsum('ij,jj->ij',
                    numpy.dot(Bc[spin],self.Qr[spin]),
                    self.Dr[spin])
                (Qlcr, Rlcr, Plcr) = scipy.linalg.qr(Ccr, pivoting=True, check_finite=False)
                Dlcr = numpy.diag(Rlcr.diagonal())
                Dinv = numpy.diag(1.0/Rlcr.diagonal())
                tmp = numpy.einsum('ii,ij->ij',Dinv, Rlcr)
                tmp[:,Plcr] = tmp[:,range(self.nbasis)]
                Tlcr = numpy.dot(tmp, self.Tr[spin])
            else:
                # print("center_ix > 0 else second")
                (Qlcr, Rlcr, Plcr) = scipy.linalg.qr(Bc[spin], pivoting=True, check_finite=False)
                # Form D matrices
                Dlcr = numpy.diag(Rlcr.diagonal())
                Dinv = numpy.diag(1.0/Rlcr.diagonal())
                Tlcr = numpy.einsum('ii,ij->ij',Dinv, Rlcr)
                Tlcr[:,Plcr] = Tlcr[:,range(self.nbasis)]

            if (center_ix < self.stack.nbins-1): # there exists left bit
                # print("center_ix < self.stack.nbins-1 second")
                # assume left stack is all diagonal
                Clcr = numpy.einsum('ii,ij->ij',
                        self.Dl[spin],
                        numpy.einsum('ij,jj->ij',Qlcr, Dlcr))

                (Qlcr, Rlcr, Plcr) = scipy.linalg.qr(Clcr, pivoting=True, check_finite=False)
                Dlcr = numpy.diag(Rlcr.diagonal())
                Dinv = numpy.diag(1.0/Rlcr.diagonal())

                tmp = numpy.einsum('ii,ij->ij',Dinv, Rlcr)
                tmp[:,Plcr] = tmp[:,range(self.nbasis)]
                Tlcr = numpy.dot(tmp, Tlcr)

            # G^{-1} = 1+A = 1+QDT = Q (Q^{-1}T^{-1}+D) T
            # Write D = Db^{-1} Ds
            # Then G^{-1} = Q Db^{-1}(Db Q^{-1}T^{-1}+Ds) T
            Db = numpy.zeros(Bc[spin].shape, Bc[spin].dtype)
            Ds = numpy.zeros(Bc[spin].shape, Bc[spin].dtype)
            for i in range(Db.shape[0]):
                absDlcr = abs(Dlcr[i,i])
                if absDlcr > 1.0:
                    Db[i,i] = 1.0 / absDlcr
                    Ds[i,i] = numpy.sign(Dlcr[i,i])
                else:
                    Db[i,i] = 1.0
                    Ds[i,i] = Dlcr[i,i]

            T1inv = scipy.linalg.inv(Tlcr, check_finite=False)
            # C = (Db Q^{-1}T^{-1}+Ds)
            C = numpy.dot(
                numpy.einsum('ii,ij->ij',Db, Qlcr.conj().T),
                T1inv) + Ds
            Cinv = scipy.linalg.inv(C, check_finite = False)

            # Then G = T^{-1} C^{-1} Db Q^{-1}
            # Q is unitary.
            if inplace:
                self.G[spin] = numpy.dot(numpy.dot(T1inv, Cinv),
                             numpy.einsum('ii,ij->ij',Db, Qlcr.conj().T))
            else:
                G[spin] = numpy.dot(numpy.dot(T1inv, Cinv),
                            numpy.einsum('ii,ij->ij',Db, Qlcr.conj().T))
        return G

    def greens_function_qr_strat(self, trial, slice_ix=None, inplace=True):
        # Use Stratification method (DOI 10.1109/IPDPS.2012.37)
        if (slice_ix == None):
            slice_ix = self.stack.time_slice

        bin_ix = slice_ix // self.stack.stack_size
        # For final time slice want first block to be the rightmost (for energy
        # evaluation).
        if bin_ix == self.stack.nbins:
            bin_ix = -1

        if not inplace:
            G = numpy.zeros(self.G.shape, self.G.dtype)
        else:
            G = None

        for spin in [0, 1]:
            # Need to construct the product A(l) = B_l B_{l-1}..B_L...B_{l+1} in
            # stable way. Iteratively construct column pivoted QR decompositions
            # (A = QDT) starting from the rightmost (product of) propagator(s).
            B = self.stack.get((bin_ix+1)%self.stack.nbins)

            (Q1, R1, P1) = scipy.linalg.qr(B[spin], pivoting=True,
                                           check_finite=False)
            # Form D matrices
            D1 = numpy.diag(R1.diagonal())
            D1inv = numpy.diag(1.0/R1.diagonal())
            T1 = numpy.einsum('ii,ij->ij', D1inv, R1)
            # permute them
            T1[:,P1] = T1 [:, range(self.nbasis)]

            for i in range(2, self.stack.nbins+1):
                ix = (bin_ix + i) % self.stack.nbins
                B = self.stack.get(ix)
                C2 = numpy.dot(numpy.dot(B[spin], Q1), D1)
                (Q1, R1, P1) = scipy.linalg.qr(C2, pivoting=True,
                                               check_finite=False)
                # Compute D matrices
                D1inv = numpy.diag(1.0/R1.diagonal())
                D1 = numpy.diag(R1.diagonal())
                tmp = numpy.einsum('ii,ij->ij',D1inv, R1)
                tmp[:,P1] = tmp[:,range(self.nbasis)]
                T1 = numpy.dot(tmp, T1)

            # G^{-1} = 1+A = 1+QDT = Q (Q^{-1}T^{-1}+D) T
            # Write D = Db^{-1} Ds
            # Then G^{-1} = Q Db^{-1}(Db Q^{-1}T^{-1}+Ds) T
            Db = numpy.zeros(B[spin].shape, B[spin].dtype)
            Ds = numpy.zeros(B[spin].shape, B[spin].dtype)
            for i in range(Db.shape[0]):
                absDlcr = abs(Db[i,i])
                if absDlcr > 1.0:
                    Db[i,i] = 1.0 / absDlcr
                    Ds[i,i] = numpy.sign(D1[i,i])
                else:
                    Db[i,i] = 1.0
                    Ds[i,i] = D1[i,i]

            T1inv = scipy.linalg.inv(T1, check_finite = False)
            # C = (Db Q^{-1}T^{-1}+Ds)
            C = numpy.dot(numpy.einsum('ii,ij->ij',Db, Q1.conj().T), T1inv) + Ds
            Cinv = scipy.linalg.inv(C, check_finite = False)

            # Then G = T^{-1} C^{-1} Db Q^{-1}
            # Q is unitary.
            if inplace:
                self.G[spin] = numpy.dot(numpy.dot(T1inv, Cinv),
                                         numpy.einsum('ii,ij->ij', Db, Q1.conj().T))
            else:
                G[spin] = numpy.dot(numpy.dot(T1inv, Cinv),
                                    numpy.einsum('ii,ij->ij', Db, Q1.conj().T))
        return G

    def local_energy(self, system, two_rdm=None):
        rdm = one_rdm_from_G(self.G)
        return local_energy(system, rdm, two_rdm=two_rdm, opt=False)

    def get_buffer(self):
        """Get walker buffer for MPI communication

        Returns
        -------
        buff : dict
            Relevant walker information for population control.
        """
        buff = {
            'stack': self.stack.get_buffer(),
            'G': self.G,
            'weight': self.weight,
            'phase': self.phase,
            'Tl': self.Tl,
            'Ql': self.Ql,
            'Dl': self.Dl,
            'Tr': self.Tr,
            'Qr': self.Qr,
            'Dr': self.Dr
        }
        return buff

    def set_buffer(self, buff):
        """Set walker buffer following MPI communication

        Parameters
        -------
        buff : dict
            Relevant walker information for population control.
        """
        self.stack.set_buffer(buff['stack'])
        self.G = numpy.copy(buff['G'])
        self.weight = buff['weight']
        self.phase = buff['phase']
        self.Tl = [numpy.copy(buff['Tl'][0]), numpy.copy(buff['Tl'][1])]
        self.Ql = [numpy.copy(buff['Ql'][0]), numpy.copy(buff['Ql'][1])]
        self.Dl = [numpy.copy(buff['Dl'][0]), numpy.copy(buff['Dl'][1])]
        self.Tr = [numpy.copy(buff['Tr'][0]), numpy.copy(buff['Tr'][1])]
        self.Qr = [numpy.copy(buff['Qr'][0]), numpy.copy(buff['Qr'][1])]
        self.Dr = [numpy.copy(buff['Dr'][0]), numpy.copy(buff['Dr'][1])]

def unit_test():
    from pauxy.systems.ueg import UEG
    from pauxy.trial_density_matrices.onebody import OneBody
    from pauxy.thermal_propagation.planewave import PlaneWave
    from pauxy.qmc.options import QMCOpts

    inputs = {'nup':1,
    'ndown':1,
    'rs':1.0,
    'ecut':0.5,
    "name": "one_body",
    "mu":1.94046021,
    "beta":0.5,
    "dt": 0.05,
    "optimised": True
    }
    beta = inputs ['beta']
    dt = inputs['dt']

    system = UEG(inputs, verbose = False)

    qmc = QMCOpts(inputs, system, True)
    trial = OneBody(inputs, system, beta, dt, system.H1, verbose=False)

    walker = ThermalWalker(inputs, system, trial, True)
    # walker.greens_function(trial)
    E, T, V = walker.local_energy(system)
    numpy.random.seed(0)
    inputs['optimised'] = False
    propagator = PlaneWave(inputs, qmc, system, trial, verbose=False)

    propagator.propagate_walker_free(system, walker, trial, False)

    Gold = walker.G[0].copy()

    system = UEG(inputs, verbose=False)

    qmc = QMCOpts(inputs, system, verbose=False)
    trial = OneBody(inputs, system, beta, dt, system.H1, verbose=False)


    propagator = PlaneWave(inputs, qmc, system, trial, True)
    walker = ThermalWalker(inputs, system, trial, verbose=False)
    # walker.greens_function(trial)
    E, T, V = walker.local_energy(system)
    numpy.random.seed(0)
    inputs['optimised'] = True
    propagator = PlaneWave(inputs, qmc, system, trial, verbose=False)

    propagator.propagate_walker_free(system, walker, trial, False)

    Gnew = walker.G[0].copy()

    assert(scipy.linalg.norm(Gold[:,0] - Gnew[:,0]) < 1e-10)

    inputs['stack_size'] = 1
    walker = ThermalWalker(inputs, system, trial, verbose=False)
    numpy.random.seed(0)
    propagator = PlaneWave(inputs, qmc, system, trial, verbose=False)
    for i in range(0,5):
        propagator.propagate_walker(system, walker, trial)
    Gs1 = walker.G[0].copy()
    for ts in range(walker.stack_length):
        walker.greens_function(trial, slice_ix=ts*walker.stack_size)
        E, T, V = walker.local_energy(system)
        # print(E)

    inputs['stack_size'] = 5
    walker = ThermalWalker(inputs, system, trial, verbose=False)
    numpy.random.seed(0)
    propagator = PlaneWave(inputs, qmc, system, trial, verbose=False)
    for i in range(0,5):
        propagator.propagate_walker(system, walker, trial)
    Gs5 = walker.G[0].copy()
    for ts in range(walker.stack_length):
        walker.greens_function(trial, slice_ix=ts*walker.stack_size)
        E, T, V = walker.local_energy(system)
        # print(E)
    assert(numpy.linalg.norm(Gs1-Gs5) < 1e-10)

    N = 5
    A = numpy.random.rand(N,N)
    Q, R, P = scipy.linalg.qr(A, pivoting=True)

#### test permutation start
    # Pmat = numpy.zeros((N,N))
    # for i in range (N):
    #     Pmat[P[i],i] = 1
    # print(P)
    # tmp = Q.dot(R)#.dot(Pmat.T)
    # print(tmp)
    # print("==================")
    # tmp2 = tmp.dot(Pmat.T)
    # print(tmp2)
    # print("==================")
    # tmp[:,P] = tmp [:,range(N)]
    # print(tmp)
#### test permutation end

    B = numpy.random.rand(N,N)
    (Q1, R1, P1) = scipy.linalg.qr(B, pivoting=True, check_finite = False)
    # Form permutation matrix
    P1mat = numpy.zeros(B.shape, B.dtype)
    P1mat[P1,range(len(P1))] = 1.0
    # Form D matrices
    D1 = numpy.diag(R1.diagonal())
    D1inv = numpy.diag(1.0/R1.diagonal())
    T1 = numpy.dot(numpy.dot(D1inv, R1), P1mat.T)

    assert(numpy.linalg.norm(B - numpy.einsum('ij,jj->ij',Q1,D1).dot(T1)) < 1e-10)

    # tmp[:,:] = tmp[:,P]
    # print(A - tmp)
    # print(Q * Q.T)
    # print(R)

    # Test walker green's function.
    from pauxy.systems.hubbard import Hubbard
    from pauxy.estimators.thermal import greens_function, one_rdm_from_G
    from pauxy.estimators.hubbard import local_energy_hubbard

    sys_dict = {'name': 'Hubbard', 'nx': 4, 'ny': 4,
                'nup': 7, 'ndown': 7, 'U': 4, 't': 1}
    system = Hubbard(sys_dict)
    beta = 4
    mu = 1
    trial = OneBody({"mu": mu}, system, beta, dt, verbose=True)

    dt = 0.05
    num_slices = int(beta/dt)

    eref = 0
    for ek in system.eks:
        eref += 2 * ek * 1.0 / (numpy.exp(beta*(ek-mu))+1)
    walker = ThermalWalker({"stack_size": 1}, system, trial)
    Gs1 = walker.G[0].copy()
    rdm = one_rdm_from_G(walker.G)
    ekin = local_energy_hubbard(system, rdm)[1]
    try:
        assert(abs(eref-ekin) < 1e-8)
    except AssertionError:
        print("Error in kinetic energy check. Ref: %13.8e Calc:%13.8e"%(eref, ekin))
    walker = ThermalWalker({"stack_size": 10}, system, trial)
    rdm = one_rdm_from_G(walker.G)
    ekin = local_energy_hubbard(system, rdm)[1]
    try:
        assert(abs(eref-ekin) < 1e-8)
    except AssertionError:
        print("Error in kinetic energy check. Ref: %13.10e Calc: %13.10e"
              " Error: %13.8e"%(eref.real, ekin.real, abs(eref-ekin)))
    for ts in range(walker.stack_length):
        walker.greens_function(trial, slice_ix=ts*walker.stack_size)
        assert(numpy.linalg.norm(Gs1-walker.G[0]) < 1e-10)

if __name__=="__main__":
    unit_test()
