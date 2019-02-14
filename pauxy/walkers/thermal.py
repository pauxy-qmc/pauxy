import copy
import cmath
import numpy
import scipy.linalg
from pauxy.utils.linalg import regularise_matrix_inverse
from pauxy.estimators.thermal import greens_function, one_rdm_from_G
from pauxy.estimators.mixed import local_energy

class ThermalWalker(object):

    def __init__(self, walker_opts, system, trial, verbose=False):
        self.weight = walker_opts.get('weight', 1.0)
        self.phase = 1.0 + 0.0j
        self.alive = True
        self.num_slices = trial.ntime_slices
        if verbose:
            print("# Number of slices = {}".format(self.num_slices))
        if system.name == "UEG" or system.name == "Generic":
            dtype = numpy.complex128
        else:
            dtype = numpy.float64
        self.G = numpy.zeros(trial.dmat.shape, dtype=dtype)
        self.nbasis = trial.dmat[0].shape[0]
        self.stack_size = walker_opts.get('stack_size', None)
        max_diff_diag = numpy.linalg.norm((numpy.diag(trial.dmat[0].diagonal())-trial.dmat[0]))
        if max_diff_diag < 1e-10:
            self.diagonal_trial = True
        else:
            self.diagonal_trial = False

        if self.stack_size == None:
            if verbose:
                print("# Stack size is determined by BT")
            emax = numpy.max(numpy.diag(trial.dmat[0]))
            emin = numpy.min(numpy.diag(trial.dmat[0]))
            self.stack_size = min(self.num_slices,
                int(1.5 / ((cmath.log(float(emax)) - cmath.log(float(emin))) / 8.0).real))
            if verbose:
                print("# Initial stack size is {}".format(self.stack_size))
        # adjust stack size
        lower_bound = min(self.stack_size, self.num_slices)
        upper_bound = min(self.stack_size, self.num_slices)

        while (self.num_slices//lower_bound) * lower_bound < self.num_slices:
            lower_bound -= 1
        while (self.num_slices//upper_bound) * upper_bound < self.num_slices:
            upper_bound += 1

        if (self.stack_size-lower_bound) <= (upper_bound - self.stack_size):
            self.stack_size = lower_bound
        else:
            self.stack_size = upper_bound

        self.stack_length = self.num_slices // self.stack_size

        if verbose:
            print("# upper_bound is {}".format(upper_bound))
            print("# lower_bound is {}".format(lower_bound))
            print("# Adjusted stack size is {}".format(self.stack_size))
            print("# Number of stacks is {}".format(self.stack_length))
            # print("# Trial dmat = {}".format(trial.dmat[0]))


        if verbose and self.diagonal_trial:
            print("# Trial density matrix is diagonal.")
        self.stack = PropagatorStack(self.stack_size, trial.ntime_slices,
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

        cond = numpy.linalg.cond(trial.dmat[0])
        if verbose:
            print("# condition number of BT = {}".format(cond))

    def greens_function(self, trial, slice_ix=None, inplace=True):
        if self.diagonal_trial:
            return self.greens_function_qr_strat(trial, slice_ix=slice_ix,
                                                 inplace=inplace)
        else:
            return self.greens_function_svd(trial, slice_ix=slice_ix,
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

            (Q1, R1, P1) = scipy.linalg.qr(B[spin], pivoting=True, check_finite = False)
            # Form D matrices
            D1 = numpy.diag(R1.diagonal())
            D1inv = numpy.diag(1.0/R1.diagonal())
            T1 = numpy.einsum('ii,ij->ij',D1inv, R1)
            # permute them
            T1[:,P1] = T1 [:, range(self.nbasis)]

            for i in range(2, self.stack.nbins+1):
                ix = (bin_ix + i) % self.stack.nbins
                B = self.stack.get(ix)
                C2 = numpy.dot(numpy.dot(B[spin], Q1), D1)
                (Q1, R1, P1) = scipy.linalg.qr(C2, pivoting=True, check_finite = False)
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
                                         numpy.einsum('ii,ij->ij',Db, Q1.conj().T))
            else:
                G[spin] = numpy.dot(numpy.dot(T1inv, Cinv),
                                    numpy.einsum('ii,ij->ij',Db, Q1.conj().T))
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


class PropagatorStack:
    def __init__(self, bin_size, ntime_slices, nbasis, dtype, BT, BTinv,
                 diagonal=False):
        self.time_slice = 0
        self.stack_size = bin_size
        self.ntime_slices = ntime_slices
        self.nbins = ntime_slices // bin_size
        self.diagonal_trial = diagonal

        if self.nbins * self.stack_size < self.ntime_slices:
            print("stack_size must divide the total path length")
            assert(self.nbins * self.stack_size == self.ntime_slices)

        self.nbasis = nbasis
        self.dtype = dtype
        self.BT = BT
        self.BTinv = BTinv
        self.counter = 0
        self.block = 0
        self.stack = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis),
                                 dtype=dtype)
        self.left = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis),
                                dtype=dtype)
        self.right = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis),
                                 dtype=dtype)
        # set all entries to be the identity matrix
        self.reset()

    def get(self, ix):
        return self.stack[ix]

    def get_buffer(self):
        buff = {
            'left': self.left,
            'right': self.right,
            'stack': self.stack,
        }
        return buff

    def set_buffer(self, buff):
        self.stack = numpy.copy(buff['stack'])
        self.left = numpy.copy(buff['left'])
        self.right = numpy.copy(buff['right'])

    def set_all(self, BT):
        # Diagonal = True assumes BT is diagonal and left is also diagonal
        if self.diagonal_trial:
            for i in range(0, self.ntime_slices):
                ix = i // self.stack_size # bin index
                # Commenting out these two. It is only useful for Hubbard
                # self.left[ix,0] = numpy.diag(numpy.einsum("ii,ii->i",BT[0],self.left[ix,0]))
                # self.left[ix,1] = numpy.diag(numpy.einsum("ii,ii->i",BT[1],self.left[ix,1]))
                self.left[ix,0] = numpy.diag(numpy.multiply(BT[0].diagonal(),self.left[ix,0].diagonal()))
                self.left[ix,1] = numpy.diag(numpy.multiply(BT[1].diagonal(),self.left[ix,1].diagonal()))
                self.stack[ix,0] = self.left[ix,0].copy()
                self.stack[ix,1] = self.left[ix,1].copy()
        else:
            for i in range(0, self.ntime_slices):
                ix = i // self.stack_size # bin index
                self.left[ix,0] = numpy.dot(BT[0],self.left[ix,0])
                self.left[ix,1] = numpy.dot(BT[1],self.left[ix,1])
                self.stack[ix,0] = self.left[ix,0].copy()
                self.stack[ix,1] = self.left[ix,1].copy()


    def reset(self):
        self.time_slice = 0
        self.block = 0
        for i in range(0, self.nbins):
            self.stack[i,0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.stack[i,1] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.right[i,0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.right[i,1] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.left[i,0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.left[i,1] = numpy.identity(self.nbasis, dtype=self.dtype)

    def update(self, B):
        if self.counter == 0:
            self.stack[self.block,0] = numpy.identity(B.shape[-1], dtype=B.dtype)
            self.stack[self.block,1] = numpy.identity(B.shape[-1], dtype=B.dtype)
        self.stack[self.block,0] = B[0].dot(self.stack[self.block,0])
        self.stack[self.block,1] = B[1].dot(self.stack[self.block,1])
        self.time_slice = self.time_slice + 1
        self.block = self.time_slice // self.stack_size
        self.counter = (self.counter + 1) % self.stack_size

    def update_new(self, B):
        # Diagonal = True assumes BT is diagonal and left is also diagonal
        if self.counter == 0:
            self.right[self.block,0] = numpy.identity(B.shape[-1], dtype=B.dtype)
            self.right[self.block,1] = numpy.identity(B.shape[-1], dtype=B.dtype)

        if self.diagonal_trial:
            self.left[self.block,0] = numpy.diag(numpy.multiply(self.left[self.block,0].diagonal(),self.BTinv[0].diagonal()))
            self.left[self.block,1] = numpy.diag(numpy.multiply(self.left[self.block,1].diagonal(),self.BTinv[1].diagonal()))
        else:
            self.left[self.block,0] = self.left[self.block,0].dot(self.BTinv[0])
            self.left[self.block,1] = self.left[self.block,1].dot(self.BTinv[1])

        self.right[self.block,0] = B[0].dot(self.right[self.block,0])
        self.right[self.block,1] = B[1].dot(self.right[self.block,1])


        if self.diagonal_trial:
            self.stack[self.block,0] = numpy.einsum('ii,ij->ij',self.left[self.block,0],self.right[self.block,0])
            self.stack[self.block,1] = numpy.einsum('ii,ij->ij',self.left[self.block,1],self.right[self.block,1])
        else:
            self.stack[self.block,0] = self.left[self.block,0].dot(self.right[self.block,0])
            self.stack[self.block,1] = self.left[self.block,1].dot(self.right[self.block,1])

        self.time_slice = self.time_slice + 1 # Count the time slice
        self.block = self.time_slice // self.stack_size # move to the next block if necessary
        self.counter = (self.counter + 1) % self.stack_size # Counting within a stack

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

    print(Gold[:,0] - Gnew[:,0])
    # (Q, R, P) = scipy.linalg.qr(walker.stack.get(0)[0], pivoting = True)

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

    print(B - numpy.einsum('ij,jj->ij',Q1,D1).dot(T1))

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
    rdm = one_rdm_from_G(walker.G)
    ekin = local_energy_hubbard(system, rdm)[1]
    try:
        assert(abs(eref-ekin) < 1e-10)
    except AssertionError:
        print("Error in kinetic energy check. Ref: %f Calc:%f"%(eref, ekin))
    walker = ThermalWalker({"stack_size": 10}, system, trial)
    rdm = one_rdm_from_G(walker.G)
    ekin = local_energy_hubbard(system, rdm)[1]
    try:
        assert(abs(eref-ekin) < 1e-10)
    except AssertionError:
        print("Error in kinetic energy check. Ref: %f Calc:%f"%(eref, ekin))

if __name__=="__main__":
    unit_test()
