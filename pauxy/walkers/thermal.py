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
            dtype = numpy.complex64
        self.G = numpy.zeros(trial.dmat.shape, dtype=dtype)
        self.stack_size = walker_opts.get('stack_size', None)

        if (self.stack_size == None):
            if verbose:
                print ("# Stack size is determined by BT")
            emax = numpy.max(numpy.diag(trial.dmat[0]))
            emin = numpy.min(numpy.diag(trial.dmat[0]))
            self.stack_size = min(self.num_slices,
                int (1.5 / ((cmath.log(float(emax)) - cmath.log(float(emin))) / 8.0).real))
            if verbose:
                print ("# Initial stack size is {}".format(self.stack_size))

        # adjust stack size
        lower_bound = min(self.stack_size, self.num_slices)
        upper_bound = min(self.stack_size, self.num_slices)

        while ((self.num_slices//lower_bound) * lower_bound < self.num_slices):
            lower_bound -= 1
        while ((self.num_slices//upper_bound) * upper_bound < self.num_slices):
            upper_bound += 1

        if ((self.stack_size-lower_bound) <= (upper_bound - self.stack_size)):
            self.stack_size = lower_bound
        else:
            self.stack_size = upper_bound

        if verbose:
            print ("# upper_bound is {}".format(upper_bound))
            print ("# lower_bound is {}".format(lower_bound))
            print ("# Adjusted stack size is {}".format(self.stack_size))

        self.stack_length = self.num_slices // self.stack_size

        self.stack = PropagatorStack(self.stack_size, trial.ntime_slices,
                                     trial.dmat.shape[-1], dtype, BT=trial.dmat,BTinv=trial.dmat_inv)

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
        # self.greens_function_svd(trial, slice_ix)
        return self.greens_function_qr_strat(trial, slice_ix=slice_ix,
                                             inplace=inplace)

    def greens_function_svd(self, trial, slice_ix=None, inplace=True):
        if (slice_ix == None):
            slice_ix = self.stack.time_slice
        bin_ix = slice_ix // self.stack.stack_size
        # For final time slice want first block to be the rightmost (for energy
        # evaluation).
        if bin_ix == self.stack.nbins:
            bin_ix = -1
        if not inplace:
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
        return (G, A)


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
            (U1, V1) = numpy.linalg.qr(B[spin])

            for i in range(2, self.stack.nbins+1):
                ix = (bin_ix + i) % self.stack.nbins
                B = self.stack.get(ix)
                T1 = numpy.dot(B[spin], U1)
                (U1, V) = scipy.linalg.qr(T1, pivoting = False)
                V1 = numpy.dot(V, V1)

            # Final SVD decomposition to construct G(l) = [I + A(l)]^{-1}.
            # Care needs to be taken when adding the identity matrix.
            V1inv = scipy.linalg.solve_triangular(V1, numpy.identity(V1.shape[0]))

            T3 = numpy.dot(U1.conj().T, V1inv) + numpy.identity(V1.shape[0])
            (U2, V2) = scipy.linalg.qr(T3, pivoting = False)

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
                B = self.stack.get(0)
                (self.Qr[spin], R1, P1) = scipy.linalg.qr(B[spin], pivoting=True)
                # Form permutation matrix
                P1mat = numpy.zeros(B[spin].shape, B[spin].dtype)
                P1mat[P1,range(len(P1))] = 1.0
                # Form D matrices
                self.Dr[spin] = numpy.diag(R1.diagonal())
                D1inv = numpy.diag(1.0/R1.diagonal())
                self.Tr[spin] = numpy.dot(numpy.dot(D1inv, R1), P1mat.T)

                for i in range(2, center_ix):
                    ix = (i-1) % self.stack.nbins
                    B = self.stack.get(ix)
                    C2 = numpy.dot(numpy.dot(B[spin], self.Qr[spin]), self.Dr[spin])
                    (self.Qr[spin], R1, P1) = scipy.linalg.qr(C2, pivoting=True)
                    # Form permutation matrix
                    P1mat = numpy.zeros(B[spin].shape, B[spin].dtype)
                    P1mat[P1,range(len(P1))] = 1.0
                    # Compute D matrices
                    D1inv = numpy.diag(1.0/R1.diagonal())
                    self.Dr[spin] = numpy.diag(R1.diagonal())
                    self.Tr[spin] = numpy.dot(numpy.dot(D1inv, R1), numpy.dot(P1mat.T, self.Tr[spin]))

            # left bit
            # B(l) ... B(left)
            if (center_ix < self.stack.ntime_slices-1 ):
                B = self.stack.get(center_ix+1)
                (self.Ql[spin], R1, P1) = scipy.linalg.qr(B[spin], pivoting=True)
                # Form permutation matrix
                P1mat = numpy.zeros(B[spin].shape, B[spin].dtype)
                P1mat[P1,range(len(P1))] = 1.0
                # Form D matrices
                self.Dl[spin] = numpy.diag(R1.diagonal())
                D1inv = numpy.diag(1.0/R1.diagonal())
                self.Tl[spin] = numpy.dot(numpy.dot(D1inv, R1), P1mat.T)

                for i in range(center_ix+1, self.stack.ntime_slices+1):
                    ix = (i-1) % self.stack.nbins
                    B = self.stack.get(ix)
                    C2 = numpy.dot(numpy.dot(B[spin], self.Ql[spin]), self.Dl[spin])
                    (self.Ql[spin], R1, P1) = scipy.linalg.qr(C2, pivoting=True)
                    # Form permutation matrix
                    P1mat = numpy.zeros(B[spin].shape, B[spin].dtype)
                    P1mat[P1,range(len(P1))] = 1.0
                    # Compute D matrices
                    D1inv = numpy.diag(1.0/R1.diagonal())
                    self.Dl[spin] = numpy.diag(R1.diagonal())
                    self.Tl[spin] = numpy.dot(numpy.dot(D1inv, R1), numpy.dot(P1mat.T, self.Tl[spin]))
    
    def greens_function_left_right (self, center_ix, inplace = False):
        if not inplace:
            G = numpy.zeros(self.G.shape, self.G.dtype)
        else:
            G = None
        
        Bc = self.stack.get(center_ix+1)

        for spin in [0,1]:
            if (center_ix > 0): # there exists right bit
                Ccr = numpy.dot(numpy.dot(Bc[spin],self.Qr[spin]),self.Dr[spin])
                (Qlcr, Rlcr, Plcr) = scipy.linalg.qr(Ccr, pivoting=True)
                Dlcr = numpy.diag(Rlcr.diagonal())
                Dinv = numpy.diag(1.0/Rlcr.diagonal())
                P1mat = numpy.zeros(Bc[spin].shape, Bc[spin].dtype)
                P1mat[Plcr,range(len(Plcr))] = 1.0
                Tlcr = numpy.dot(numpy.dot(Dinv, Rlcr), numpy.dot(P1mat.T, self.Tr[spin]))
            else:
                (Qlcr, Rlcr, Plcr) = scipy.linalg.qr(Bc[spin], pivoting=True)
                # Form permutation matrix
                P1mat = numpy.zeros(Bc[spin].shape, Bc[spin].dtype)
                P1mat[Plcr,range(len(Plcr))] = 1.0
                # Form D matrices
                Dlcr = numpy.diag(Rlcr.diagonal())
                Dinv = numpy.diag(1.0/Rlcr.diagonal())
                Tlcr = numpy.dot(numpy.dot(D1inv, Rlcr), P1mat.T)

            if (center_ix < self.stack.ntime_slices-1 ): # there exists left bit
                Clcr = numpy.dot(numpy.dot(self.Ql[spin], numpy.dot(self.Dl[spin], self.Tl[spin])), numpy.dot(Qlcr, Dlcr))
                (Qlcr, Rlcr, Plcr) = scipy.linalg.qr(Clcr, pivoting=True)
                Dlcr = numpy.diag(Rlcr.diagonal())
                Dinv = numpy.diag(1.0/Rlcr.diagonal())
                P1mat = numpy.zeros(Bc[spin].shape, Bc[spin].dtype)
                P1mat[Plcr,range(len(Plcr))] = 1.0
                Tlcr = numpy.dot(numpy.dot(Dinv, Rlcr), numpy.dot(P1mat.T, self.Tr[spin]))


            # G^{-1} = 1+A = 1+QDT = Q (Q^{-1}T^{-1}+D) T
            # Write D = Db^{-1} Ds
            # Then G^{-1} = Q Db^{-1}(Db Q^{-1}T^{-1}+Ds) T
            Db = numpy.zeros(Bc[spin].shape, Bc[spin].dtype)
            Ds = numpy.zeros(Bc[spin].shape, Bc[spin].dtype)
            for i in range(Db.shape[0]):
                if abs(Dlcr[i,i]) > 1.0:
                    Db[i,i] = 1.0 / abs(Dlcr[i,i])
                    Ds[i,i] = numpy.sign(Dlcr[i,i])
                else:
                    Db[i,i] = 1.0
                    Ds[i,i] = Dlcr[i,i]

            T1inv = numpy.linalg.pinv(Tlcr)
            # C = (Db Q^{-1}T^{-1}+Ds)
            C = numpy.dot(numpy.dot(Db, Qlcr.conj().T), T1inv) + Ds
            Cinv = numpy.linalg.pinv(C)

            # Then G = T^{-1} C^{-1} Db Q^{-1}
            # Q is unitary.
            if inplace:
                self.G[spin] = numpy.dot(numpy.dot(T1inv, Cinv),
                                         numpy.dot(Db, Qlcr.conj().T))
            else:
                G[spin] = numpy.dot(numpy.dot(T1inv, Cinv),
                                    numpy.dot(Db, Qlcr.conj().T))


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

            (Q1, R1, P1) = scipy.linalg.qr(B[spin], pivoting=True)
            # Form permutation matrix
            P1mat = numpy.zeros(B[spin].shape, B[spin].dtype)
            P1mat[P1,range(len(P1))] = 1.0
            # Form D matrices
            D1 = numpy.diag(R1.diagonal())
            D1inv = numpy.diag(1.0/R1.diagonal())
            T1 = numpy.dot(numpy.dot(D1inv, R1), P1mat.T)

            for i in range(2, self.stack.nbins+1):
                ix = (bin_ix + i) % self.stack.nbins
                B = self.stack.get(ix)
                C2 = numpy.dot(numpy.dot(B[spin], Q1), D1)
                (Q1, R1, P1) = scipy.linalg.qr(C2, pivoting=True)
                # Form permutation matrix
                P1mat = numpy.zeros(B[spin].shape, B[spin].dtype)
                P1mat[P1,range(len(P1))] = 1.0
                # Compute D matrices
                D1inv = numpy.diag(1.0/R1.diagonal())
                D1 = numpy.diag(R1.diagonal())
                T1 = numpy.dot(numpy.dot(D1inv, R1), numpy.dot(P1mat.T, T1))

            # G^{-1} = 1+A = 1+QDT = Q (Q^{-1}T^{-1}+D) T
            # Write D = Db^{-1} Ds
            # Then G^{-1} = Q Db^{-1}(Db Q^{-1}T^{-1}+Ds) T
            Db = numpy.zeros(B[spin].shape, B[spin].dtype)
            Ds = numpy.zeros(B[spin].shape, B[spin].dtype)
            for i in range(Db.shape[0]):
                if abs(D1[i,i]) > 1.0:
                    Db[i,i] = 1.0 / abs(D1[i,i])
                    Ds[i,i] = numpy.sign(D1[i,i])
                else:
                    Db[i,i] = 1.0
                    Ds[i,i] = D1[i,i]

            T1inv = numpy.linalg.pinv(T1)
            # C = (Db Q^{-1}T^{-1}+Ds)
            C = numpy.dot(numpy.dot(Db, Q1.conj().T), T1inv) + Ds
            Cinv = numpy.linalg.pinv(C)

            # Then G = T^{-1} C^{-1} Db Q^{-1}
            # Q is unitary.
            if inplace:
                self.G[spin] = numpy.dot(numpy.dot(T1inv, Cinv),
                                         numpy.dot(Db, Q1.conj().T))
            else:
                G[spin] = numpy.dot(numpy.dot(T1inv, Cinv),
                                    numpy.dot(Db, Q1.conj().T))
        return G

    def local_energy(self, system):
        rdm = one_rdm_from_G(self.G)
        return local_energy(system, rdm)

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


class PropagatorStack:
    def __init__(self, bin_size, ntime_slices, nbasis, dtype, BT, BTinv, sparse=True):
        self.time_slice = 0
        self.stack_size = bin_size
        self.ntime_slices = ntime_slices
        self.nbins = ntime_slices // bin_size

        if (self.nbins * self.stack_size < self.ntime_slices):
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
        for i in range(0, self.ntime_slices):
            ix = i // self.stack_size # bin index
            self.stack[ix,0] = BT[0].dot(self.stack[ix,0])
            self.stack[ix,1] = BT[1].dot(self.stack[ix,1])
            self.left[ix,0] = BT[0].dot(self.left[ix,0])
            self.left[ix,1] = BT[1].dot(self.left[ix,1])

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
        if self.counter == 0:
            self.right[self.block,0] = numpy.identity(B.shape[-1], dtype=B.dtype)
            self.right[self.block,1] = numpy.identity(B.shape[-1], dtype=B.dtype)

        self.left[self.block,0] = self.left[self.block,0].dot(self.BTinv[0])
        self.left[self.block,1] = self.left[self.block,1].dot(self.BTinv[1])

        self.right[self.block,0] = B[0].dot(self.right[self.block,0])
        self.right[self.block,1] = B[1].dot(self.right[self.block,1])

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
    "beta":2.0,
    "dt": 0.05
    }
    beta = inputs ['beta']
    dt = inputs['dt']

    system = UEG(inputs, True)

    qmc = QMCOpts(inputs, system, True)
    trial = OneBody(inputs, system, beta, dt, system.H1, verbose=True)

    propagator = PlaneWave(inputs, qmc, system, trial, True)
    # def __init__(self, walker_opts, system, trial, verbose=False):
    walker = ThermalWalker(inputs, system, trial, True)
    walker.greens_function(trial)
    E, T, V = walker.local_energy(system)
    print(E,T,V)
    (Q, R, P) = scipy.linalg.qr(walker.stack.get(0)[0], pivoting = True)
    N = 5

    A = numpy.random.rand(N,N)
    Q, R, P = scipy.linalg.qr(A, pivoting=True)
    Pmat = numpy.zeros((N,N))
    for i in range (N):
        Pmat[P[i],i] = 1
    print(A - Q.dot(R).dot(Pmat.T))
    print(Q * Q.T)
    print(R)

    # print (P)
    # R = numpy.random.rand(system.nbasis, system.nbasis)
    # R = numpy.triu(R)
    # R = scipy.sparse.csc_matrix(R)
    # print(R*R)
    # system.scaled_density_operator_incore(system.qvecs)

    # for (i, qi) in enumerate(system.qvecs):
        # rho_q = system.density_operator(i)
        # A = rho_q + scipy.sparse.csc_matrix.transpose(rho_q)
        # print(A.dot(rho_q).diagonal().sum())
        # exit()
        # print(scipy.sparse.csc_matrix.transpose(rho_q).shape)
        # rho_mq = system.density_operator(-i)
        # print (numpy.linalg.norm(rho_q-rho_mq.T))
if __name__=="__main__":
    unit_test()
