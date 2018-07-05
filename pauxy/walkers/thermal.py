import copy
import cmath
import numpy
import scipy.linalg
from pauxy.estimators.thermal import greens_function, one_rdm_from_G
from pauxy.estimators.mixed import local_energy

class ThermalWalker(object):

    def __init__(self, walker_opts, system, trial, verbose=False):
        self.weight = walker_opts.get('weight', 1)
        self.alive = True
        self.num_slices = trial.ntime_slices
        if verbose:
            print("# Number of slices = {}".format(self.num_slices))
        if system.name == "UEG":
            dtype = numpy.complex128
        else:
            dtype = numpy.float
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
                                     trial.dmat.shape[-1], dtype)

        # Initialise all propagators to the trial density matrix.
        self.stack.set_all(trial.dmat)
        self.greens_function(trial)
        self.ot = 1.0

        cond = numpy.linalg.cond(trial.dmat[0])
        if verbose:
            print("# condition number of BT = {}".format(cond))

    def greens_function(self, trial, slice_ix = None):
        # self.greens_function_svd(trial, slice_ix)
        self.greens_function_qr(trial, slice_ix)

    def greens_function_svd(self, trial, slice_ix = None):
        if (slice_ix == None):
            slice_ix = self.stack.time_slice
        bin_ix = slice_ix // self.stack.stack_width
        # For final time slice want first block to be the rightmost (for energy
        # evaluation).
        if bin_ix == self.stack.nbins:
            bin_ix = -1
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
            # Final SVD decomposition to construct G(l) = [I + A(l)]^{-1}.
            # Care needs to be taken when adding the identity matrix.
            T3 = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
            (U2, S2, V2) = scipy.linalg.svd(T3)
            U3 = numpy.dot(U1, U2)
            D3 = numpy.diag(1.0/S2)
            V3 = numpy.dot(V2, V1)
            # G(l) = (U3 S2 V3)^{-1}
            #      = V3^{\dagger} D3 U3^{\dagger}
            self.G[spin] = (V3.conj().T).dot(D3).dot(U3.conj().T)

    def greens_function_qr(self, trial, slice_ix = None):
        if (slice_ix == None):
            slice_ix = self.stack.time_slice

        bin_ix = slice_ix // self.stack.stack_width
        # For final time slice want first block to be the rightmost (for energy
        # evaluation).
        if bin_ix == self.stack.nbins:
            bin_ix = -1
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
            self.G[spin] = (V3inv).dot(U3.conj().T)

    # def construct_greens_function_unstable(self, slice_ix):
    #     I = numpy.identity(self.G.shape[-1], dtype=self.G.dtype)
    #     A = numpy.array([I,I])
    #     for i in range(0, self.stack.nbins):
    #         ix = (slice_ix + i) % self.stack.nbins
    #         B = self.stack.get(ix)
    #         A[0] = B[0].dot(A[0])
    #         A[1] = B[1].dot(A[1])
    #     self.G = greens_function(A)

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
            'stack': self.stack.stack,
            'G': self.G,
            'weight': self.weight,
        }
        return buff

    def set_buffer(self, buff):
        """Set walker buffer following MPI communication

        Parameters
        -------
        buff : dict
            Relevant walker information for population control.
        """
        self.stack.stack = numpy.copy(buff['stack'])
        self.G = numpy.copy(buff['G'])
        self.weight = buff['weight']


class PropagatorStack:
    def __init__(self, bin_size, ntime_slices, nbasis, dtype, BT=None, sparse=True):
        self.time_slice = 0
        self.stack_width = bin_size
        self.ntime_slices = ntime_slices
        self.nbins = ntime_slices // bin_size
        self.nbasis = nbasis
        self.dtype = dtype
        self.counter = 0
        self.block = 0
        self.stack = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis),
                                 dtype=dtype)
        # set all entries to be the identity matrix
        self.reset()

    def get(self, ix):
        return self.stack[ix]

    def set_all(self, BT):
        for i in range(0, self.ntime_slices):
            ix = i // self.stack_width
            self.stack[ix,0] = BT[0].dot(self.stack[ix,0])
            self.stack[ix,1] = BT[1].dot(self.stack[ix,1])

    def reset(self):
        self.time_slice = 0
        self.block = 0
        for i in range(0, self.nbins):
            self.stack[i,0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.stack[i,1] = numpy.identity(self.nbasis, dtype=self.dtype)

    def update(self, B):
        if self.counter == 0:
            self.stack[self.block,0] = numpy.identity(B.shape[-1], dtype=B.dtype)
            self.stack[self.block,1] = numpy.identity(B.shape[-1], dtype=B.dtype)
        self.stack[self.block,0] = B[0].dot(self.stack[self.block,0])
        self.stack[self.block,1] = B[1].dot(self.stack[self.block,1])
        self.time_slice = self.time_slice + 1
        self.block = self.time_slice // self.stack_width
        self.counter = (self.counter + 1) % self.stack_width

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
    walker = ThermalWalker(1.0, system, trial, stack_size=None)
    walker.greens_function(trial)
    E, T, V = walker.local_energy(system)
    print(E,T,V)
    # (Q, R, P) = scipy.linalg.qr(walker.stack.get(0)[0], pivoting = True)
    # N = 100

    # A = numpy.random.rand(N,N)
    # Q, R, P = scipy.linalg.qr(A, pivoting = True)
    # Pmat = numpy.zeros((N,N))
    # for i in range (N):
    #     Pmat[P[i],i] = 1

    # print(A - Q.dot(R).dot(Pmat.T))
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
