import copy
import numpy
import scipy.linalg
from pauxy.estimators.thermal import greens_function

class ThermalWalker(object):

    def __init__(self, weight, system, trial):
        self.weight = weight
        self.num_slices = trial.ntime_slices
        self.G = numpy.zeros(trial.dmat.shape, dtype=trial.dmat.dtype)
        self.stack_length = self.num_slices // 10
        # todo: Fix this hardcoded value
        self.stack = PropagatorStack(10, trial.ntime_slices,
                                     trial.dmat.shape[-1], trial.dmat.dtype)
        # Initialise all propagators to the trial density matrix.
        self.stack.set_all(trial.dmat)

    def construct_greens_function_stable(self, slice_ix):
        bin_ix = slice_ix // self.stack.bin_size
        for spin in [0, 1]:
            # Need to construct the product A(l) = B_l B_{l-1}..B_L...B_{l+1}
            # in stable way. Iteratively construct SVD decompositions starting
            # from the rightmost (product of) propagator(s).
            B = self.stack.get((bin_ix+1)%self.stack.nbins)
            (U1, S1, V1) = scipy.linalg.svd(B[spin])
            for i in range(1, self.stack.nbins):
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
            U3 = numpy.dot((U1.conj().T), U2.conj().T)
            D3 = numpy.diag(1.0/S2)
            V3 = numpy.dot(V2.conj().T, V1.conj().T)
            # G(l) = (U3 S2 V3)^{-1}
            #      = V3^{\dagger} D3 U3^{\dagger}
            self.G[spin] = (V3.conj().T).dot(D3).dot(U3.conj().T)

    def construct_greens_function_unstable(self, slice_ix):
        I = numpy.identity(self.G.shape[-1], dtype=self.G.dtype)
        A = numpy.array([I,I])
        for i in range(0, self.stack.nbins):
            ix = (slice_ix + i) % self.stack.nbins
            B = self.stack.get(ix)
            A[0] = B[0].dot(A[0])
            A[1] = B[1].dot(A[1])
        self.G = greens_function(A)


class PropagatorStack:
    def __init__(self, bin_size, ntime_slices, nbasis, dtype, BT=None):
        self.time_slice = 0
        self.stack_width = bin_size
        self.ntime_slices = ntime_slices
        self.nbins = ntime_slices // bin_size
        self.nbasis = nbasis
        self.dtype = dtype
        self.counter = 0
        self.stack = numpy.zeros(shape=(ntime_slices//bin_size, 2, nbasis, nbasis),
                                 dtype=dtype)
        self.reset_stack()

    def get(self, ix):
        return self.stack[ix]

    def set_all(self, BT):
        for i in range(0, self.ntime_slices):
            ix = i // self.stack_width
            self.stack[ix,0] = BT[0].dot(self.stack[ix,0])
            self.stack[ix,1] = BT[0].dot(self.stack[ix,1])

    def reset_stack(self):
        self.time_slice = 0
        for i in range(0, self.nbins):
            self.stack[i,0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.stack[i,1] = numpy.identity(self.nbasis, dtype=self.dtype)

    def update(self, B):
        if self.counter == 0:
            self.stack[self.time_slice,0] = numpy.identity(B.shape[-1])
            self.stack[self.time_slice,1] = numpy.identity(B.shape[-1])
        self.stack[self.time_slice,0] = B[0].dot(self.stack[self.time_slice,0])
        self.stack[self.time_slice,1] = B[1].dot(self.stack[self.time_slice,1])
        self.time_slice = (self.time_slice + 1) // self.stack_width
        self.counter = (self.counter + 1) % self.bin_width
