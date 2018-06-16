import copy
import numpy
import scipy.linalg
from pauxy.estimators.thermal import greens_function

class ThermalWalker(object):

    def __init__(self, weight, system, trial):
        self.weight = weight
        self.num_slices = trial.ntime_slices
        self.G = numpy.zeros(trial.dmat.shape, dtype=trial.dmat.dtype)
        self.stack_length = self.num_slices // self.bin_size
        # todo: Fix this hardcoded value
        self.stack = Stack(10, trial.ntime_slices, trial.dmat.shape[-1],
                           trial.dmat.dtype)

    def construct_greens_function_stable(self, stack, slice_ix):
        for spin in range(0, 2):
            (U1, S1, V1) = scipy.linalg.svd(stack[slice_ix,spin])
            for i in range(1, stack):
                ix = (slice_ix + i) % self.stack_length
                T1 = numpy.dot(self.stack[ix,spin], U1)
                T2 = numpy.dot(T1, numpy.diag(S1))
                (U1, S1, V) = scipy.linalg.svd(T2)
                V1 = numpy.dot(V, V1)
            T3 = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
            (U2, S2, V2) = scipy.linalg.svd(T3)
            U3 = numpy.dot((U1.conj().T), U2.conj().T)
            D3 = numpy.diag(1.0/S2)
            V3 = numpy.dot(V2.conj().T, V1.conj().T)
            self.G[spin] = U3.dot(D3).dot(V3)

    def construct_greens_function_unstable(self, slice_ix):
        I = numpy.identity(self.G.shape[-1], dtype=self.G.dtype)
        A = numpy.array([I,I])
        for i in range(0, self.stack_length):
            ix = (slice_ix + i) % self.stack_length
            A[0] = self.stack[ix,0].dot(A[0])
            A[1] = self.stack[ix,1].dot(A[1])
        self.G = greens_function(A)

    def recompute_greens_function(self, stack, trial, time_slice):
        # Super stupid
        BTAlpha = numpy.linalg.matrix_power(trial.dmat[0],
                                            trial.ntime_slices-time_slice)
        BTBeta = numpy.linalg.matrix_power(trial.dmat[1],
                                           trial.ntime_slices-time_slice)
        BT = numpy.array([BTAlpha, BTBeta])


class Stack:
    def __init__(self, bin_size, ntime_slices, nbasis, dtype, BT=None):
        self.time_slice = 0
        self.stack_width = bin_size
        self.ntime_slices = ntime_slices
        self.nbasis = nbasis
        self.dtype = dtype
        self.stack = numpy.zeros(shape=(ntime_slices, 2, nbasis, nbasis),
                                 dtype=dtype)
        self.reset_stack()

    def reset_stack(self):
        self.time_slice = 0
        for i in range(0, self.ntime_slices):
            self.stack[i,0] = numpy.identity(nbasis, dtype=dtype)
            self.stack[i,1] = numpy.identity(nbasis, dtype=dtype)

    def update_stack(self, B):
        self.stack[self.time_slice,0] = B[0].dot(self.stack[self.time_slice,0])
        self.stack[self.time_slice,1] = B[1].dot(self.stack[self.time_slice,1])
        self.time_slice = (self.time_slice + 1) // self.stack_width
