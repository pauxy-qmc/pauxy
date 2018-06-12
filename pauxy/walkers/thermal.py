import copy
import numpy
import scipy.linalg
from pauxy.estimators.thermal import greens_function

class ThermalWalker(object):

    def __init__(self, weight, system, trial, num_slices, bin_size=10):
        self.weight = weight
        self.num_slices = num_slices
        self.bin_size = bin_size
        self.G = numpy.zeros(trial.dmat.shape, dtype=trial.dmat.dtype)
        self.stack_length = num_slices // self.bin_size
        self.stack = numpy.zeros(shape=(self.stack_length,)+trial.dmat.shape,
                                 dtype=trial.dmat.dtype)
        self.create_stack(trial)

    def create_stack(self, trial):
        for i in range(0, self.stack.shape[0]):
            self.stack[i,0] = numpy.identity(trial.dmat[0].shape[0],
                                             dtype=trial.dmat.dtype)
            self.stack[i,1] = numpy.identity(trial.dmat[1].shape[1],
                                             dtype=trial.dmat.dtype)
        for i in range(0, self.num_slices):
            ix = i // self.bin_size
            self.stack[ix,0] = trial.dmat[0].dot(self.stack[ix,0])
            self.stack[ix,1] = trial.dmat[1].dot(self.stack[ix,1])

    def construct_greens_function_stable(self, slice_ix):
        for spin in range(0, 1):
            (U1, S1, V1) = scipy.linalg.svd(self.stack[slice_ix,spin])
            for i in range(1, self.stack_length):
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
