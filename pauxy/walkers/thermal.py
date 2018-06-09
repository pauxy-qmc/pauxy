import copy
import numpy
import scipy.linalg

class ThermalWalker(object):

    def __init__(self, weight, system, trial, num_slices, stack_size=10):
        self.weight = weight
        self.num_slices = num_slices
        self.stack_size = stack_size
        self.G = numpy.zeros(trial.shape, dtype=trial.dtype)
        stack_length = num_slices // stack_size
        self.stack = numpy.zeros(shape=(stack_length,)+trial.propg.shape,
                                 dtype=trial.propg.dtype)

    def create_stack(self, trial):
        for i in range(0, self.stack.shape[0]):
            self.stack[i] = numpy.identity(self.propg.shape,
                                           dtype=self.propg.dtype)
        for i in range(0, self.num_slices):
            ix = i // self.stack_size
            self.stack[ix] = trial.dot(self.stack_size[ix])

    def construct_greens_function_stable(self, slice_ix):
        # A = numpy.identity(self.G.shape, dtype=self.G.dtype)
        (U1, S1, V1) = scipy.linalg.svd(self.stack[(slice_ix+1)%self.stack_size])
        for i in range(1, self.stack_size):
            ix = (slice_ix + i) % self.stack_size
            T1 = numpy.dot(self.stack[ix], U1)
            T2 = numpy.dot(T1, numpy.diag(S1))
            (U1, S1, V) = scipy.linalg.svd(T2)
            V1 = numpy.dot(V, V1)
        T3 = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
        (U2, S2, V2) = scipy.linalg.svd(T3)
        U3 = numpy.dot((U1.conj().T), U2.conj().T)
        D3 = numpy.diag(1.0/S2)
        V3 = numpy.dot(V2.conj().T, V1.conj().T)
        G = U3.dot(D3).dot(V3)

    def construct_greens_function_unstable(self, slice_ix):
        A = numpy.copy(self.stack[slice_ix])
        for i in range(1, self.stack_size):
            ix = (slice_ix + i) % self.stack_size
            A = self.stack[ix].dot(A)
        G = scipy.linalg.inv(numpy.identity(A.shape, dtype=A.dtype) + A)
