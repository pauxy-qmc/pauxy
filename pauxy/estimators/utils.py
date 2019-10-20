import pyfftw
import numpy
from scipy.fftpack.helper import next_fast_len

def convolve(f, g, mesh, backend = numpy.fft):
# def convolve(f, g, mesh, backend = pyfftw.interfaces.numpy_fft):
    f_ = f.reshape(*mesh)
    g_ = g.reshape(*mesh)
    shape = numpy.maximum(f_.shape, g_.shape)
    min_shape = numpy.array(f_.shape) + numpy.array(g_.shape) - 1

    nqtot = numpy.prod(min_shape)
    fshape = [next_fast_len(d) for d in min_shape]

    finv = backend.ifftn(f_, s=fshape)
    ginv = backend.ifftn(g_, s=fshape)
    fginv = finv * ginv
    fq = backend.fftn(fginv).copy().ravel()
    fq = fq.reshape(fshape)
    fq = fq[:min_shape[0],:min_shape[1],:min_shape[2]]
    fq = fq.reshape(nqtot) * numpy.prod(fshape)
    return fq

class H5EstimatorHelper(object):
    """Helper class for pushing data to hdf5 dataset of fixed length.

    Parameters
    ----------
    h5f : :class:`h5py.File`
        Output file object.
    name : string
        Dataset name.
    shape : tuple
        Shape of output data.
    dtype : type
        Output data type.

    Attributes
    ----------
    store : :class:`h5py.File.DataSet`
        Dataset object.
    index : int
        Counter for incrementing data.
    """
    def __init__(self, h5f, name, shape, dtype):
        self.store = h5f.create_dataset(name, shape, dtype=dtype)
        self.index = 0

    def push(self, data):
        """Push data to dataset.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            Data to push.
        """
        self.store[self.index] = data
        self.index = self.index + 1

    def reset(self):
        self.index = 0
