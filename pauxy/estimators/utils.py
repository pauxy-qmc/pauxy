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
