import h5py
import numpy
import os
from mpi4py import MPI
from pauxy.estimators.mixed import local_energy
from pauxy.estimators.thermal import one_rdm_from_G
from pauxy.estimators.utils import H5EstimatorHelper
from pauxy.utils.io import format_fixed_width_strings, format_fixed_width_floats

class Energy(object):

    def __init__(self, nstep, nblock, root, options={}, verbose=False):
        if verbose:
            print("# Setting up estimator object.")
        if root:
            index = options.get('index', 0)
            self.filename = options.get('filename', None)
            self.basename = options.get('basename', 'estimates')
            if self.filename is None:
                overwrite = options.get('overwrite', True)
                self.filename = self.basename + '.%s.h5' % index
                while os.path.isfile(self.filename) and not overwrite:
                    index = int(self.filename.split('.')[1])
                    index = index + 1
                    self.filename = self.basename + '.%s.h5' % index
            with h5py.File(self.filename, 'w') as fh5:
                pass
            if verbose:
                print("# Writing estimator data to {}.".format(self.filename))
        else:
            self.filename = None
        self.header = ['Iteration', 'Phase', 'ETotal',
                       'E1Body', 'E2Body', 'Nav']
        self.nstep = nstep
        self.nblock = nblock
        self.data_step = numpy.zeros(5, dtype=numpy.complex128)
        self.data_block = numpy.zeros(5, dtype=numpy.complex128)
        self.print_header()
        if root:
            self.setup_output(self.filename)

    def update_step(self, system, G, weight, phase):
        P = one_rdm_from_G(G)
        (etot, e1b, e2b) = local_energy(system, P)
        nav = P[0].trace() + P[1].trace()
        self.data_step[0] += weight*phase
        self.data_step[1] += weight*phase*etot
        # print(self.data_step[1])
        self.data_step[2] += weight*phase*e1b
        self.data_step[3] += weight*phase*e2b
        self.data_step[4] += weight*phase*nav

    def update_block(self):
        self.data_block += self.data_step / self.nstep
        # print(self.data_block)
        self.data_step[:] = 0.0

    def write(self, comm, iblock):
        data = comm.reduce(self.data_block, op=MPI.SUM) / (self.nblock*comm.size)
        if comm.rank == 0:
            print(format_fixed_width_floats([iblock]+list(data.real)))
            self.output.push([iblock]+list(data), 'energies')
            self.output.increment()
        self.data_block[:] = 0.0
        self.data_step[:] = 0.0

    def setup_output(self, filename):
        with h5py.File(filename, 'a') as fh5:
            fh5['basic/headers'] = numpy.array(self.header).astype('S')
        self.output = H5EstimatorHelper(filename, 'basic')

    def dump_metadata(self):
        with h5py.File(self.filename, 'a') as fh5:
            fh5['metadata'] = self.json_string

    def print_header(self, eol='', encode=False):
        r"""Print out header for estimators

        Parameters
        ----------
        eol : string, optional
            String to append to output, Default : ''.
        encode : bool
            In True encode output to be utf-8.

        Returns
        -------
        None
        """
        s = format_fixed_width_strings(self.header) + eol
        if encode:
            s = s.encode('utf-8')
        print(s)
