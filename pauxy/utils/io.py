import ast
import h5py
import json
import numpy
import scipy.sparse
from pauxy.utils.misc import serialise

def format_fixed_width_strings(strings):
    return ' '.join('{:>17}'.format(s) for s in strings)


def format_fixed_width_floats(floats):
    return ' '.join('{: .10e}'.format(f) for f in floats)


def read_fortran_complex_numbers(filename):
    with open(filename) as f:
        content = f.readlines()
    # Converting fortran complex numbers to python. ugh
    # Be verbose for clarity.
    useable = [c.strip() for c in content]
    tuples = [ast.literal_eval(u) for u in useable]
    orbs = [complex(t[0], t[1]) for t in tuples]
    return numpy.array(orbs)


def fcidump_header(nel, norb, spin):
    header = (
        "&FCI\n" +
        "NORB=%d,\n"%int(norb) +
        "NELEC=%d,\n"%int(nel) +
        "MS2=%d,\n"%int(spin) +
        "UHF=.FALSE.,\n" +
        "ORBSYM=" + ",".join([str(1)]*norb) + ",\n"
        "&END\n"
    )
    return header

def to_json(afqmc):
    json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')
    json_string = json.dumps(serialise(afqmc, verbose=1),
                             sort_keys=False, indent=4)
    return json_string

def to_qmcpack_index(matrix):
    try:
        indptr = matrix.indptr
        indices = matrix.indices
        data = matrix.data
    except AttributeError:
        print ("Matrix provided was not in CSC format. Converting.")
        matrix = scipy.sparse.csc_matrix(matrix)
        indptr = matrix.indptr
        indices = matrix.indices
        data = matrix.data
    # QMCPACK expects ([i,j], m_{ij}) pairs
    unpacked = []
    idx = []
    for col in range(0, len(indptr)-1):
        idx += [[i, col] for i in indices[indptr[col]:indptr[col+1]]]
        unpacked += [[v.real, v.imag] for v in data[indptr[col]:indptr[col+1]]]
    return (unpacked, numpy.array(idx).flatten())

def dump_qmcpack_cholesky(h1, h2, nelec, nmo, e0=0.0, name='hamiltonian.h5'):
    dump = h5py.File(name, 'w')
    dump['Hamiltonian/Energies'] = numpy.array([e0.real, e0.imag])
    (h1_unpacked, idx) = to_qmcpack_index(h1[0])
    dump['Hamiltonian/H1_indx'] = idx
    dump['Hamiltonian/H1'] = h1_unpacked
    if len(h2) == 2:
        # Number of non zero elements for two-body
        nnz = h2[0].nnz + h2[1].nnz
        # number of cholesky vectors
        nchol_vecs = h2[0].shape[-1] + h2[1].shape[-1]
        dump['Hamiltonian/Factorized/block_sizes'] = numpy.array([nnz])
        (h2_unpacked_a, idx_a) = to_qmcpack_index(h2[0])
        (h2_unpacked_b, idx_b) = to_qmcpack_index(h2[1])
        dump['Hamiltonian/Factorized/index_0'] = numpy.array(idx_a+idx_b)
        dump['Hamiltonian/Factorized/vals_0'] = numpy.array(h2_unpacked_a+h2_unpacked_b)
    # Number of integral blocks used for chunked HDF5 storage.
    # Currently hardcoded for simplicity.
    nint_block = 1
    (nalpha, nbeta) = nelec
    # unused parameter as far as I can tell.
    unused = 0
    dump['Hamiltonian/dims'] = numpy.array([len(h1_unpacked), nnz, nint_block, nmo,
                                            nalpha, nbeta, unused, nchol_vecs])
    occups = [i for i in range(0, nalpha)]
    occups += [i+nmo for i in range(0, nbeta)]
    dump['Hamiltonian/occups'] = numpy.array(occups)
