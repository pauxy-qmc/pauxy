import ast
import h5py
import json
import numpy
import scipy.sparse
from pauxy.utils.misc import serialise
from pauxy.utils.linalg import (
        molecular_orbitals_rhf, molecular_orbitals_uhf,
        modified_cholesky
)

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

def to_qmcpack_index(matrix, offset=0):
    try:
        indptr = matrix.indptr
        indices = matrix.indices
        data = matrix.data
    except AttributeError:
        matrix = scipy.sparse.csr_matrix(matrix)
        indptr = matrix.indptr
        indices = matrix.indices
        data = matrix.data
    # QMCPACK expects ([i,j], m_{ij}) pairs
    unpacked = []
    idx = []
    counter = 0
    for row in range(0, len(indptr)-1):
        idx += [[row, i+offset] for i in indices[indptr[row]:indptr[row+1]]]
        unpacked += [[v.real, v.imag] for v in data[indptr[row]:indptr[row+1]]]
        # print ("NZ: %d %d"%(row, len(indices[indptr[row]:indptr[row+1]])))
        if (len(data[indptr[row]:indptr[row+1]])) > 0:
            counter = counter + 1
            # print (row, len(data[indptr[row]:indptr[row+1]]))
    return (unpacked, numpy.array(idx).flatten())

def dump_qmcpack_cholesky(h1, h2, nelec, nmo, e0=0.0, filename='hamiltonian.h5'):
    dump = h5py.File(filename, 'w')
    dump['Hamiltonian/Energies'] = numpy.array([e0.real, e0.imag])
    (h1_unpacked, idx) = to_qmcpack_index(h1[0])
    dump['Hamiltonian/H1_indx'] = idx
    dump['Hamiltonian/H1'] = h1_unpacked
    # Number of non zero elements for two-body
    nnz = h2.nnz
    # number of cholesky vectors
    nchol_vecs = h2.shape[-1]
    dump['Hamiltonian/Factorized/block_sizes'] = numpy.array([nnz])
    (h2_unpacked, idx) = to_qmcpack_index(scipy.sparse.csr_matrix(h2))
    dump['Hamiltonian/Factorized/index_0'] = numpy.array(idx)
    dump['Hamiltonian/Factorized/vals_0'] = numpy.array(h2_unpacked)
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

def from_qmcpack_cholesky(filename):
    with h5py.File(filename, 'r') as fh5:
        enuc = fh5['Hamiltonian/Energies'][:].view(numpy.complex128).ravel()[0]
        idx = fh5['Hamiltonian/H1_indx'][:]
        row_ix = idx[::2]
        col_ix = idx[1::2]
        h1 = fh5['Hamiltonian/H1'][:].view(numpy.complex128).ravel()
        hcore = scipy.sparse.csr_matrix((h1, (row_ix, col_ix))).toarray()
        idx = fh5['Hamiltonian/Factorized/index_0'][:]
        h2 = fh5['Hamiltonian/Factorized/vals_0'][:].view(numpy.complex128).ravel()
        dims = fh5['Hamiltonian/dims'][:]
        nbasis = dims[3]
        nalpha = dims[4]
        nbeta = dims[5]
        row_ix = idx[::2]
        col_ix = idx[1::2]
        chol_vecs = scipy.sparse.csr_matrix((h2, (row_ix, col_ix)))
        return (hcore, chol_vecs, enuc, nbasis, nalpha, nbeta)

def dump_native(filename, hcore, eri, orthoAO, fock, nelec, enuc, verbose=True):
    if verbose:
        print (" # Constructing trial wavefunctiom in ortho AO basis.")
    if len(fock.shape) == 3:
        if verbose:
            print (" # Writing UHF trial wavefunction.")
        (mo_energies, mo_coeff) = molecular_orbitals_uhf(fock, orthoAO)
    else:
        if verbose:
            print (" # Writing RHF trial wavefunction.")
        (mo_energies, mo_coeff) = molecular_orbitals_rhf(fock, orthoAO)
    nbasis = hcore.shape[-1]
    mem = 64*nbasis**4/(1024.0*1024.0*1024.0)
    if verbose:
        print (" # Total number of elements in ERI tensor: %d"%nbasis**4)
        print (" # Total memory required for ERI tensor: %13.8e GB"%(mem))
    with h5py.File(filename, 'w') as fh5:
        fh5.create_dataset('hcore', data=hcore)
        fh5.create_dataset('nelec', data=nelec)
        fh5.create_dataset('eri', data=eri)
        fh5.create_dataset('enuc', data=[enuc])
        fh5.create_dataset('mo_coeff', data=mo_coeff)

def dump_qmcpack(filename, wfn_file, hcore, eri, orthoAO, fock, nelec, enuc,
                 verbose=True, threshold=1e-5, sparse_zero=1e-16):
    if verbose:
        print (" # Constructing trial wavefunctiom in ortho AO basis.")
    if len(fock.shape) == 3:
        if verbose:
            print (" # Writing UHF trial wavefunction.")
        (mo_energies, mo_coeff) = molecular_orbitals_uhf(fock, orthoAO)
    else:
        if verbose:
            print (" # Writing RHF trial wavefunction.")
        (mo_energies, mo_coeff) = molecular_orbitals_rhf(fock, orthoAO)
    dump_qmcpack_trial_wfn(mo_coeff, nelec, wfn_file)
    nbasis = hcore.shape[-1]
    if verbose:
        print (" # Performing modified Cholesky decomposition on ERI tensor.")
    msq = nbasis * nbasis
    chol_vecs = modified_cholesky(eri.reshape((msq, msq)), threshold,
                                  verbose=verbose).T
    chol_vecs[numpy.abs(chol_vecs) < sparse_zero] = 0
    chol_vecs = scipy.sparse.csr_matrix(chol_vecs)
    mem = 64*chol_vecs.nnz/(1024.0**3)
    if verbose:
        print (" # Total number of non-zero elements in sparse cholesky ERI"
               " tensor: %d"%chol_vecs.nnz)
        print (" # Total memory required for ERI tensor: %13.8e GB"%(mem))
    dump_qmcpack_cholesky(numpy.array([hcore, hcore]), chol_vecs, nelec,
                                      nbasis, enuc, filename=filename)

def dump_qmcpack_trial_wfn(wfn, nelec, filename='wfn.dat'):
    UHF = len(wfn.shape) == 3
    namelist = "&FCI\n UHF = %d\n FullMO \n NCI = 1\n TYPE = matrix\n/"%UHF
    # Single determinant for the moment.
    with open(filename, 'w') as f:
        f.write(namelist+'\n')
        f.write('Coefficients: 1.0\n')
        f.write('Determinant: 1\n')
        nao = wfn.shape[-1]
        if UHF:
            nao = wfn[0].shape[-1]
            write_qmcpack_wfn(f, wfn[0], nao)
            nao = wfn[1].shape[-1]
            write_qmcpack_wfn(f, wfn[1], nao)
        else:
            write_qmcpack_wfn(f, wfn, nao)

def write_qmcpack_wfn(out, mos, nao):
    for i in range(0, nao):
        for j in range(0, nao):
            val = mos[i,j]
            out.write('(%.10e,%.10e) '%(val.real, val.imag))
        out.write('\n')

def read_qmcpack_wfn(filename):
    with open(filename) as f:
        content = f.readlines()[8:]
    useable = numpy.array([c.split() for c in content]).flatten()
    tuples = [ast.literal_eval(u) for u in useable]
    orbs = [complex(t[0], t[1]) for t in tuples]
    return numpy.array(orbs)
