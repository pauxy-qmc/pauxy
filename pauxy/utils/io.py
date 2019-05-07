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
    json_string = json.dumps(serialise(afqmc, verbose=afqmc.verbosity),
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
    hcore = h1[0].astype(numpy.complex128).view(numpy.float64)
    hcore = hcore.reshape(h1[0].shape+(2,))
    dump['Hamiltonian/hcore'] = hcore
    # dump['Hamiltonian/hcore'].dims = numpy.array([h1[0].shape[0], h1[0].shape[1]])
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
    dump['Hamiltonian/dims'] = numpy.array([unused, nnz, nint_block, nmo,
                                            nalpha, nbeta, unused, nchol_vecs])
    occups = [i for i in range(0, nalpha)]
    occups += [i+nmo for i in range(0, nbeta)]
    dump['Hamiltonian/occups'] = numpy.array(occups)

def from_qmcpack_cholesky(filename):
    with h5py.File(filename, 'r') as fh5:
        real_ints = False
        try:
            enuc = fh5['Hamiltonian/Energies'][:].view(numpy.complex128).ravel()[0]
        except ValueError:
            enuc = fh5['Hamiltonian/Energies'][:][0]
            real_ints = True
        dims = fh5['Hamiltonian/dims'][:]
        nmo = dims[3]
        try:
            hcore = fh5['Hamiltonian/hcore'][:]
            hcore = hcore.view(numpy.complex128).reshape(nmo,nmo)
        except KeyError:
            # Old sparse format.
            hcore = fh5['Hamiltonian/H1'][:].view(numpy.complex128).ravel()
            idx = fh5['Hamiltonian/H1_indx'][:]
            row_ix = idx[::2]
            col_ix = idx[1::2]
            hcore = scipy.sparse.csr_matrix((hcore, (row_ix, col_ix))).toarray()
            hcore = numpy.tril(hcore, -1) + numpy.tril(hcore, 0).conj().T
        except ValueError:
            # Real format.
            hcore = fh5['Hamiltonian/hcore'][:]
            real_ints = True
        chunks = dims[2]
        idx = []
        h2 = []
        for ic in range(chunks):
            idx.append(fh5['Hamiltonian/Factorized/index_%i'%ic][:])
            if real_ints:
                h2.append(fh5['Hamiltonian/Factorized/vals_%i'%ic][:].ravel())
            else:
                h2.append(fh5['Hamiltonian/Factorized/vals_%i'%ic][:].view(numpy.complex128).ravel())
        idx = numpy.array([i for sub in idx for i in sub])
        h2 = numpy.array([v for sub in h2 for v in sub])
        nalpha = dims[4]
        nbeta = dims[5]
        nchol = dims[7]
        row_ix = idx[::2]
        col_ix = idx[1::2]
        chol_vecs = scipy.sparse.csr_matrix((h2, (row_ix, col_ix)),
                                            shape=(nmo*nmo,nchol))
        return (hcore, chol_vecs, enuc, int(nmo), int(nalpha), int(nbeta))

def dump_native(filename, hcore, eri, orthoAO, fock, nelec, enuc,
                orbs=None, verbose=True, coeffs=None):
    if verbose:
        print (" # Constructing trial wavefunctiom in ortho AO basis.")
    if len(fock.shape) == 3:
        if verbose:
            print (" # Writing UHF trial wavefunction.")
        if orbs is None:
            (mo_energies, orbs) = molecular_orbitals_uhf(fock, orthoAO)
        else:
            orbs = orbs
    else:
        if verbose:
            print (" # Writing RHF trial wavefunction.")
        if orbs is None:
            (mo_energies, orbs) = molecular_orbitals_rhf(fock, orthoAO)
        else:
            orbs = orbs
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
        fh5.create_dataset('orbs', data=orbs)
        fh5.create_dataset('coeffs', data=coeffs)

def dump_qmcpack(filename, wfn_file, hcore, eri, orthoAO, fock, nelec, enuc,
                 verbose=True, threshold=1e-5, sparse_zero=1e-16, orbs=None):
    if verbose:
        print (" # Constructing trial wavefunctiom in ortho AO basis.")
    if len(hcore.shape) == 3:
        if verbose:
            print (" # Writing UHF trial wavefunction.")
        if orbs is None:
            (mo_energies, orbs) = molecular_orbitals_uhf(fock, orthoAO)
        else:
            orbs = orbs
    else:
        if verbose:
            print (" # Writing RHF trial wavefunction.")
        if orbs is None:
            (mo_energies, orbs) = molecular_orbitals_rhf(fock, orthoAO)
        else:
            orbs = orbs
    dump_qmcpack_trial_wfn(orbs, nelec, wfn_file)
    nbasis = hcore.shape[-1]
    if verbose:
        print (" # Performing modified Cholesky decomposition on ERI tensor.")
    msq = nbasis * nbasis
    # Why did I transpose everything?
    # QMCPACK expects [M^2, N_chol]
    # Internally store [N_chol, M^2]
    if isinstance(eri, list):
        chol_vecsa = modified_cholesky(eri[0].reshape((msq, msq)), threshold,
                                       verbose=verbose).T
        chol_vecsa = modified_cholesky(eri[1].reshape((msq, msq)), threshold,
                                       verbose=verbose).T
    elif len(eri.shape) == 4:
        chol_vecs = modified_cholesky(eri.reshape((msq, msq)), threshold,
                                       verbose=verbose).T
        chol_vecs[numpy.abs(chol_vecs) < sparse_zero] = 0
        chol_vecs = scipy.sparse.csr_matrix(chol_vecs)
        mem = 64*chol_vecs.nnz/(1024.0**3)
    else:
        chol_vecs = eri.T
        chol_vecs[numpy.abs(chol_vecs) < sparse_zero] = 0
        chol_vecs = scipy.sparse.csr_matrix(chol_vecs)
        mem = 64*chol_vecs.nnz/(1024.0**3)
    if verbose:
        print (" # Total number of non-zero elements in sparse cholesky ERI"
               " tensor: %d"%chol_vecs.nnz)
        nelem = chol_vecs.shape[0]*chol_vecs.shape[1]
        print (" # Sparsity of ERI Cholesky tensor: "
               "%f"%(1-float(chol_vecs.nnz/nelem)))
        print (" # Total memory required for ERI tensor: %13.8e GB"%(mem))
    dump_qmcpack_cholesky(numpy.array([hcore, hcore]), chol_vecs, nelec,
                                      nbasis, enuc, filename=filename)

def qmcpack_wfn_namelist(nci, uhf, fullmo=True):
    return "&FCI\n UHF = %d\n NCI = %d \n %s TYPE = matrix\n/\n"%(uhf,nci,'FullMO\n' if fullmo else '')

def dump_qmcpack_trial_wfn(wfn, nelec, filename='wfn.dat'):
    UHF = len(wfn.shape) == 3
    # Single determinant for the moment.
    namelist = qmcpack_wfn_namelist(1, UHF)
    with open(filename, 'w') as f:
        f.write(namelist)
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

def read_qmcpack_wfn(filename, skip=9):
    with open(filename) as f:
        content = f.readlines()[skip:]
    useable = numpy.array([c.split() for c in content]).flatten()
    tuples = [ast.literal_eval(u) for u in useable]
    orbs = [complex(t[0], t[1]) for t in tuples]
    return numpy.array(orbs)

def read_phfmol(filename, nmo):
    with open(filename) as f:
        content = f.read().split()
    start = False
    idet = 0
    data = []
    for (i,f) in enumerate(content):
        if 'NCI' in f:
            try:
                ndets = int(content[i+1])
            except ValueError:
                ndets = int(content[i+2])
            dets = numpy.zeros((ndets,nmo,nmo), dtype=numpy.complex128)
        # print(f,start,data)
        # print(len(data),f)
        if 'Coefficients' in f:
            string_coeffs = content[i+1:i+1+ndets]
        if 'Determinant' in f:
            break
    start = i + 2
    coeffs = []
    for c in string_coeffs:
        v = ast.literal_eval(c)
        coeffs.append(complex(v[0],v[1]))

    for idet in range(ndets):
        end = start+nmo*nmo
        data = []
        for line in content[start:end]:
            v = ast.literal_eval(line)
            data.append(complex(v[0],v[1]))
        C = numpy.copy(numpy.array(data).reshape(nmo,nmo).T)
        dets[idet] = C
        dets[idet] = C
        start = end + 2
    return numpy.array(coeffs), dets

def write_phfmol_wavefunction(coeffs, dets, filename='wfn.dat', padding=0):
    with open(filename, 'w') as f:
        UHF = len(dets.shape) == 4
        namelist = qmcpack_wfn_namelist(len(coeffs), UHF)
        f.write(namelist)
        f.write('Coefficients:\n')
        for c in coeffs:
            f.write('({:13.8e},{:13.8e})\n'.format(c.real,c.imag))
        for idet, C in enumerate(dets):
            nmo = C.shape[-1]
            padded = numpy.pad(C,[(0,padding), (0,padding)],'constant')
            f.write('Determinant: {}\n'.format(idet+1))
            # Write in fortran order.
            for cij in numpy.ravel(padded, order='F'):
                f.write('({:13.8e},{:13.8e})\n'.format(cij.real,cij.imag))
