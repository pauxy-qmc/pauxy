import ast
import h5py
import json
import numpy
import scipy.sparse
import sys
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
    if len(h2.shape) == 3:
        h2 = h2.reshape((-1,nmo*nmo)).T.copy()
        h2 = scipy.sparse.csr_matrix(h2)
    nnz = h2.nnz
    # number of cholesky vectors
    nchol_vecs = h2.shape[-1]
    dump['Hamiltonian/Factorized/block_sizes'] = numpy.array([nnz])
    (h2_unpacked, idx) = to_qmcpack_index(h2)
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

def from_qmcpack_complex(data, shape):
    return data.view(numpy.complex128).ravel().reshape(shape)

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
            out.write('(%.16e,%.16e) '%(val.real, val.imag))
        out.write('\n')

def read_qmcpack_wfn(filename, skip=9):
    with open(filename) as f:
        content = f.readlines()[skip:]
    useable = numpy.array([c.split() for c in content]).flatten()
    tuples = [ast.literal_eval(u) for u in useable]
    orbs = [complex(t[0], t[1]) for t in tuples]
    return numpy.array(orbs)

def read_phfmol(filename, nmo, na, nb):
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
            dets = numpy.zeros((ndets,nmo,na+nb), dtype=numpy.complex128)
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
        dets[idet,:,:na] = C[:,:na]
        dets[idet,:,na:] = C[:,:nb]
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

def get_input_value(inputs, key, default=0, alias=None, verbose=False):
    """Helper routine to parse input options.
    """
    val = inputs.get(key, None)
    if val is None:
        if alias is not None:
            for a in alias:
                val = inputs.get(a, None)
                if val is not None:
                    break
        if val is None:
            val = default
            if verbose:
                print("# Warning: {} not specified. Setting to default value"
                      " of {}.".format(key, default))
    return val

def read_qmcpack_wfn_hdf(filename):
    try:
        with h5py.File(filename, 'r') as fh5:
            wgroup = fh5['Wavefunction/NOMSD']
            wfn, psi0 = read_qmcpack_nomsd_hdf5(wgroup)
    except KeyError:
        with h5py.File(filename, 'r') as fh5:
            wgroup = fh5['Wavefunction/PHMSD']
            wfn, psi0 = read_qmcpack_phmsd_hdf5(wgroup)
    except KeyError:
        print("Wavefunction not found.")
        sys.exit()
    return wfn, psi0

def read_qmcpack_nomsd_hdf5(wgroup):
    dims = wgroup['dims']
    nmo = dims[0]
    na = dims[1]
    nb = dims[2]
    walker_type = dims[3]
    if walker_type == 2:
        uhf = True
    else:
        uhf = False
    nci = dims[4]
    coeffs = from_qmcpack_complex(wgroup['ci_coeffs'][:], (nci,))
    psi0a = from_qmcpack_complex(wgroup['Psi0_alpha'][:], (nmo,na))
    if uhf:
        psi0b = from_qmcpack_complex(wgroup['Psi0_beta'][:], (nmo,nb))
    psi0 = numpy.zeros((nmo,na+nb),dtype=numpy.complex128)
    psi0[:,:na] = psi0a.copy()
    if uhf:
        psi0[:,na:] = psi0b.copy()
    else:
        psi0[:,na:] = psi0a.copy()
    wfn = numpy.zeros((nci,nmo,na+nb), dtype=numpy.complex128)
    for idet in range(nci):
        ix = 2*idet if uhf else idet
        pa = from_qmcpack_sparse(wgroup['PsiT_{:d}/'.format(idet)])
        wfn[idet,:,:na] = pa
        if uhf:
            ix = 2*idet + 1
            wfn[idet,:,na:] = from_qmcpack_sparse(wgroup['PsiT_{:d}/'.format(ix)])
        else:
            wfn[idet,:,na:] = pa
    return (coeffs,wfn), psi0

def read_qmcpack_phmsd_hdf5(wgroup):
    dims = wgroup['dims']
    nmo = dims[0]
    na = dims[1]
    nb = dims[2]
    walker_type = dims[3]
    if walker_type == 2:
        uhf = True
    else:
        uhf = False
    nci = dims[4]
    coeffs = from_qmcpack_complex(wgroup['ci_coeffs'][:], (nci,))
    occs = wgroup['occs'][:].reshape((nci,na+nb))
    occa = occs[:,:na]
    occb = occs[:,na:]-nmo
    wfn = (coeffs, occa, occb)
    psi0a = from_qmcpack_complex(wgroup['Psi0_alpha'][:], (nmo,na))
    if uhf:
        psi0b = from_qmcpack_complex(wgroup['Psi0_beta'][:], (nmo,nb))
    psi0 = numpy.zeros((nmo,na+nb),dtype=numpy.complex128)
    psi0[:,:na] = psi0a.copy()
    if uhf:
        psi0[:,na:] = psi0b.copy()
    else:
        psi0[:,na:] = psi0a.copy()
    return wfn, psi0

def write_qmcpack_wfn(filename, wfn, walker_type, nelec, norb, init=None):
    # User defined wavefunction.
    # PHMSD is a list of tuple of (ci, occa, occb).
    # NOMSD is a tuple of (list, numpy.ndarray).
    if len(wfn) == 3:
        coeffs, occa, occb = wfn
        wfn_type = 'PHMSD'
    elif len(wfn) == 2:
        coeffs, wfn = wfn
        wfn_type = 'NOMSD'
    else:
        print("Unknown wavefunction type passed.")
        sys.exit()

    fh5 = h5py.File(filename, 'a')
    nalpha, nbeta = nelec
    # TODO: FIX for GHF eventually.
    if walker_type == 'ghf':
        walker_type = 3
    elif walker_type == 'uhf':
        walker_type = 2
        uhf = True
    else:
        walker_type = 1
        uhf = False
    if wfn_type == 'PHMSD':
        walker_type = 2
    if wfn_type == 'NOMSD':
        try:
            wfn_group = fh5.create_group('Wavefunction/NOMSD')
        except ValueError:
            del fh5['Wavefunction/NOMSD']
            wfn_group = fh5.create_group('Wavefunction/NOMSD')
        write_nomsd(wfn_group, wfn, uhf, nelec, init=init)
    else:
        try:
            wfn_group = fh5.create_group('Wavefunction/PHMSD')
        except ValueError:
            # print(" # Warning: Found existing wavefunction group. Removing.")
            del fh5['Wavefunction/PHMSD']
            wfn_group = fh5.create_group('Wavefunction/PHMSD')
        write_phmsd(wfn_group, occa, occb, nelec, norb, init=init)
    wfn_group['ci_coeffs'] = to_qmcpack_complex(coeffs)
    dims = [norb, nalpha, nbeta, walker_type, len(coeffs)]
    wfn_group['dims'] = numpy.array(dims, dtype=numpy.int32)
    fh5.close()

def write_nomsd(fh5, wfn, uhf, nelec, thresh=1e-8, init=None):
    """Write NOMSD to HDF.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    wfn : :class:`numpy.ndarray`
        NOMSD trial wavefunctions.
    uhf : bool
        UHF style wavefunction.
    nelec : tuple
        Number of alpha and beta electrons.
    thresh : float
        Threshold for writing wavefunction elements.
    """
    nalpha, nbeta = nelec
    wfn[abs(wfn) < thresh] = 0.0
    if init is not None:
        fh5['Psi0_alpha'] = to_qmcpack_complex(init[0])
        fh5['Psi0_beta'] = to_qmcpack_complex(init[1])
    else:
        fh5['Psi0_alpha'] = to_qmcpack_complex(wfn[0,:,:nalpha].copy())
        if uhf:
            fh5['Psi0_beta'] = to_qmcpack_complex(wfn[0,:,nalpha:].copy())
    for idet, w in enumerate(wfn):
        # QMCPACK stores this internally as a csr matrix, so first convert.
        ix = 2*idet if uhf else idet
        psia = scipy.sparse.csr_matrix(w[:,:nalpha].conj().T)
        write_nomsd_single(fh5, psia, ix)
        if uhf:
            ix = 2*idet + 1
            psib = scipy.sparse.csr_matrix(w[:,nalpha:].conj().T)
            write_nomsd_single(fh5, psib, ix)

def write_nomsd_single(fh5, psi, idet):
    """Write single component of NOMSD to hdf.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    psi : :class:`scipy.sparse.csr_matrix`
        Sparse representation of trial wavefunction.
    idet : int
        Determinant number.
    """
    base = 'PsiT_{:d}/'.format(idet)
    dims = [psi.shape[0], psi.shape[1], psi.nnz]
    fh5[base+'dims'] = numpy.array(dims, dtype=numpy.int32)
    fh5[base+'data_'] = to_qmcpack_complex(psi.data)
    fh5[base+'jdata_'] = psi.indices
    fh5[base+'pointers_begin_'] = psi.indptr[:-1]
    fh5[base+'pointers_end_'] = psi.indptr[1:]

def write_phmsd(fh5, occa, occb, nelec, norb, init=None):
    """Write NOMSD to HDF.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    nelec : tuple
        Number of alpha and beta electrons.
    """
    # TODO: Update if we ever wanted "mixed" phmsd type wavefunctions.
    na, nb = nelec
    if init is not None:
        fh5['Psi0_alpha'] = to_qmcpack_complex(init[0])
        fh5['Psi0_beta'] = to_qmcpack_complex(init[1])
    else:
        init = numpy.eye(norb, dtype=numpy.complex128)
        fh5['Psi0_alpha'] = to_qmcpack_complex(init[:,occa[0]].copy())
        fh5['Psi0_beta'] = to_qmcpack_complex(init[:,occb[0]].copy())
    fh5['fullmo'] = numpy.array([0], dtype=numpy.int32)
    fh5['type'] = 0
    occs = numpy.zeros((len(occa), na+nb), dtype=numpy.int32)
    occs[:,:na] = numpy.array(occa)
    occs[:,na:] = norb + numpy.array(occb)
    # Reading 1D array currently in qmcpack.
    fh5['occs'] = occs.ravel()

def from_qmcpack_sparse(dset):
    """Will read actually A^{H} but return A.
    """
    dims = dset['dims'][:]
    wfn_shape = (dims[0],dims[1])
    nnz = dims[2]
    data = from_qmcpack_complex(dset['data_'][:],(nnz,))
    indices = dset['jdata_'][:]
    pbb = dset['pointers_begin_'][:]
    pbe = dset['pointers_end_'][:]
    indptr = numpy.zeros(dims[0]+1)
    indptr[:-1] = pbb
    indptr[-1] = pbe[-1]
    wfn = scipy.sparse.csr_matrix((data,indices,indptr),shape=wfn_shape)
    return wfn.toarray().conj().T.copy()

def to_qmcpack_complex(array):
    shape = array.shape
    return array.view(numpy.float64).reshape(shape+(2,))
