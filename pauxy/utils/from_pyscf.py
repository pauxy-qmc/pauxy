"""Generate AFQMC data from PYSCF (molecular) simulation."""
import h5py
import numpy
import time
from pauxy.utils.io import dump_native, dump_qmcpack
from pauxy.utils.linalg import get_orthoAO
from pyscf.lib.chkfile import load_mol
from pyscf import ao2mo, scf, fci, mcscf, hci
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.gto import cell
from pyscf.pbc.lib import chkfile
from pyscf.tools import fcidump

def dump_pauxy(chkfile=None, mol=None, mf=None, outfile='fcidump.h5',
               verbose=True, qmcpack=False, wfn_file='wfn.dat',
               chol_cut=1e-5, sparse_zero=1e-16, pbc=False, cholesky=False):
    if chkfile is not None:
        (hcore, fock, orthoAO, enuc, mol, orbs, mf, coeffs) = from_pyscf_chkfile(chkfile, verbose, pbc)
    else:
        (hcore, fock, orthoAO, enuc) = from_pyscf_mol(mol, mf)
    if verbose:
        print (" # Transforming hcore and eri to ortho AO basis.")
    h1e = numpy.dot(orthoAO.conj().T, numpy.dot(hcore, orthoAO))
    nbasis = h1e.shape[-1]
    if cholesky:
        eri = chunked_cholesky(mol, max_error=chol_cut, verbose=True)
        if verbose:
            print (" # Orthogonalising Cholesky vectors.")
        start = time.time()
        for i, c in enumerate(eri):
            half = numpy.dot(c.reshape(nbasis,nbasis), orthoAO)
            eri[i] = numpy.dot(orthoAO.T, half).ravel()
        if verbose:
            print (" # Time to orthogonalise: %f"%(time.time() - start))
    elif pbc:
        eri = mf.with_df.ao2mo(orthoAO, compact=False).reshape(nbasis, nbasis,
                                                               nbasis, nbasis)
    else:
        if len(orthoAO.shape) == 3:
            eria = ao2mo.kernel(mol, orthoAO[0], compact=False).reshape(nbasis,nbasis,nbasis,nbasis)
            erib = ao2mo.kernel(mol, orthoAO[1], compact=False).reshape(nbasis,nbasis,nbasis,nbasis)
            eri = [eria, erib]
        else:
            eri = ao2mo.kernel(mol, orthoAO, compact=False).reshape(nbasis,nbasis,nbasis,nbasis)
    if qmcpack:
        dump_qmcpack(outfile, wfn_file, h1e, eri, orthoAO, fock,
                     mol.nelec, enuc, threshold=chol_cut,
                     sparse_zero=sparse_zero, orbs=orbs)
    else:
        dump_native(outfile, h1e, eri, orthoAO, fock, mol.nelec, enuc,
                    orbs=orbs, coeffs=coeffs)
    return eri

def from_pyscf_chkfile(scfdump, verbose=True, pbc=False):
    if pbc:
        mol = chkfile.load_cell(scfdump)
        mf = pbcscf.KRHF(mol)
    else:
        mol = load_mol(scfdump)
        mf = scf.RHF(mol)
    with h5py.File(scfdump, 'r') as fh5:
        hcore = fh5['/scf/hcore'][:]
        fock = fh5['/scf/fock'][:]
        orthoAO = fh5['/scf/orthoAORot'][:]
        orbs = fh5['/scf/orbs'][:]
        try:
            coeffs = fh5['/scf/coeffs'][:]
        except KeyError:
            coeffs = numpy.array([1])
        try:
            enuc = fh5['/scf/enuc'][()]
        except KeyError:
            enuc = mf.energy_nuc()
    if verbose:
        print (" # Generating PAUXY input from %s."%chkfile)
        print (" # (nalpha, nbeta): (%d, %d)"%mol.nelec)
        print (" # nbasis: %d"%hcore.shape[-1])
    return (hcore, fock, orthoAO, enuc, mol, orbs, mf, coeffs)

def from_pyscf_mol(mol, mf, verbose=True):
    hcore = mf.get_hcore()
    fock = hcore + mf.get_veff()
    s1e = mol.intor('int1e_ovlp_sph')
    orthoAO = get_orthoAO(s1e)
    enuc = mf.energy_nuc()
    if verbose:
        print (" # Generating PAUXY input PYSCF mol and scf objects.")
        print (" # (nalpha, nbeta): (%d, %d)"%mol.nelec)
        print (" # nbasis: %d"%hcore.shape[-1])
    return (hcore, fock, orthoAO, enuc)

def write_fcidump(system):
    fcidump.from_integrals('FCIDUMP', system.T[0], system.h2e,
                           system.T[0].shape[0], system.ne, nuc=system.ecore)

def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`numpy.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = numpy.zeros(nao*nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = numpy.zeros((nchol_max, nao*nao))
    eri = numpy.zeros((nao,nao,nao,nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0,mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2*l+1)*nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0,mol.nbas):
        shls = (i,i+1,0,mol.nbas,i,i+1,0,mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag:ndiag+di*nao] = buf.reshape(di*nao,di*nao).diagonal()
        ndiag += di * nao
    nu = numpy.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print ("# Generating Cholesky decomposition of ERIs."%nchol_max)
        print ("# max number of cholesky vectors = %d"%nchol_max)
        print ("# iteration %5d: delta_max = %f"%(0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = numpy.searchsorted(dims, j)
    sl = numpy.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = numpy.zeros(nao*nao)
    # ERI[:,jl]
    eri_col = mol.intor('int2e_sph',
                         shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
    cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
    chol_vecs[0] = numpy.copy(eri_col[:,:,cj,cl].reshape(nao*nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = \sum_x L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = numpy.argmax(numpy.abs(delta))
        delta_max = numpy.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = numpy.searchsorted(dims, j)
        sl = numpy.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor('int2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
        # Select correct ERI chunk from shell.
        cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
        Munu0 = eri_col[:,:,cj,cl].reshape(nao*nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = numpy.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        chol_vecs[nchol+1] = (Munu0 - R) / (delta_max)**0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print ("# iteration %5d: delta_max = %13.8e: time = %13.8e"%info)

    return chol_vecs[:nchol]

def sci_wavefunction(mf, nelecas, ncas, ncore, select_cutoff=1e-10,
                     ci_coeff_cutoff=1e-3, verbose=False):
    """Generate SCI trial wavefunction.

    Parameters
    ----------
    nelecas : int
        Number of active orbitals.
    ncas : int or tuple of ints.
        Total number of active electrons, or number of active alpha and beta
        electrons.
    ncore : int
        Total number of core electrons.
    select_cutoff : float
        Selection criteria.
    ci_coeff_cutoff : float
        CI coefficient cutoff.
    """
    mc = mcscf.CASCI(mf, ncas, nelecas, ncore=ncore)
    mc.fcisolver = hci.SCI(mol)
    mc.fcisolver.select_cutoff = select_cutoff
    mc.fcisolver.ci_coeff_cutoff = ci_coeff_cutoff
    mc.kernel()
    occlists = fci.cistring._gen_occslst(range(cas), nelec//2)

    if verbose:
        print ("Max number of dets : %d"%len(occlists)**2)
        print ("Non-zero : %d"%non_zero)

    coeffs = []
    # for i, ia in enumerate(occlists):
        # for j, ib in enumerate(occlists):
            # coeffs.append(%mc.ci[i,j])
            # oup =
            # odown = ' '.join('{:d}'.format(x+norb+1) for x in ib)
