"""Generate AFQMC data from PYSCF (molecular) simulation."""
import h5py
import numpy
import time
import scipy.linalg
from pauxy.utils.io import dump_native, dump_qmcpack
from pauxy.utils.linalg import get_orthoAO
from pauxy.utils.misc import dotdict
from pauxy.estimators.greens_function import gab
from pauxy.estimators.generic import (
        local_energy_generic_cholesky, core_contribution_cholesky
    )
from pyscf.lib.chkfile import load_mol
from pyscf import ao2mo, scf, fci
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.gto import cell
from pyscf.pbc.lib import chkfile
from pyscf.tools import fcidump

def dump_pauxy(chkfile=None, mol=None, mf=None, outfile='fcidump.h5',
               verbose=True, qmcpack=False, wfn_file='wfn.dat',
               chol_cut=1e-5, sparse_zero=1e-16, pbc=False, cholesky=False,
               cas=None):
    if chkfile is not None:
        (hcore, fock, oao, enuc, mol, orbs, mf) = (
                                      from_pyscf_chkfile(chkfile, verbose, pbc)
                                      )
    else:
        (hcore, fock, oao, enuc) = from_pyscf_scf(mf)
        mol = mf.mol
    if verbose:
        print(" # Transforming hcore and eri to ortho AO basis.")
    nbasis = h1e.shape[-1]
    h1e, eri = generate_integrals(mol, hcore, oao, chol_cut=chol_cut,
                                  verbose=verbose, cas=cas)
    dump_qmcpack(outfile, wfn_file, h1e, eri, orthoAO, fock,
                 mol.nelec, enuc, threshold=chol_cut,
                 sparse_zero=sparse_zero, orbs=orbs)

def integrals_from_scf(mf, chol_cut=1e-5, verbose=0, cas=None, ortho_ao=True):
    mol = mf.mol
    ecore = mf.energy_nuc()
    hcore = mf.get_hcore()
    if ortho_ao:
        s1e = mf.mol.intor('int1e_ovlp_sph')
        oao = get_orthoAO(s1e)
    else:
        oao = mf.mo_coeff
    h1e, eri = generate_integrals(mol, hcore, oao,
                                  chol_cut=chol_cut,
                                  verbose=verbose,
                                  cas=cas)
    return h1e, eri, ecore, oao

def integrals_from_chkfile(chkfile, chol_cut=1e-5, verbose=False, cas=None):
    (hcore, fock, oao, ecore, mol, orbs, mf, coeffs) = from_pyscf_chkfile(chkfile, verbose)
    h1e, eri = generate_integrals(mol, hcore, oao,
                                  chol_cut=chol_cut,
                                  verbose=verbose,
                                  cas=cas)
    return h1e, eri, ecore, oao, mol

def generate_integrals(mol, hcore, oao, chol_cut=1e-5, verbose=0, cas=None):
    eri = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)
    if verbose:
        print("# Orthogonalising Cholesky vectors.")
    start = time.time()
    h1e = numpy.dot(oao.conj().T, numpy.dot(hcore, oao))
    ao2mo_chol(eri, oao)
    if verbose:
        print("# Time to orthogonalise: %f"%(time.time() - start))
    if cas is not None:
        nfzc = (sum(mol.nelec)-cas[0])//2
        ncas = cas[1]
        nfzv = nbasis - ncas - nfzc
        h1e, eri, enuc = freeze_core(h1e, eri, enuc, nfzc, ncas, verbose)
        h1e = h1e[0]
        nelec = (mol.nelec[0]-nfzc, mol.nelec[1]-nfzc)
        mol.nelec = nelec
        if len(orbs.shape) == 3:
            orbs = orbs[:,nfzc:nbasis-nfzv,nfzc:nbasis-nfzv]
        else:
            orbs = orbs[nfzc:nbasis-nfzv,nfzc:nbasis-nfzv]
    return h1e, eri

def freeze_core(h1e, chol, ecore, nc, ncas, verbose=True):
    # 1. Construct one-body hamiltonian
    nbasis = h1e.shape[-1]
    chol = chol.reshape((-1,nbasis,nbasis))
    system = dotdict({'H1': numpy.array([h1e,h1e]),
                      'chol_vecs': chol,
                      'ecore': ecore})
    psi = numpy.identity(nbasis)[:,:nc]
    Gcore = gab(psi,psi)
    ecore = local_energy_generic_cholesky(system, [Gcore,Gcore])[0]
    (hc_a, hc_b) = core_contribution_cholesky(system, [Gcore,Gcore])
    h1e = numpy.array([h1e,h1e])
    h1e[0] = h1e[0] + 2*hc_a
    h1e[1] = h1e[1] + 2*hc_b
    h1e = h1e[:,nc:nc+ncas,nc:nc+ncas]
    nchol = chol.shape[0]
    chol = chol[:,nc:nc+ncas,nc:nc+ncas].reshape((nchol,-1))
    # 4. Subtract one-body term from writing H2 as sum of squares.
    if verbose:
        print(" # Number of active orbitals: %d"%ncas)
        print(" # Freezing %d core electrons and %d virtuals."
              %(2*nc, nbasis-nc-ncas))
        print(" # Frozen core energy : %13.8e"%ecore.real)
    return h1e, chol, ecore

def ao2mo_chol(eri, C):
    nb = C.shape[-1]
    for i, cv in enumerate(eri):
        half = numpy.dot(cv.reshape(nb,nb), C)
        eri[i] = numpy.dot(C.T, half).ravel()

def from_pyscf_chkfile(scfdump, verbose=True, pbc=False):
    if pbc:
        mol = chkfile.load_cell(scfdump)
        mf = pbcscf.KRHF(mol)
    else:
        mol = load_mol(scfdump)
        mf = scf.RHF(mol)
    with h5py.File(scfdump, 'r') as fh5:
        try:
            hcore = fh5['/scf/hcore'][:]
        except KeyError:
            hcore = mol.intor_symmetric('int1e_nuc')
            hcore += mol.intor_symmetric('int1e_kin')
            if len(mol._ecpbas) > 0:
                hcore += mol.intor_symmetric('ECPScalar')
        try:
            fock = fh5['/scf/fock'][:]
        except KeyError:
            fock = None
        try:
            oao = fh5['/scf/orthoAORot'][:]
        except:
            s1e = mf.mol.intor('int1e_ovlp_sph')
            oao = get_orthoAO(s1e)
        try:
            orbs = fh5['/scf/orbs'][:]
        except KeyError:
            orbs = None
        try:
            coeffs = fh5['/scf/coeffs'][:]
        except KeyError:
            coeffs = numpy.array([1])
        try:
            enuc = fh5['/scf/enuc'][()]
        except KeyError:
            enuc = mf.energy_nuc()
    if verbose:
        print("# Generating PAUXY input from %s."%chkfile)
        print("# (nalpha, nbeta): (%d, %d)"%mol.nelec)
        print("# nbasis: %d"%hcore.shape[-1])
    return (hcore, fock, oao, enuc, mol, orbs, mf, coeffs)

def from_pyscf_scf(mf, verbose=True):
    hcore = mf.get_hcore()
    fock = hcore + mf.get_veff()
    s1e = mf.mol.intor('int1e_ovlp_sph')
    orthoAO = get_orthoAO(s1e)
    enuc = mf.energy_nuc()
    if verbose:
        print ("# Generating PAUXY input PYSCF mol and scf objects.")
        print ("# (nalpha, nbeta): (%d, %d)"%mf.mol.nelec)
        print ("# nbasis: %d"%hcore.shape[-1])
    return (hcore, fock, orthoAO, enuc)

def write_fcidump(system, name='FCIDUMP'):
    fcidump.from_integrals(name, system.H1[0], system.h2e,
                           system.H1[0].shape[0], system.ne, nuc=system.ecore)

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

def multi_det_wavefunction(mc, weight_cutoff=0.95, verbose=False,
                           max_ndets=1e5, norb=None,
                           filename="multi_det.dat"):
    """Generate multi determinant particle-hole trial wavefunction.

    Format adopted to be compatable with QMCPACK PHMSD type wavefunction.

    Parameters
    ----------
    mc : pyscf CI solver type object
        Input object containing multi determinant coefficients.
    weight_cutoff : float, optional
        Print determinants until accumulated weight equals weight_cutoff.
        Default 0.95.
    verbose : bool
        Print information about process. Default False.
    max_ndets : int
        Max number of determinants to print out. Default 1e5.
    norb : int or None, optional
        Total number of orbitals in simulation. Used if we want to run CI within
        active space but QMC in full space. Deault None.
    filename : string
        Output filename. Default "multi_det.dat"
    """
    occlists = fci.cistring._gen_occslst(range(mc.ncas), mc.nelecas[0])

    ci_coeffs = mc.ci.ravel()
    # Sort coefficients in terms of increasing absolute weight.
    ix_sort = numpy.argsort(numpy.abs(ci_coeffs))[::-1]
    cweight = numpy.cumsum(ci_coeffs[ix_sort]**2)
    max_det = numpy.searchsorted(cweight, weight_cutoff)
    ci_coeffs = ci_coeffs[ix_sort]
    if verbose:
        print ("Number of dets in CAS space: %d"%len(occlists)**2)
        print ("Number of dets in CI expansion: %d"%max_det)

    output = open(filename, 'w')
    namelist = "&FCI\n UHF = 0\n NCI = %d\n TYPE = occ\n&END" % max_det
    output.write(namelist+'\n')
    output.write("Configurations:"+'\n')
    if norb is None:
        norb = mc.ncas

    for idet in range(max_det):
        if mc.ncore > 0:
            ocore_up = ' '.join('{:d}'.format(x+1) for x in range(mc.ncore))
            ocore_dn = ' '.join('{:d}'.format(x+1+norb) for x in range(mc.ncore))
        else:
            ocore_up = ' '
            ocore_dn = ' '
        coeff = '%.13f'%ci_coeffs[idet]
        ix_alpha = ix_sort[idet] // len(occlists)
        ix_beta = ix_sort[idet] % len(occlists)
        ia = occlists[ix_alpha]
        ib = occlists[ix_beta]
        oup = ' '.join('{:d}'.format(x+1+mc.ncore) for x in ia)
        odown = ' '.join('{:d}'.format(x+norb+1+mc.ncore) for x in ib)
        output.write(coeff+' '+ocore_up+' '+oup+' '+ocore_dn+' '+odown+'\n')

def get_pyscf_wfn(system, mf):
    """Return trial wavefunction from pyscf mf object.
    """
    C = mf.mo_coeff
    na = system.nup
    nb = system.ndown
    X = system.oao
    Xinv = scipy.linalg.inv(X)
    # TODO : Update for mcscf object.
    if len(C.shape) == 3:
        # UHF trial.
        pa = numpy.dot(Xinv, C[0][:,:na])
        pb = numpy.dot(Xinv, C[1][:,:nb])
    else:
        pa = numpy.dot(Xinv, C[:,:na])
        pb = pa.copy()
    wfn = numpy.zeros((system.nbasis, na+nb), dtype=numpy.complex128)
    wfn[:,:na] = pa
    wfn[:,na:] = pb
    return (numpy.array([1.0+0j]), wfn)
