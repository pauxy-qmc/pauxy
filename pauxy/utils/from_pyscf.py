"""Generate AFQMC data from PYSCF (molecular) simulation."""
import h5py
from pauxy.utils.io import dump_native, dump_qmcpack
from pauxy.utils.linalg import unitary, get_orthoAO
from pyscf.lib.chkfile import load_mol
from pyscf import ao2mo, scf, fci, mcscf, hci

def dump_pauxy(chkfile=None, mol=None, mf=None, outfile='fcidump.h5',
               verbose=True, qmcpack=False, wfn_file='wfn.dat',
               chol_cut=1e-5, sparse_zero=1e-16):
    if chkfile is not None:
        (hcore, fock, orthoAO, enuc, mol, orbs) = from_pyscf_chkfile(chkfile, verbose)
    else:
        (hcore, fock, orthoAO, enuc) = from_pyscf_mol(mol, mf)
    if verbose:
        print (" # Transforming hcore and eri to ortho AO basis.")
    h1e = unitary(hcore, orthoAO)
    nbasis = h1e.shape[-1]
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
                orbs=orbs)


def from_pyscf_chkfile(chkfile, verbose=True):
    with h5py.File(chkfile, 'r') as fh5:
        hcore = fh5['/scf/hcore'][:]
        fock = fh5['/scf/fock'][:]
        orthoAO = fh5['/scf/orthoAORot'][:]
        orbs = fh5['/scf/orbs'][:]
    mol = load_mol(chkfile)
    mf = scf.HF(mol)
    enuc = mf.energy_nuc()
    if verbose:
        print (" # Generating PAUXY input from %s."%chkfile)
        print (" # (nalpha, nbeta): (%d, %d)"%mol.nelec)
        print (" # nbasis: %d"%hcore.shape[-1])
    return (hcore, fock, orthoAO, enuc, mol, orbs)

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
