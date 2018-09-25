"""Generate AFQMC data from PYSCF (molecular) simulation."""
import h5py
from pauxy.utils.io import dump_native, dump_qmcpack
from pauxy.utils.linalg import unitary, get_orthoAO
from pyscf.lib.chkfile import load_mol
from pyscf import ao2mo, scf

def dump_pauxy(chkfile=None, mol=None, mf=None, outfile='fcidump.h5',
               verbose=True, qmcpack=False, wfn_file='wfn.dat'):
    if chkfile is not None:
        (hcore, fock, orthoAO, enuc, mol) = from_pyscf_chkfile(chkfile, verbose)
    else:
        (hcore, fock, orthoAO, enuc) = from_pyscf_mol(mol, mf)
    if verbose:
        print (" # Transforming hcore and eri to ortho AO basis.")
    h1e = unitary(hcore, orthoAO)
    nbasis = h1e.shape[-1]
    eri = ao2mo.kernel(mol, orthoAO, compact=False).reshape(nbasis,nbasis,nbasis,nbasis)
    if qmcpack:
        dump_qmcpack(outfile, wfn_file, h1e, eri,
                     orthoAO, fock, mol.nelec, enuc)
    else:
        dump_native(outfile, h1e, eri, orthoAO, fock, mol.nelec, enuc)


def from_pyscf_chkfile(chkfile, verbose=True):
    with h5py.File(chkfile, 'r') as fh5:
        hcore = fh5['/scf/hcore'][:]
        fock = fh5['/scf/fock'][:]
        orthoAO = fh5['/scf/orthoAORot'][:]
    mol = load_mol(chkfile)
    mf = scf.HF(mol)
    enuc = mf.energy_nuc()
    if verbose:
        print (" # Generating PAUXY input from %s."%chkfile)
        print (" # (nalpha, nbeta): (%d, %d)"%mol.nelec)
        print (" # nbasis: %d"%hcore.shape[-1])
    return (hcore, fock, orthoAO, enuc, mol)

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
