import argparse
import functools
import numpy
import h5py
import sys
from pyscf.lib.chkfile import load_mol
from pyscf import ao2mo, scf
from pauxy.utils.io import dump_native

def parse_args(args):
    """Parse command-line arguments.

    Parameters
    ----------
    args : list of strings
        command-line arguments.

    Returns
    -------
    options : :class:`argparse.ArgumentParser`
        Command line arguments.
    """

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('-i', '--input', dest='input_scf', type=str,
                        default=None, help='PYSCF scf chkfile.')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='fcidump.h5', help='Output file name for PAUXY data.')

    options = parser.parse_args(args)

    if not options.input_scf:
        parser.print_help()
        sys.exit(1)

    return options

def main(args):
    """Extract observable from analysed output.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """

    options = parse_args(args)
    with h5py.File(options.input_scf, 'r') as fh5:
        hcore = fh5['/scf/hcore'][:]
        fock = fh5['/scf/fock'][:]
        orthoAO = fh5['/scf/orthoAORot'][:]
        mo_coeff = fh5['/scf/mo_coeff'][:]
    mol = load_mol(options.input_scf)
    mf = scf.HF(mol)
    enuc = mf.energy_nuc()
    nbasis = fock.shape[-1]
    print (" # Generating PAUXY input from %s."%options.input_scf)
    print (" # (nalpha, nbeta): (%d, %d)"%mol.nelec)
    print (" # nbasis: %d"%hcore.shape[-1])
    print (" # Transforming hcore and eri to ortho AO basis.")
    h1e = functools.reduce(numpy.dot, (orthoAO.T, hcore, orthoAO))
    print (" # Total number of elements in ERI tensor: %d"%nbasis**4)
    mem = 64*nbasis**4/(1024.0*1024.0*1024.0)
    print (" # Total memory required for ERI tensor: %13.8e GB"%(mem))
    eri = ao2mo.kernel(mol, orthoAO, compact=False).reshape(nbasis,nbasis,nbasis,nbasis)
    print (" # Constructing trial wavefunction in ortho AO basis.")
    dump_native(options.output, h1e, eri, orthoAO, fock, mol.nelec, enuc)

if __name__ == '__main__':

    main(sys.argv[1:])
