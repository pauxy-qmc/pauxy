import argparse
import functools
import numpy
import h5py
import sys
from pauxy.utils.from_pyscf import dump_pauxy

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
    dump_pauxy(options.input_scf, outfile=options.output)

if __name__ == '__main__':

    main(sys.argv[1:])
