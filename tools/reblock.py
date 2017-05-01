#!/usr/bin/env python
'''Run a reblocking analysis on AFQMCPY QMC output files. Heavily adapted from
HANDE'''

import argparse
import os
import sys
import pandas as pd
import pyblock


def run_blocking_analysis(filename, start_iter):
    '''
'''

    with open(filename[0]) as f:
        for ln, line in enumerate(f):
            if 'End of input options' in line:
                skip = ln + 1
                break

    data = pd.read_csv(filename[0], skiprows=skip, sep=r'\s+').drop(['iteration', 'exp(delta)'], axis=1)[start_iter::]
    (data_len, reblock, covariances) = pyblock.pd_utils.reblock(data)
    cov = covariances.xs('Weight', level=1)['E_num']
    numerator = reblock.ix[:,'E_num']
    denominator = reblock.ix[:,'Weight']
    projected_energy = pyblock.error.ratio(numerator, denominator, cov, 4)
    projected_energy.columns = pd.MultiIndex.from_tuples([('Energy', col)
                                    for col in projected_energy.columns])
    reblock = pd.concat([reblock, projected_energy], axis=1)
    summary = pyblock.pd_utils.reblock_summary(reblock)

    return (reblock, summary)


def parse_args(args):
    '''Parse command-line arguments.

Parameters
----------
args : list of strings
    command-line arguments.

Returns
-------
(filenames, start_iteration)

where

filenames : list of strings
    list of QMC output files
start_iteration : int
    iteration number from which statistics should be gathered.
'''

    cols = pd.util.terminal.get_terminal_size()[0]
    if not sys.stdout.isatty():
        cols = -1

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('-s', '--start', type=int, dest='start_iteration',
                        default=None, help='Iteration number from which to '
                        'gather statistics.  Default: Try finding starting '
                        'iteration automatically. ')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        default=False, help='Increase verbosity of output.')
    parser.add_argument('filenames', nargs=argparse.REMAINDER,
                        help='Space-separated list of files to analyse.')

    options = parser.parse_args(args)

    if not options.filenames:
        parser.print_help()
        sys.exit(1)

    # options.filenames = [[fname] for fname in options.filenames]

    return options


def main(args):
    '''Run reblocking and data analysis on HANDE output.

Parameters
----------
args : list of strings
    command-line arguments.

Returns
-------
None.
'''

    options = parse_args(args)
    (reblock, summary) = run_blocking_analysis(options.filenames, options.start_iteration)

    if options.verbose:
        print (reblock)
    else:
        print (summary)

if __name__ == '__main__':

    main(sys.argv[1:])
