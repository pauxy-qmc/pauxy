#!/usr/bin/env python
'''Run a reblocking analysis on AFQMCPY QMC output files. Heavily adapted from
HANDE'''

import argparse
import os
import sys
import pandas as pd
import json
_script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(_script_dir, 'analysis'))
import analysis.blocking
import pyblock
import matplotlib.pyplot as plt


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

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('-s', '--start', type=int, dest='start_iteration',
                        default=None, help='Iteration number from which to '
                        'gather statistics.  Default: Try finding starting '
                        'iteration automatically. ')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        default=False, help='Increase verbosity of output.')
    parser.add_argument('-l', '--loops', dest='loops', action='store_true',
                        default=False, help='Average over independent simulations.')
    parser.add_argument('-t', '--tail', dest='tail', action='store_true',
                        default=False, help='Short output.')
    parser.add_argument('-i', '--input', dest='input', action='store_true',
                        default=False, help='Extract input file.')
    parser.add_argument('-b', '--back-prop', dest='back_propagation', action='store_true',
                        default=False, help='Analyse back propagated estimates.')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true',
                        default=False, help='Make plots.')
    parser.add_argument('-c', '--correlation', dest='itcf', nargs='+', type=int,
                        default=None, help='Imaginary time correlation function.')
    parser.add_argument('-f', nargs='+', dest='filenames',
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
    if options.loops:
        if options.back_propagation:
            data = analysis.blocking.average_back_propagated(options.filenames,
                                                             options.start_iteration)
            if options.plot:
                fig, ax = plt.subplots(2, sharex=True)
                ax[0].errorbar(data.nbp.values, data['T'].values,
                        yerr=data.T_error.values, fmt='o')
                ax[1].errorbar(data.nbp.values, data.V.values,
                        data.V_error.values, fmt='o')
                ax[0].set_ylabel('T')
                ax[1].set_ylabel('V')
                plt.savefig('conv_bp.pdf', fmt='pdf')
                plt.show()
        elif options.itcf is not None:
            data = analysis.blocking.average_itcf(options.filenames,
                    options.itcf, options.start_iteration)
        else:
            data = analysis.blocking.average_tau(options.filenames)
        if options.tail:
            print (data.tail(1).to_string(index=False))
        else:
            print (data.to_string(index=False))
    elif options.input:
        metadata = analysis.extraction.extract_data_sets(options.filenames)[0][0]
        print (json.dumps(metadata, sort_keys=False, indent=4))
    else:
        (reblock, summary) = analysis.blocking.run_blocking_analysis(options.filenames, options.start_iteration)
        if options.verbose:
            print (reblock)
        else:
            print (summary)

if __name__ == '__main__':

    main(sys.argv[1:])
