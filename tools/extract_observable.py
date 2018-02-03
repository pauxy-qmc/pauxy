#!/usr/bin/env python
'''Exctact element of green's function'''

import argparse
import os
import sys
import pandas as pd
import json
_script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(_script_dir, 'analysis'))
import matplotlib.pyplot as plt
from pauxy.analysis.blocking import analysed_itcf
from pauxy.analysis.blocking import analysed_energies
from pauxy.analysis.blocking import correlation_function


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
    parser.add_argument('-s', '--spin', type=str, dest='spin',
                        default=None, help='Spin component to extract.'
                        'Options: up/down')
    parser.add_argument('-t', '--type', type=str, dest='type',
                        default=None, help='Type of green\'s function to extract.'
                        'Options: lesser/greater')
    parser.add_argument('-k', '--kspace', dest='kspace', action='store_true',
                        default=False, help='Extract kspace green\'s function.')
    parser.add_argument('-e', '--elements',
                        type=lambda s: [int(item) for item in s.split(',')],
                        dest='elements', default=None,
                        help='Element to extract.')
    parser.add_argument('-o', '--observable', type=str, dest='obs',
                        default='energy', help='Data to extract')
    parser.add_argument('-f', nargs='+', dest='filename',
                        help='Space-separated list of files to analyse.')

    options = parser.parse_args(args)

    if not options.filename:
        parser.print_help()
        sys.exit(1)

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
    print_index = False
    if options.obs == 'itcf':
        results = extract_analysed_itcf(options.filename[0],
                                        options.elements,
                                        options.spin,
                                        options.type,
                                        options.kspace)
    elif options.obs == 'energy':
        results = analysed_energies(options.filename[0], 'mixed')
    elif options.obs == 'back_propagated':
        results = analysed_energies(options.filename[0], 'back_propagated')
    elif 'correlation' in options.obs:
        ctype = options.obs.replace('_correlation', '')
        results = correlation_function(options.filename[0],
                                       ctype,
                                       options.elements)
        print_index = True
    else:
        results = None
        print ('Unknown observable')

    print (results.to_string(index=print_index))

if __name__ == '__main__':

    main(sys.argv[1:])
