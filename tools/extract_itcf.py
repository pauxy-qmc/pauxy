#!/usr/bin/env python
'''Exctact element of green's function'''

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
    parser.add_argument('-s', '--spin', type=str, dest='spin',
                        default=None, help='Spin component to extract.'
                        'Options: up/down')
    parser.add_argument('-o', '--order', type=str, dest='order',
                        default=None, help='Type of green\'s function to extract.'
                        'Options: lesser/greater')
    parser.add_argument('-k', '--kspace', dest='kspace', action='store_true',
                        default=False, help='Extract kspace green\'s function.')
    parser.add_argument('-e', '--elements',
                        type=lambda s: [int(item) for item in s.split(',')],
                        dest='elements', default=None,
                        help='Element to extract.')
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
    results = analysis.extraction.extract_analysed_itcf(options.filename[0],
                                                        options.elements,
                                                        options.spin,
                                                        options.order,
                                                        options.kspace)
    print (results.to_string(index=False))

if __name__ == '__main__':

    main(sys.argv[1:])
