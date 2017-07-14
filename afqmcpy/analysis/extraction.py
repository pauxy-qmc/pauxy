import pandas as pd
import numpy
import json

def _extract_json(fhandle, find_start=False, max_end=None):
    '''Extract JSON output from a AFQMCPY output file.

Parameters
----------
fhandle : file
    File handle to a AFQMCPY output file.
find_start : boolean
    If true, search for the start of the JSON block.  If false (default), then
    the file is assumed to be opened at the start of the JSON block.
max_end : string
    If find_start is True and max_end is not None, the search for the JSON block
    is aborted if a line containing max_end is found, in which case an empty
    dict is returned.

.. note::

    AFQMCPY output contains blocks of output in JSON format.  The start of such
    a block is denoted by a line containing the string 'Start JSON block' and
    then end by a line containing the string 'End JSON block'.

Returns
-------
json_dict : dict
    JSON output loaded into a dictionary.


Stolen from HANDE source code.
'''

    found_json = True
    if find_start:
        for (sl, line) in enumerate(fhandle):
            if 'Input options' in line:
                break
            elif max_end is not None and max_end in line:
                found_json = False
                break
    json_text = ''
    if found_json:
        for (ln, line) in enumerate(fhandle):
            if 'End of input options' in line:
                break
            else:
                json_text += line
    if json_text:
        return (json.loads(json_text), sl+ln+1)
    else:
        return ({}, 0)


def extract_data_sets(files, itcf=False):

    data =  [extract_data(f, itcf) for f in files]

    return data

def extract_data(filename, itcf=False):

    with open(filename) as f:
        (metadata, skip) = _extract_json(f, True)
    if itcf:
        model = metadata['model']
        opts = metadata['qmc_options']
        if opts['itcf']['mode'] ==  'full':
            dimg = int(model['nx']*model['ny'])**2
        else:
            dimg = len(numpy.array(opts['itcf']['mode'][0]))
        nitcf = int(opts['itcf_tmax']/opts['dt']) + 1
        data = numpy.loadtxt(filename, skiprows=skip)
        nav = int(len(data.flatten())/(dimg*nitcf))
        if opts['itcf']['mode'] ==  'full':
            data = data.reshape((nav*nitcf, dimg, dimg))
        else:
            data = data.reshape((nav*nitcf, dimg))
    else:
        data = pd.read_csv(filename, skiprows=skip, sep=r'\s+', comment='#')

    return (metadata, data)

def pretty_table(summary, metadata):

    vals = summary.ix['Energy',:]
    model = metadata['model']
    table = pd.DataFrame({'model': model['name'],
                          'lattice': r'%sX%s'%(model['nx'],model['ny']),
                          'filling': r'(%s,%s)'%(model['nup'], model['ndown']),
                          'U': model['U'],
                          'E': vals['mean'],
                          'E_error': vals['standard error'],
                          'E_error_error': vals['standard error error']},
                          index=[0])

    return table


def pretty_table_loop(results, model):

    columns = ['tau', 'E', 'E_error', 'model', 'lattice', 'filling', 'U']
    table = pd.DataFrame({'tau': results['tau'],
                          'E': results['E'],
                          'E_error': results['E_error'],
                          'model': model['name'],
                          'lattice': r'%sX%s'%(model['nx'],model['ny']),
                          'filling': r'(%s,%s)'%(model['nup'], model['ndown']),
                          'U': model['U']}, columns=columns)

    return table
