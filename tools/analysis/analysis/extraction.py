import pandas as pd
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
        for line in fhandle:
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
        return (json.loads(json_text), ln)
    else:
        return ({}, 0)


def extract_data(filename):

    with open(filename) as f:
        (metadata, skip) = _extract_json(f, True)
    data = pd.read_csv(filename, skiprows=skip+2, sep=r'\s+', comment='#')

    return (metadata, data)
