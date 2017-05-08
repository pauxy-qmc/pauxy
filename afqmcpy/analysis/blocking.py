#!/usr/bin/env python
'''Run a reblocking analysis on AFQMCPY QMC output files. Heavily adapted from
HANDE'''

import pandas as pd
import pyblock
import analysis.extraction
import numpy


def run_blocking_analysis(filename, start_iter):
    '''
'''

    (metadata, data) = analysis.extraction.extract_data(filename[0])
    (data_len, reblock, covariances) = pyblock.pd_utils.reblock(data.drop(['iteration',
                                                                           'time',
                                                                           'exp(delta)'],
                                                                           axis=1))
    cov = covariances.xs('Weight', level=1)['E_num']
    numerator = reblock.ix[:,'E_num']
    denominator = reblock.ix[:,'Weight']
    projected_energy = pyblock.error.ratio(numerator, denominator, cov, 4)
    projected_energy.columns = pd.MultiIndex.from_tuples([('Energy', col)
                                    for col in projected_energy.columns])
    reblock = pd.concat([reblock, projected_energy], axis=1)
    summary = pyblock.pd_utils.reblock_summary(reblock)
    useful_table = analysis.extraction.pretty_table(summary, metadata)

    return (reblock, useful_table)


def average_tau(filenames):

    data = analysis.extraction.extract_data_sets(filenames)
    frames = []

    for (m,d) in data:
        frames.append(d)

    frames = pd.concat(frames).groupby('iteration')
    means = frames.mean()
    err = frames.var()
    covs = frames.cov()
    energy = means['E_num'] / means['Weight']
    energy_err = energy*numpy.sqrt((err['E_num']/means['E_num'])**2.0 +
                                   (err['Weight']/means['Weight'])**2.0)
                                   # 2*covs[['E_num','Weight']]/(means['E_num']*means['Weight']))
    tau = m['qmc_options']['dt']
    results = pd.DataFrame({'E': energy, 'E_error': energy_err}).reset_index()
    results['iteration'] = results['iteration'] * tau

    return results
