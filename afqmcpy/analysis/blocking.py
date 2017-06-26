#!/usr/bin/env python
'''Run a reblocking analysis on AFQMCPY QMC output files. Heavily adapted from
HANDE'''

import pandas as pd
import pyblock
import numpy
import scipy.stats
import analysis.extraction


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
    data_len = frames.size()
    means = frames.mean()
    err = numpy.sqrt(frames.var())
    covs = frames.cov().loc[:,'E_num'].loc[:, 'E_denom']
    energy = means['E_num'] / means['E_denom']
    energy_err = abs(energy/numpy.sqrt(data_len))*((err['E_num']/means['E_num'])**2.0 +
                                   (err['E_denom']/means['E_denom'])**2.0 -
                                   2*covs/(means['E_num']*means['E_denom']))**0.5

    tau = m['qmc_options']['dt']
    nsites = m['model']['nx']*m['model']['ny']
    results = pd.DataFrame({'E': energy/nsites, 'E_error': energy_err/nsites}).reset_index()
    results['tau'] = results['iteration'] * tau

    return analysis.extraction.pretty_table_loop(results, m['model'])


def average_itcf(filenames, element, start_iteration=0):

    data = analysis.extraction.extract_data_sets(filenames, itcf=True)
    frames = []
    # I'm pretty sure there's a faster way of doing this.


def average_back_propagated(filenames, start_iteration=0):

    data = analysis.extraction.extract_data_sets(filenames)
    frames = []

    # I'm pretty sure there's a faster way of doing this.
    for (m,d) in data:
        d['nbp'] = m['qmc_options']['nback_prop']
        frames.append(d.loc[:,'E':][start_iteration:])

    frames = pd.concat(frames).groupby('nbp')
    data_len = frames.size()
    means = frames.mean().reset_index()
    # calculate standard error of the mean for grouped objects. ddof does
    # default to 1 for scipy but it's different elsewhere, so let's be careful.
    errs = frames.aggregate(lambda x: scipy.stats.sem(x, ddof=1)).reset_index()
    full = pd.merge(means, errs, on='nbp', suffixes=('','_error'))
    return full
