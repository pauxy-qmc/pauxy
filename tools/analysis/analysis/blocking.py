#!/usr/bin/env python
'''Run a reblocking analysis on AFQMCPY QMC output files. Heavily adapted from
HANDE'''

import pandas as pd
import pyblock
import numpy
import scipy.stats
import analysis.extraction
import matplotlib.pyplot as pl
import h5py

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

    eproj = means['E']
    eproj_err = err['E']/numpy.sqrt(data_len)
    weight = means['Weight']
    weight_error = err['Weight']
    numerator = means['E_num']
    numerator_error = err['E_num']
    tau = m['qmc_options']['dt']
    nsites = m['model']['nx']*m['model']['ny']
    results = pd.DataFrame({'E': energy/nsites, 'E_error': energy_err/nsites,
                            'Eproj': eproj/nsites,
                            'Eproj_error': eproj_err/nsites,
                            'weight': weight,
                            'weight_error': weight_error,
                            'numerator': numerator,
                            'numerator_error': numerator_error}).reset_index()
    results['tau'] = results['iteration'] * tau

    return analysis.extraction.pretty_table_loop(results, m['model'])


def analyse_back_propagation(frames):
    frames = frames.groupby('nbp')
    data_len = frames.size()
    means = frames.mean().reset_index()
    # calculate standard error of the mean for grouped objects. ddof does
    # default to 1 for scipy but it's different elsewhere, so let's be careful.
    errs = frames.aggregate(lambda x: scipy.stats.sem(x, ddof=1)).reset_index()
    full = pd.merge(means, errs, on='nbp', suffixes=('','_error'))
    full.columns.values[1:].sort()
    return full

def analyse_itcf(itcf):
    means = itcf.mean(axis=(0,1), dtype=numpy.float64)
    n = itcf.shape[0]*itcf.shape[1]
    errs = (
        itcf.std(axis=(0,1), ddof=1, dtype=numpy.float64) / numpy.sqrt(n)
    )
    return (means, errs)

def analyse_estimates(filenames, start_iteration=0, skip=0): 
    data = analysis.extraction.extract_hdf5_data_sets(filenames)
    bp_data = []
    itcf_data = []
    itcfk_data = []
    mds = []
    for g in data:
        (m, bp, itcf, itcfk) = g
        bp['nbp'] = m.get('estimates').get('back_propagation').get('nback_prop')
        bp_data.append(bp[start_iteration:])
        itcf_data.append(itcf[skip:])
        itcfk_data.append(itcfk[skip:])
        mds.append(str(m))

    bp_data = pd.concat(bp_data)
    itcf_data = numpy.reshape(itcf_data, (len(itcf_data),)+itcf_data[0].shape)
    itcfk_data = numpy.reshape(itcf_data, (len(itcf_data),)+itcf_data[0].shape)
    bp_av = analyse_back_propagation(bp_data)
    (itcf_av, itcf_err) = analyse_itcf(itcf_data)
    (itcfk_av, itcfk_err) = analyse_itcf(itcf_data)
    store = h5py.File('analysed_estimates.h5', 'w')
    store.create_dataset('metadata', data=numpy.array(mds, dtype=object),
                         dtype=h5py.special_dtype(vlen=str))
    store.create_dataset('real_itcf', data=itcf_av, dtype=float)
    store.create_dataset('real_itcf_err', data=itcf_err, dtype=float)
    store.create_dataset('kspace_itcf', data=itcfk_err, dtype=float)
    store.create_dataset('kspace_itcf_err', data=itcfk_err, dtype=float)
    store.close()

    return bp_av
