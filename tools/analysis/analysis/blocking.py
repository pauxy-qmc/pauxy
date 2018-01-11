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
import afqmcpy.hubbard

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

def average_single(frame):
    short = frame.drop(['time', 'iteration', 'E_denom', 'E_num'], axis=1)
    short = short.groupby('dt')
    means = short.mean()
    err = short.aggregate(lambda x: scipy.stats.sem(x, ddof=1))
    averaged = means.merge(err, left_index=True, right_index=True,
                           suffixes=('', '_error'))
    columns = sorted(averaged.columns.values)
    averaged.reset_index(inplace=True)
    columns = numpy.insert(columns, 0, 'dt')
    return averaged[columns]

def average_rdm(filename, name, skip=0):
    data = h5py.File(filename, 'r')
    conv = {'mixed': 'mixed_estimates',
            'back_prop': 'back_propagated_estimates'}
    gf = data[conv[name]+'/single_particle_greens_function'][:].real
    gf_av = gf[skip:].mean(axis=0)
    gf_err = gf[skip:].std(axis=0) / len(gf[skip:])**0.5
    return (gf_av, gf_err)

def average_correlation(filename, name, skip=0):
    data = h5py.File(filename, 'r')
    conv = {'mixed': 'mixed_estimates',
            'back_prop': 'back_propagated_estimates'}
    gf = data[conv[name]+'/single_particle_greens_function'][:].real
    nzero = numpy.nonzero(gf)[0][-1]
    gf = gf[skip:nzero]
    ni = numpy.diagonal(gf, axis1=2, axis2=3)
    mg = gf.mean(axis=0)
    # print (sum(ni[0].mean(axis=0)), sum(ni[1].mean(axis=0)), mg.shape,
            # mg[0].trace(), mg[1].trace(), gf[0,0].trace())
    hole = 1.0 - numpy.sum(ni, axis=1)
    hole_err = hole.std(axis=0, ddof=1) / len(hole)**0.5
    spin = 0.5*(ni[:,0,:]-ni[:,1,:])
    spin_err = spin.std(axis=0, ddof=1) / len(hole)**0.5
    return (hole.mean(axis=0), hole_err, spin.mean(axis=0), spin_err, gf)

def plot_correlations(cfunc, cfunc_err, ix, nx, ny, stag=False):
    iy = [i for i in range(ny)]
    idx = [afqmcpy.hubbard.encode_basis(ix,i,nx) for i in iy]
    if stag:
        c = [((-1)**(ix+i))*cfunc[ib] for (i, ib) in zip(iy,idx)]
    else:
        c = [cfunc[ib] for ib in idx]
    err = [cfunc_err[i] for i in idx]
    frame = pd.DataFrame({'iy': iy, 'c': c, 'c_err': err})
    pl.errorbar(iy, c, yerr=err, fmt='o')
    pl.show()
    return frame

def average_tau(frames):

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
    results = pd.DataFrame({'E': energy, 'E_error': energy_err,
                            'Eproj': eproj,
                            'Eproj_error': eproj_err,
                            'weight': weight,
                            'weight_error': weight_error,
                            'numerator': numerator,
                            'numerator_error': numerator_error}).reset_index()

    return results


def analyse_back_propagation(frames):
    frames = frames.groupby(['nbp','dt'])
    data_len = frames.size()
    means = frames.mean().reset_index()
    # calculate standard error of the mean for grouped objects. ddof does
    # default to 1 for scipy but it's different elsewhere, so let's be careful.
    errs = frames.aggregate(lambda x: scipy.stats.sem(x, ddof=1)).reset_index()
    full = pd.merge(means, errs, on=['nbp','dt'], suffixes=('','_error'))
    columns = sorted(full.columns.values[2:])
    columns = numpy.insert(columns, 0, 'nbp')
    columns = numpy.insert(columns, 1, 'dt')
    return full[columns]

def analyse_itcf(itcf):
    means = itcf.mean(axis=(0,1), dtype=numpy.float64)
    n = itcf.shape[0]*itcf.shape[1]
    errs = (
        itcf.std(axis=(0,1), ddof=1, dtype=numpy.float64) / numpy.sqrt(n)
    )
    return (means, errs)

def analyse_estimates(files, start_time=0, multi_sim=False):
    data = analysis.extraction.extract_hdf5_data_sets(files)
    bp_data = []
    norm_data = []
    itcf_data = []
    itcfk_data = []
    mds = []
    nsim = 0
    for g in data:
        (m, norm, bp, itcf, itcfk) = g
        dt = m.get('qmc').get('dt')
        step = m.get('qmc').get('nmeasure')
        norm['dt'] = dt
        norm['iteration'] = numpy.arange(0, step*len(norm), step)
        nzero = numpy.nonzero(norm['Weight'].values)[0][-1]
        start = int(start_time/(step*dt)) + 1
        norm_data.append(norm[start:nzero].apply(numpy.real))
        if bp is not None:
            nbp = m.get('estimators').get('estimators').get('back_prop').get('nmax')
            bp['dt'] = dt
            bp['nbp'] = nbp
            nzero = numpy.nonzero(bp['E'].values)[0][-1]
            skip = max(1, int(start*step/nbp))
            bp_data.append(bp[skip:nzero].apply(numpy.real))
        if itcf is not None:
            itcf_tmax = m.get('estimators').get('estimators').get('itcf').get('tmax')
            nits = int(itcf_tmax/(step*dt)) + 1
            skip = max([1, int(start/nits)])
            nzero = numpy.nonzero(itcf)[0][-1]
            itcf_data.append(itcf[skip:nzero])
        if itcfk is not None:
            itcfk_data.append(itcfk[skip:nzero])
        mds.append(str(m))

    store = h5py.File('analysed_estimates.h5', 'w')
    store.create_dataset('metadata', data=numpy.array(mds, dtype=object),
                         dtype=h5py.special_dtype(vlen=str))
    if itcf is not None:
        itcf_data = numpy.reshape(itcf_data, (len(itcf_data),)+itcf_data[0].shape)
        (itcf_av, itcf_err) = analyse_itcf(itcf_data)
        store.create_dataset('real_itcf', data=itcf_av)
        store.create_dataset('real_itcf_err', data=itcf_err)
    if itcfk is not None:
        itcfk_data = numpy.reshape(itcfk_data, (len(itcf_data),)+itcf_data[0].shape)
        (itcfk_av, itcfk_err) = analyse_itcf(itcfk_data)
        store.create_dataset('kspace_itcf', data=itcfk_av)
        store.create_dataset('kspace_itcf_err', data=itcfk_err)
    if bp is not None:
        bp_data = pd.concat(bp_data)
        bp_av = analyse_back_propagation(bp_data)
        bp_group = store.create_group('back_propagated')
        bp_group.create_dataset('estimates', data=bp_av.as_matrix())
        bp_group.create_dataset('headers', data=bp_av.columns.values,
                dtype=h5py.special_dtype(vlen=str))
    else:
        bp_av = None
    if multi_sim:
        norm_data = pd.concat(norm_data).groupby('iteration')
        norm_av = average_tau(norm_data)
    else:
        norm_data = pd.concat(norm_data)
        norm_av = average_single(norm_data)
    basic = store.create_group('mixed_estimators')
    basic.create_dataset('estimates', data=norm_av.as_matrix())
    basic.create_dataset('headers', data=norm_av.columns.values,
            dtype=h5py.special_dtype(vlen=str))
    store.close()

    return (bp_av, norm_av)
