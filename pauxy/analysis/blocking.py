#!/usr/bin/env python
'''Run a reblocking analysis on pauxy QMC output files.'''

import pandas as pd
import numpy
import scipy.stats
import pauxy.analysis.extraction
import matplotlib.pyplot as pl
import h5py
import json

def average_single(frame):
    short = frame.drop(['time', 'iteration', 'E_denom', 'E_num', 'Weight'], axis=1)
    short = short.groupby(['dt','ndets'])
    means = short.mean()
    err = short.aggregate(lambda x: scipy.stats.sem(x, ddof=1))
    averaged = means.merge(err, left_index=True, right_index=True,
                           suffixes=('', '_error'))
    columns = [c for c in averaged.columns.values if '_error' not in c]
    columns = [[c, c+'_error'] for c in columns]
    columns = [item for sublist in columns for item in sublist]
    averaged.reset_index(inplace=True)
    columns = numpy.insert(columns, 0, 'dt')
    columns = numpy.insert(columns, 0, 'ndets')
    return averaged[columns]

def average_rdm(gf):
    gf_av = gf.mean(axis=0)
    gf_err = gf.std(axis=0) / len(gf)**0.5
    return (gf_av, gf_err)

def average_correlation(gf):
    ni = numpy.diagonal(gf, axis1=2, axis2=3)
    mg = gf.mean(axis=0)
    hole = 1.0 - numpy.sum(ni, axis=1)
    hole_err = hole.std(axis=0, ddof=1) / len(hole)**0.5
    spin = 0.5*(ni[:,0,:]-ni[:,1,:])
    spin_err = spin.std(axis=0, ddof=1) / len(hole)**0.5
    return (hole.mean(axis=0), hole_err, spin.mean(axis=0), spin_err, gf)

def plot_correlations(cfunc, cfunc_err, ix, nx, ny, stag=False):
    c, err = get_strip(cfunc, cfunc_err, ix, nx, ny, stag)
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
    frames[['E', 'T', 'V']] = frames[['E','T','V']].div(frames.weight, axis=0)
    frames = frames.apply(numpy.real)
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

def analyse_estimates(files, start_time=0, multi_sim=False, cfunc=False):
    data = pauxy.analysis.extraction.extract_hdf5_data_sets(files)
    bp_data = []
    bp_rdms = []
    norm_data = []
    itcf_data = []
    itcfk_data = []
    mds = []
    nsim = 0
    for g in data:
        (m, norm, bp, itcf, itcfk, mixed_rdm, bp_rdm) = g
        dt = m.get('qmc').get('dt')
        step = m.get('qmc').get('nmeasure')
        norm['dt'] = dt
        norm['iteration'] = numpy.arange(0, step*len(norm), step)
        ndets = m.get('trial').get('ndets')
        if ndets is not None:
            norm['ndets'] = ndets
        else:
            norm['ndets'] = 1
        nzero = numpy.nonzero(norm['Weight'].values)[0][-1]
        start = int(start_time/(step*dt)) + 1
        norm_data.append(norm[start:nzero].apply(numpy.real))
        if mixed_rdm is not None:
            mrdm, mrdm_err = average_rdm(mixed_rdm[start:nzero])
            if cfunc:
                (m_hole, m_hole_err, m_spin, m_spin_err, m_gf) = average_correlation(mixed_rdm[start:nzero])
        if bp is not None:
            nbp = m.get('estimators').get('estimators').get('back_prop').get('nmax')
            bp['dt'] = dt
            bp['nbp'] = nbp
            weights = bp['weight'].values.real
            nzero = numpy.nonzero(bp['E'].values)[0][-1]
            skip = max(1, int(start*step/nbp))
            bp_data.append(bp[skip:nzero:2])
            if bp_rdm is not None:
                if len(bp_rdm.shape) == 3:
                    # GHF format
                    w = weights[skip:nzero,None,None]
                else:
                    # UHF format
                    w = weights[skip:nzero,None,None,None]
                bp_rdm = bp_rdm[skip:nzero] / w
                rdm, rdm_err = average_rdm(bp_rdm[skip:nzero])
                bp_rdms.append(numpy.array([rdm,rdm_err]))
                if cfunc:
                    (bp_hole, bp_hole_err, bp_spin, bp_spin_err, bp_gf) = average_correlation(bp_rdm[skip:nzero])
                # free projection / weight restoration..
        if itcf is not None:
            itcf_tmax = m.get('estimators').get('estimators').get('itcf').get('tmax')
            nits = int(itcf_tmax/(step*dt)) + 1
            skip = max([1, int(start/nits)])
            nzero = numpy.nonzero(itcf)[0][-1]
            itcf_data.append(itcf[skip:nzero])
        if itcfk is not None:
            itcfk_data.append(itcfk[skip:nzero])
        mds.append(json.dumps(m))

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
        if bp_rdm is not None:
            bp_group.create_dataset('rdm', data=numpy.array(bp_rdms))
            if cfunc:
                bp_group.create_dataset('correlation',
                                        data=numpy.array([bp_hole, bp_hole_err,
                                                          bp_spin, bp_spin_err]))
    else:
        bp_av = None
    if multi_sim:
        norm_data = pd.concat(norm_data).groupby('iteration')
        norm_av = average_tau(norm_data)
    else:
        norm_data = pd.concat(norm_data)
        norm_av = average_single(norm_data)
    basic = store.create_group('mixed')
    basic.create_dataset('estimates', data=norm_av.as_matrix())
    basic.create_dataset('headers', data=norm_av.columns.values,
            dtype=h5py.special_dtype(vlen=str))
    if mixed_rdm is not None:
        basic.create_dataset('rdm',
                             data=numpy.array([mrdm, mrdm_err]))
        if cfunc:
            basic.create_dataset('correlation',
                                 data=numpy.array([m_hole, m_hole_err, m_spin,
                                                   m_spin_err]))
    store.close()

    return (bp_av, norm_av)
