#!/usr/bin/env python
'''Run a reblocking analysis on pauxy QMC output files.'''

import glob
import h5py
import json
import matplotlib.pyplot as pl
import numpy
import pandas as pd
import pyblock
import scipy.stats
import pauxy.analysis.extraction


def average_single(frame, delete=True):
    short = frame
    means = short.mean().to_frame().T
    err = short.aggregate(lambda x: scipy.stats.sem(x, ddof=1)).to_frame().T
    averaged = means.merge(err, left_index=True, right_index=True,
                           suffixes=('', '_error'))
    columns = [c for c in averaged.columns.values if '_error' not in c]
    columns = [[c, c+'_error'] for c in columns]
    columns = [item for sublist in columns for item in sublist]
    averaged.reset_index(inplace=True)
    delcol = ['ENumer', 'ENumer_error', 'EDenom',
              'EDenom_error', 'Weight', 'Weight_error']
    for d in delcol:
        if delete:
            columns.remove(d)
    return averaged[columns]


def average_ratio(numerator, denominator):
    re_num = numerator.real
    re_den = denominator.real
    im_num = numerator.imag
    im_den = denominator.imag
    # When doing FP we need to compute E = \bar{ENumer} / \bar{EDenom}
    # Only compute real part of the energy
    num_av = (re_num.mean()*re_den.mean()+im_num.mean()*im_den.mean())
    den_av = (re_den.mean()**2 + im_den.mean()**2)
    mean = num_av / den_av
    # Doing error analysis properly is complicated. This is not correct.
    re_nume = scipy.stats.sem(re_num)
    re_dene = scipy.stats.sem(re_den)
    # Ignoring the fact that the mean includes complex components.
    cov = numpy.cov(re_num, re_den)[0,1]
    nsmpl = len(re_num)
    error = abs(mean) * ((re_nume/re_num.mean())**2 +
                         (re_dene/re_den.mean())**2 -
                         2*cov/(nsmpl*re_num.mean()*re_den.mean()))**0.5

    return (mean, error)


def average_fp(frame):
    real = average_single(frame.apply(numpy.real), False)
    imag = average_single(frame.apply(numpy.imag), False)
    results = pd.DataFrame()
    re_num = real.ENumer
    re_den = real.EDenom
    im_num = imag.ENumer
    im_den = imag.EDenom
    # When doing FP we need to compute E = \bar{ENumer} / \bar{EDenom}
    # Only compute real part of the energy
    results['E'] = (re_num*re_den+im_num*im_den) / (re_den**2 + im_den**2)
    # Doing error analysis properly is complicated. This is not correct.
    re_nume = real.E_num_error
    re_dene = real.E_denom_error
    # Ignoring the fact that the mean includes complex components.
    cov = frame.apply(numpy.real).cov()
    cov_nd = cov['ENumer']['EDenom']
    nsmpl = len(frame)
    results['E_error'] = results.E * ((re_nume/re_num)**2 +
                                      (re_dene/re_den)**2 -
                                      2*cov_nd/(nsmpl*re_num*re_den))**0.5
    return results


def reblock_mixed(frame):
    short = frame.drop(['Time', 'EDenom', 'ENumer', 'Weight'], axis=1)
    analysed = []
    (data_len, blocked_data, covariance) = pyblock.pd_utils.reblock(short)
    reblocked = pd.DataFrame()
    for c in short.columns:
        try:
            rb = pyblock.pd_utils.reblock_summary(blocked_data.loc[:,c])
            reblocked[c] = rb['mean'].values
            reblocked[c+'_error'] = rb['standard error'].values
        except KeyError:
            print("Reblocking of {:4} failed. Insufficient "
                  "statistics.".format(c))
    analysed.append(reblocked)

    return pd.concat(analysed)


def reblock_free_projection(frame):
    short = frame.drop(['Time', 'Weight', 'ETotal'], axis=1)
    analysed = []
    (data_len, blocked_data, covariance) = pyblock.pd_utils.reblock(short)
    reblocked = pd.DataFrame()
    denom = blocked_data.loc[:,'EDenom']
    for c in short.columns:
        if c != 'EDenom':
            nume = blocked_data.loc[:,c]
            cov = covariance.xs('EDenom', level=1)[c]
            ratio = pyblock.error.ratio(nume, denom, cov, data_len)
            rb = pyblock.pd_utils.reblock_summary(ratio)
            try:
                if c == 'ENumer':
                    c = 'ETotal'
                reblocked[c] = rb['mean'].values
                reblocked[c+'_error'] = rb['standard error'].values
            except KeyError:
                print("Reblocking of {:4} failed. Insufficient "
                      "statistics.".format(c))
    analysed.append(reblocked)

    if len(analysed) == 0:
        return None
    else:
        return pd.concat(analysed)


def reblock_local_energy(filename, skip=0):
    data = pauxy.analysis.extraction.extract_mixed_estimates(filename)
    results = reblock_mixed(data.apply(numpy.real)[skip:])
    if results is None:
        return None
    else:
        try:
            energy = results['ETotal'].values[0]
            error = results['ETotal_error'].values[0]
            return (energy, error)
        except KeyError:
            return None


def reblock_rdm(files, skip=1, est_type='back_propagated',
                free_projection=False, rdm_type='one_rdm'):

    rdm_series, weights = pauxy.analysis.extraction.extract_rdm(files, skip,
                                                                est_type=est_type,
                                                                rdm_type=rdm_type)

    if not free_projection:
        rdm_series = rdm_series / weights
    else:
        print("Analysis for FP RDM not implemented.")

    rdm, rdm_err = average_rdm(rdm_series)
    return rdm, rdm_err


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
    covs = frames.cov().loc[:,'ENumer'].loc[:, 'EDenom']
    energy = means['ENumer'] / means['EDenom']
    sqrtn = numpy.sqrt(data_len)
    energy_err = ((err['ENumer']/means['ENumer'])**2.0 +
                  (err['EDenom']/means['EDenom'])**2.0 -
                  2*covs/(means['ENumer']*means['EDenom']))**0.5

    energy_err = abs(energy/sqrtn) * energy_err
    eproj = means['ETotal']
    eproj_err = err['ETotal']/numpy.sqrt(data_len)
    weight = means['Weight']
    weight_error = err['Weight']
    numerator = means['ENumer']
    numerator_error = err['ENumer']
    results = pd.DataFrame({'ETotal': energy, 'ETotal_error': energy_err,
                            'Eproj': eproj,
                            'Eproj_error': eproj_err,
                            'weight': weight,
                            'weight_error': weight_error,
                            'numerator': numerator,
                            'numerator_error': numerator_error}).reset_index()

    return results


def analyse_back_propagation(frames):
    frames[['E', 'E1b', 'E2b']] = frames[['E','E1b','E2b']].div(frames.Weight, axis=0)
    frames = frames.apply(numpy.real)
    frames = frames.groupby(['nbp','dt'])
    data_len = frames.size()
    means = frames.mean().reset_index()
    # calculate standard error of the mean for grouped objects. ddof does
    # default to 1 for scipy but it's different elsewhere, so let's be careful.
    errs = frames.aggregate(lambda x: scipy.stats.sem(x, ddof=1)).reset_index()
    full = pd.merge(means, errs, on=['nbp','dt'], suffixes=('','_error'))
    columns = full.columns.values[2:]
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


def analyse_simple(files, start_time):
    data = pauxy.analysis.extraction.extract_hdf5_data_sets(files)
    norm_data = []
    for (g, f) in zip(data, files):
        (m, norm, bp, itcf, itcfk, mixed_rdm, bp_rdm) = g
        dt = m.get('qmc').get('dt')
        free_projection = m.get('propagators').get('free_projection')
        step = m.get('qmc').get('nmeasure')
        read_rs = m.get('psi').get('read_file') is not None
        nzero = numpy.nonzero(norm['Weight'].values)[0][-1]
        start = int(start_time/(step*dt)) + 1
        if read_rs:
            start = 0
        if free_projection:
            reblocked = average_fp(norm[start:nzero])
        else:
            reblocked = reblock_mixed(norm[start:nzero].apply(numpy.real))
            columns = pauxy.analysis.extraction.set_info(reblocked, m)
        norm_data.append(reblocked)
    return pd.concat(norm_data)


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
        ndets = m.get('trial').get('ndets')
        write_rs = m.get('psi').get('write_restart')
        read_rs = m.get('psi').get('read_file') is not None
        free_projection = m.get('propagators').get('free_projection', False)
        nzero = numpy.nonzero(norm['Weight'].values)[0][-1]
        start = int(start_time/(step*dt)) + 1
        if read_rs:
            start = 0
        norm_data.append(norm[start:nzero].apply(numpy.real))
        if mixed_rdm is not None:
            mrdm, mrdm_err = average_rdm(mixed_rdm[start:nzero])
            if cfunc:
                (m_hole, m_hole_err, m_spin, m_spin_err, m_gf) = average_correlation(mixed_rdm[start:nzero])
        if bp is not None:
            nbp = m.get('estimators').get('estimators').get('back_prop').get('nmax')
            bp['dt'] = dt
            bp['nbp'] = nbp
            weights = bp['Weight'].values.real
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

    base = files[0].split('/')[-1]
    outfile = 'analysed_' + base
    store = h5py.File(outfile, 'w')
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
        bp_group.create_dataset('estimates', data=bp_av.values)
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
        norm_data = pd.concat(norm_data).groupby('Iteration')
        norm_av = average_tau(norm_data)
    else:
        norm_data = pd.concat(norm_data)
        if free_projection:
            norm_av = reblock_free_projection(norm_data)
        else:
            norm_av = reblock_mixed(norm_data)
    basic = store.create_group('mixed')
    basic.create_dataset('estimates', data=norm_av.values.astype(float))
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

    fmt = lambda x: "{:13.8f}".format(x)
    print(norm_av.to_string(float_format=fmt))
    return (bp_av, norm_av)
