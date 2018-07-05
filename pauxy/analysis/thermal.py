from pauxy.analysis.extraction import extract_hdf5_data_sets
import pandas as pd
import scipy.stats
import numpy

def analyse_energy(files):
    data = extract_hdf5_data_sets(files)
    sims = []
    for (i, g) in enumerate(data):
        (m, norm, bp, itcf, itcfk, mixed_rdm, bp_rdm) = g
        dt = m.get('qmc').get('dt')
        mu = m.get('trial').get('mu')
        norm['sim'] = i
        norm['mu'] = mu
        norm['dt'] = dt
        sims.append(norm[1:].apply(numpy.real))
    full = pd.concat(sims).groupby(['sim','dt', 'mu'])
    means = full.mean()
    err = full.aggregate(lambda x: scipy.stats.sem(x, ddof=1))
    averaged = means.merge(err, left_index=True, right_index=True,
                           suffixes=('', '_error'))
    return (averaged.reset_index())
