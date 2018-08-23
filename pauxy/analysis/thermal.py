from pauxy.analysis.extraction import extract_hdf5_data_sets, set_info
from pauxy.analysis.blocking import average_ratio
import pandas as pd
import scipy.stats
import numpy

def analyse_energy(files):
    data = extract_hdf5_data_sets(files)
    sims = []
    for (i, g) in enumerate(data):
        (m, norm, bp, itcf, itcfk, mixed_rdm, bp_rdm) = g
        free_projection = m.get('propagators').get('free_projection', False)
        set_info(norm, m)
        sims.append(norm[1:])
    full = pd.concat(sims).groupby(['dt', 'mu', 'beta', 'free_projection', 'rs', 'ecut', 'nup', 'ndown', 'nwalkers'])
    analysed = []
    for (i, g) in full:
        if free_projection:
            cols = ['E_num', 'Nav']
            obs = ['E', 'Nav']
            averaged = pd.DataFrame(index=[0])
            for (c, o) in zip(cols, obs):
                (value, error)  = average_ratio(g[c].values, g['E_denom'].values)
                averaged[o] = [value]
                averaged[o+'_error'] = [error]
            for (k, v) in zip(full.keys, i):
                averaged[k] = v
            analysed.append(averaged)
        else:
            real_g = g.apply(numpy.real)
            means = real_g.mean().to_frame().T
            err = real_g.aggregate(lambda x: scipy.stats.sem(x, ddof=1)).to_frame().T
            averaged = means.merge(err, left_index=True, right_index=True,
                                   suffixes=('', '_error'))
            analysed.append(averaged)
    return (pd.concat(analysed).reset_index())
