import numpy
import pandas as pd
import scipy.stats
import scipy.optimize
from pauxy.analysis.extraction import extract_data, set_info, get_metadata
from pauxy.analysis.blocking import average_ratio

def analyse_energy(files):
    sims = []
    for f in files:
        data = extract_data(f, 'basic', 'energies')
        md = get_metadata(f)
        keys = set_info(data, md)
        sims.append(data[1:])
    full = pd.concat(sims).groupby(keys)
    analysed = []
    for (i, g) in full:
        if g['free_projection'].values[0]:
            cols = ['ENumer', 'Nav']
            obs = ['ETotal', 'Nav']
            averaged = pd.DataFrame(index=[0])
            for (c, o) in zip(cols, obs):
                (value, error)  = average_ratio(g[c].values, g['EDenom'].values)
                averaged[o] = [value]
                averaged[o+'_error'] = [error]
            for (k, v) in zip(full.keys, i):
                averaged[k] = v
            analysed.append(averaged)
        else:
            cols = ['ETotal', 'E1Body', 'E2Body', 'Nav']
            averaged = pd.DataFrame(index=[0])
            for c in cols:
                mean = numpy.real(g[c].values).mean()
                error = scipy.stats.sem(numpy.real(g[c].values), ddof=1)
                averaged[c] = [mean]
                averaged[c+'_error'] = [error]
            for (k, v) in zip(full.keys, i):
                averaged[k] = v
            analysed.append(averaged)
    return (pd.concat(analysed).reset_index(drop=True))

def nav_mu(mu, coeffs):
    return numpy.polyval(coeffs, mu)

def find_chem_pot(data, target, vol, order=3, plot=False):
    print("# System volume: {}.".format(vol))
    print("# Target number of electrons: {}.".format(vol*target))
    nav = data.Nav.values / vol
    nav_error = data.Nav_error.values / vol
    # Half filling special case where error bar is zero.
    zeros = numpy.where(nav_error==0)[0]
    nav_error[zeros] = 1e-8
    mus = data.mu.values
    delta = nav - target
    fit = numpy.polyfit(mus, delta, order, w=1.0/nav_error)
    a = min(mus)
    b = max(mus)
    try:
        mu, r = scipy.optimize.brentq(nav_mu, a, b, args=fit, full_output=True)
    except ValueError:
        mu = None
        print("Root not found in interval.")

    if plot:
        import matplotlib.pyplot as pl
        beta = data.beta[0]
        pl.errorbar(mus, delta, yerr=nav_error, fmt='o',
                    label=r'$\beta = {}$'.format(beta), color='C0')
        xs = numpy.linspace(a,b,101)
        ys = nav_mu(xs, fit)
        pl.plot(xs,ys,':', color='C0')
        if mu is not None and r.converged:
            pl.axvline(mu, linestyle=':', label=r'$\mu^* = {}$'.format(mu),
                       color='C3')
        pl.xlabel(r"$\mu$")
        pl.ylabel(r"$n-n_{\mathrm{av}}$")
        pl.legend(numpoints=1)
        pl.show()
    if mu is not None:
        if r.converged:
            return mu
        else:
            return None
