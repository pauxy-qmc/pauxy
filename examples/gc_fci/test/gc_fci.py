import pyhande
import glob
from uegpy import utils
import numpy
import sys
import matplotlib.pyplot as pl
import scipy.optimize
import pandas as pd

files = glob.glob('*.out')

data = pyhande.extract.extract_data_sets(files)

eigs = []
eigs.append([0, 0])
for d in data:
    m = d[0]
    nel = m.get('system').get('nel')
    e = d[1]
    eigs.append([nel, e])

def nav(mu, data, beta):
    total = 0
    Z = 0
    for d in data:
        N = d[0]
        eigs = d[1]
        boltzmann = numpy.exp(-beta*(eigs-mu*N))
        total += N * numpy.sum(boltzmann)
        Z += numpy.sum(boltzmann)
    return total / Z

def energy(data, beta, mu):
    num = 0
    den = 0
    for d in data:
        N = d[0]
        eigs = numpy.array(d[1])
        boltzmann = numpy.exp(-beta*(eigs-mu*N))
        num += eigs.dot(boltzmann)
        den += numpy.sum(boltzmann)
    return num / den

# mus = numpy.linspace(1.37,1.44,20)
# beta = 1.0 / utils.calcT(1, 0.5, 0)

# mus = numpy.linspace(1.8,2.0,20)
# beta = 1.0 / utils.calcT(1, 0.25, 0)

mus = numpy.linspace(0.0,0.2,20)
beta = 1.0 / utils.calcT(1, 0.0625, 0)

nelec = 2
navs = [nav(m, eigs, beta) for m in mus]
energies = [energy(eigs, beta, m) for m in mus]

results = pd.DataFrame({'mu':mus, 'nav':navs, 'energy':energies})
print(results)

# pl.plot(mus, navs)
# pl.show()

#def delta(mu, eigs, beta, nel):
#    return nav(mu, eigs, beta) - nel
#
#def find_mu(eigs, beta, nel):
#    mu = scipy.optimize.newton(delta, 0, args=(eigs, beta, nel))
#    return mu
#
#rs = 1
#nelec = 2
#thetas = [2**n for n in range(-4, 4)]
#betas = [1.0 / utils.calcT(rs, t, 0) for t in thetas]
#mus = [find_mu(eigs, b, nelec) for b in betas]
#energies = [energy(eigs, b, m) for (b, m) in zip(betas, mus)]
#
#results = pd.DataFrame({'beta': betas, 'theta': thetas,
#                        'beta': betas, 'mu': mus, 'energy': energies},
#                        columns=['beta', 'theta', 'mu', 'energy'])
#
#results['nav'] = 2
#results['rs'] = 1
#
#print (results.to_string())
