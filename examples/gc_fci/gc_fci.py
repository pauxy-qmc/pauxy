import pyhande
import glob
from uegpy import utils
import numpy
import sys
import matplotlib.pyplot as pl
import scipy.optimize
import pandas as pd
from uegpy.utils import calcT

files = glob.glob('ueg.hande.fci.out')

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

# test: scan over mu
# mus = numpy.linspace(0.12,0.14,100)
# beta = 1.0 / utils.calcT(1, 1, 0)
# nelec = 2
# navs = [nav(m, eigs, beta) for m in mus]
# pl.plot(mus, navs)
# pl.show()

def delta(mu, eigs, beta, nel):
    return nav(mu, eigs, beta) - nel

def find_mu(eigs, beta, nel):
    mu = scipy.optimize.newton(delta, 0, args=(eigs, beta, nel))
    return mu

rs = 1
nelec = 2
mus = numpy.linspace(0, 2, 11)
navs = [nav(m, eigs, 1.0/calcT(rs, 0.5, 0)) for m in mus]
energies = [energy(eigs, 1.0/calcT(rs, 0.5, 0), m) for m in mus]

results = pd.DataFrame({'mu': mus, 'energy': energies, 'nav': navs},
                        columns=['mu', 'energy', 'nav'])

print (results.to_string())
