import numpy as np
import random

def comb(psi, nw):
    new_psi = [ ]
    new_ovlps = np.zeros(nw)
    weights = [w.weight for w in psi]
    print weights
    total_weight = sum(weights)
    cprobs = np.cumsum(weights)
    print cprobs

    # Apply the comb method of population control / branching.
    # See Booth & Gubernatis PRE 80, 046704 (2009).
    r = random.random()
    comb = [(i+r) * (total_weight/nw) for i in range(nw)]
    # print comb
    # print r
    for (ic, c) in enumerate(comb):
        for (iw, w) in enumerate(weights):
            if c < w:
                new_psi.append(psi[iw].phi)
                new_ovlps[ic] = psi[iw].ot

    # Copy back new information
    for i in range(nw):
        psi[i].phi = new_psi[i]
        psi[i].ot = new_ovlps[i]
        psi[i].weight = 1.0
