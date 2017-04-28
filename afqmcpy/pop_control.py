import numpy as np
import random

def comb(psi, nw):
    new_psi = 100*[0]
    new_ovlps = np.zeros(nw)
    weights = [w.weight for w in psi]
    total_weight = sum(weights)
    cprobs = np.cumsum(weights)

    # Apply the comb method of population control / branching.
    # See Booth & Gubernatis PRE 80, 046704 (2009).
    r = random.random()
    comb = [(i+r) * (total_weight/nw) for i in range(nw)]
    # print comb
    # print cprobs
    for (ic, c) in enumerate(comb):
        for (iw, w) in enumerate(cprobs):
            # print iw, w, c
            if c < w:
                print iw, w, ic, c
                new_psi[ic] = psi[iw].phi
                new_ovlps[ic] = psi[iw].ot
                break
        # print ic, new_ovlps[ic], len(comb)

    # Copy back new information
    for i in range(nw):
        psi[i].phi = new_psi[i]
        psi[i].ot = new_ovlps[i]
        psi[i].weight = 1.0
