import numpy as np
import random
import copy

def comb(psi, nw):
    new_psi = nw*[0]
    new_ovlps = np.zeros(nw)
    weights = [w.weight for w in psi]
    total_weight = sum(weights)
    cprobs = np.cumsum(weights)

    # Apply the comb method of population control / branching.
    # See Booth & Gubernatis PRE 80, 046704 (2009).
    r = random.random()
    comb = [(i+r) * (total_weight/nw) for i in range(nw)]
    for (ic, c) in enumerate(comb):
        for (iw, w) in enumerate(cprobs):
            if c < w:
                new_psi[ic] = copy.copy(psi[iw].phi)
                new_ovlps[ic] = psi[iw].ot
                break

    # Copy back new information
    for i in range(nw):
        psi[i].phi = new_psi[i]
        psi[i].ot = new_ovlps[i]
        psi[i].weight = 1.0
