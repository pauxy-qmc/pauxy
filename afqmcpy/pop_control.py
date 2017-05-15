import numpy
import copy

def comb(psi, nw):
    new_psi = nw*[0]
    new_ovlps = numpy.zeros(nw, dtype=type(psi[0].ot))
    weights = [w.weight for w in psi]
    total_weight = sum(weights)
    cprobs = numpy.cumsum(weights)

    # Apply the comb method of population control / branching.
    # See Booth & Gubernatis PRE 80, 046704 (2009).
    r = numpy.random.random()
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
