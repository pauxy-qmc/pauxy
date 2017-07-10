import numpy
import copy

def comb(psi, nw, psi_history=None):
    new_psi = copy.deepcopy(psi)
    weights = [w.weight for w in psi]
    parent_ix = numpy.arange(len(psi))
    parent_link = numpy.arange(len(psi))
    total_weight = sum(weights)
    cprobs = numpy.cumsum(weights)

    # Apply the comb method of population control / branching.
    # See Booth & Gubernatis PRE 80, 046704 (2009).
    r = numpy.random.random()
    comb = [(i+r) * (total_weight/nw) for i in range(nw)]
    for (ic, c) in enumerate(comb):
        for (iw, w) in enumerate(cprobs):
            if c < w:
                parent_ix[ic] = iw
                break

    # Copy back new information
    for (i,p) in enumerate(parent_ix):
        psi[i] = copy.deepcopy(new_psi[p])
        psi[i].weight = 1.0
        if psi_history is not None:
            psi_history[i,:] = copy.deepcopy(psi_history[p,:])
