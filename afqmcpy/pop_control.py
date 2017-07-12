import numpy
import copy

def comb(psi, nw, psi_history=None):
    # Need make a copy to since the elements in psi are only references to
    # walker objects in memory. We don't want future changes in a given element
    # of psi having unintended consequences.
    new_psi = copy.deepcopy(psi)
    weights = [w.weight for w in psi]
    parent_ix = numpy.arange(len(psi))
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
        # Todo: look at what we actually need to copy.
        psi[i] = copy.deepcopy(new_psi[p])
        psi[i].weight = 1.0
    # Need to be a bit careful here. When we perform population control by
    # updating psi_history using the population control map we are only copying
    # references to the walkers which are stored in memory. Since these are not
    # modified anywhere else in the code (we deepcopy historic wavefunctions
    # into psi_history in the first place) we don't need to worry about any
    # modifications to walkers being propagated through to other walkers with a
    # common ancestor. This only serves to copy the parent's field history up
    # to this point in the simulation if branching/death has occured by copying
    # the reference to the parent's walker object in memory and thus avoids an
    # expensive deepcopy operation.
    if psi_history is not None:
        psi_history = psi_history[parent_ix]
    return psi_history
