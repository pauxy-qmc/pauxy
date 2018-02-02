""" Population control algorithms. """
import numpy
import copy

def comb(psi, nw):
    """ Apply the comb method of population control / branching.

    See Booth & Gubernatis PRE 80, 046704 (2009).

    .. warning::
        This algorithm is biased and not necessarily correct.

    Todo : implement consistent algorithm.

    Parameters
    ----------
    psi : list of :class:`pauxy.walker.Walker` objects
        current distribution of walkers, i.e., at the current iteration in the
        simulation corresponding to :math:`\tau'=\tau+\tau_{bp}`.
    nw : int
        Number of walkers on current processor.
    """
    # Need make a copy to since the elements in psi are only references to
    # walker objects in memory. We don't want future changes in a given element
    # of psi having unintended consequences.
    new_psi = copy.deepcopy(psi)
    weights = [w.weight for w in psi.walkers]
    parent_ix = numpy.arange(len(psi.walkers))
    total_weight = sum(weights)
    cprobs = numpy.cumsum(weights)

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
        if i != p:
            psi.walkers[i] = copy.deepcopy(new_psi.walkers[p])
        psi.walkers[i].weight = 1.0
