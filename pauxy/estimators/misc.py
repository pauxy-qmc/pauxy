import numpy
import scipy.linalg
import itertools

def simple_fci(system, dets=False):
    """Very dumb FCI routine."""
    orbs = numpy.arange(system.nbasis)
    oa = [c for c in itertools.combinations(orbs, system.nup)]
    ob = [c for c in itertools.combinations(orbs, system.ndown)]
    oa, ob = zip(*itertools.product(oa,ob))
    # convert to spin orbitals
    dets = [[2*i for i in c] for c in oa]
    dets = [alp+[2*i+1 for i in c] for (alp,c) in zip(dets,ob)]
    dets = [numpy.sort(d) for d in dets]
    ndets = len(dets)
    H = numpy.zeros((ndets,ndets))
    for i in range(ndets):
        for j in range(i,ndets):
            H[i,j] = get_hmatel(system, dets[i], dets[j])
    if dets:
        return scipy.linalg.eigh(H, lower=False), (dets,oa,ob)
    else:
        return scipy.linalg.eigh(H, lower=False)


def get_hmatel(system, di, dj):
    i = None
    a = None
    from_orb = list(set(dj)-set(di))
    to_orb = list(set(di)-set(dj))
    nex = len(from_orb)
    perm = get_perm(from_orb, to_orb, di, dj)
    if nex == 0:
        hmatel = slater_condon0(system, di)[0]
    elif nex == 1:
        i, si = map_orb(from_orb[0])
        a, sa = map_orb(to_orb[0])
        assert si == sa
        hmatel = slater_condon1(system, (i,si), (a,sa), di, perm)
    elif nex == 2:
        # < ij | ab > or < ij | ba >
        i, si = map_orb(from_orb[0])
        j, sj = map_orb(from_orb[1])
        a, sa = map_orb(to_orb[0])
        b, sb = map_orb(to_orb[1])
        hmatel = slater_condon2(system, (i,si), (j,sj), (a,sa), (b,sb), perm)
    else:
        hmatel = 0.0
    return hmatel

def get_perm(from_orb, to_orb, di, dj):
    """Determine sign of permutation needed to align two determinants.

    Stolen from HANDE.
    """
    nmove = 0
    perm = 0
    for o in from_orb:
        io = numpy.where(dj==o)[0]
        perm += len(dj) - io - 1 + nmove
    nmove = 0
    for o in to_orb:
        io = numpy.where(di==o)[0]
        perm += len(di) - io - 1 + nmove
    return perm % 2 == 1

def slater_condon0(system, occs):
    e1b = 0.0
    e2b = 0.0
    e1b = system.ecore
    for i in range(len(occs)):
        ii, spin_ii = map_orb(occs[i])
        # Todo: Update if H1 is ever spin dependent.
        e1b += system.H1[0,ii,ii]
        for j in range(i+1,len(occs)):
            jj, spin_jj = map_orb(occs[j])
            e2b += system.hijkl(ii,jj,ii,jj)
            if spin_ii == spin_jj:
                e2b -= system.hijkl(ii,jj,jj,ii)
    hmatel = e1b + e2b
    return hmatel, e1b, e2b

def slater_condon1(system, i, a, occs, perm):
    ii, si = i
    aa, sa = a
    hmatel = system.H1[0,ii,aa]
    nel = system.nup + system.ndown
    for j in range(nel):
        # \sum_j <ij|aj> - <ij|ja>
        oj = occs[j]
        oj, soj = map_orb(oj)
        if 2*oj+soj != 2*ii+si:
            hmatel += system.hijkl(ii,oj,aa,oj)
            if soj == si:
                hmatel -= system.hijkl(ii,oj,oj,aa)
    if perm:
        return -hmatel
    else:
        return hmatel

def slater_condon2(system, i, j, a, b, perm):
    ii, si = i
    jj, sj = j
    aa, sa = a
    bb, sb = b
    hmatel = 0.0
    if si == sa:
        hmatel = system.hijkl(ii,jj,aa,bb)
    if si == sb:
        hmatel -= system.hijkl(ii,jj,bb,aa)
    if perm:
        return -hmatel
    else:
        return hmatel

def map_orb(orb):
    """Map spin orbital to spatial index."""
    if orb % 2 == 0:
        s = 0
        ix = orb // 2
    else:
        s = 1
        ix = (orb-1) // 2
    return ix, s
