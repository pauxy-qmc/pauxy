import numpy
import scipy.linalg
import itertools

def simple_fci(system, gen_dets=False, occs=None, hamil=False):
    """Very dumb FCI routine."""
    orbs = numpy.arange(system.nbasis)
    if occs is None:
        oa = [c for c in itertools.combinations(orbs, system.nup)]
        ob = [c for c in itertools.combinations(orbs, system.ndown)]
        oa, ob = zip(*itertools.product(oa,ob))
    else:
        oa, ob = occs
    # convert to spin orbitals
    dets = [[j for j in a] + [i+system.nbasis for i in c] for (a,c) in zip(oa,ob)]
    dets = [numpy.sort(d) for d in dets]
    ndets = len(dets)
    H = numpy.zeros((ndets,ndets))
    for i in range(ndets):
        for j in range(i,ndets):
            H[i,j] = get_hmatel(system, dets[i], dets[j])
    if gen_dets:
        return scipy.linalg.eigh(H, lower=False), (dets,numpy.array(oa),numpy.array(ob))
    elif hamil:
        return scipy.linalg.eigh(H, lower=False), H
    else:
        return scipy.linalg.eigh(H, lower=False)


def get_hmatel(system, di, dj):
    from_orb = list(set(dj)-set(di))
    to_orb = list(set(di)-set(dj))
    from_orb.sort()
    to_orb.sort()
    nex = len(from_orb)
    perm = get_perm(from_orb, to_orb, di, dj)
    if nex == 0:
        hmatel = slater_condon0(system, di)[0]
    elif nex == 1:
        i, si = map_orb(from_orb[0], system.nbasis)
        a, sa = map_orb(to_orb[0], system.nbasis)
        hmatel = slater_condon1(system, (i,si), (a,sa), di, perm)
    elif nex == 2:
        # < ij | ab > or < ij | ba >
        i, si = map_orb(from_orb[0], system.nbasis)
        j, sj = map_orb(from_orb[1], system.nbasis)
        a, sa = map_orb(to_orb[0], system.nbasis)
        b, sb = map_orb(to_orb[1], system.nbasis)
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
        perm += io - nmove
        nmove += 1
    nmove = 0
    for o in to_orb:
        io = numpy.where(di==o)[0]
        perm += io - nmove
        nmove += 1
    return perm % 2 == 1

def slater_condon0(system, occs):
    e1b = 0.0
    e2b = 0.0
    e1b = system.ecore
    for i in range(len(occs)):
        ii, spin_ii = map_orb(occs[i], system.nbasis)
        # Todo: Update if H1 is ever spin dependent.
        e1b += system.H1[0,ii,ii]
        for j in range(i+1,len(occs)):
            jj, spin_jj = map_orb(occs[j], system.nbasis)
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
        oj, soj = map_orb(oj, system.nbasis)
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

def map_orb(orb, nbasis):
    """Map spin orbital to spatial index."""
    if orb // nbasis == 0:
        s = 0
        ix = orb
    else:
        s = 1
        ix = orb - nbasis
    return ix, s

def get_one_body_matel(ints, di, dj):
    from_orb = list(set(dj)-set(di))
    to_orb = list(set(di)-set(dj))
    nex = len(from_orb)
    perm = get_perm(from_orb, to_orb, di, dj)
    matel = 0.0
    if nex == 0:
        for i in range(len(di)):
            ii, spin_ii = map_orb(di[i], ints.shape[-1])
            matel += ints[ii,ii]
    elif nex == 1:
        i, si = map_orb(from_orb[0], ints.shape[-1])
        a, sa = map_orb(to_orb[0], ints.shape[-1])
        assert si == sa
        matel = ints[i,a]
    else:
        matel = 0.0
    if perm:
        return -matel
    else:
        return matel
