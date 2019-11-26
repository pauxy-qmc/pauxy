import numpy
import scipy.linalg
import itertools

def simple_fci_bose_fermi(system, nboson_max = 1, gen_dets=False, occs=None, hamil=False):
    """Very dumb FCI routine."""
    orbs = numpy.arange(system.nbasis)
    # bosons
    perms = []
    for ib in range(nboson_max+1):
        perms += [c for c in itertools.product(orbs, repeat=ib)]

    # fermions
    if occs is None:
        oa = [c for c in itertools.combinations(orbs, system.nup)]
        ob = [c for c in itertools.combinations(orbs, system.ndown)]
        print(oa)
        oa, ob = zip(*itertools.product(oa,ob))
    else:
        oa, ob = occs

    # convert to spin orbitals
    dets = [[j for j in a] + [i+system.nbasis for i in c] for (a,c) in zip(oa,ob)]
    print("dets = {}".format(dets))
    print("perms = {}".format(perms))
    dets = [numpy.sort(d) for d in dets]

    nperms = len(perms)
    ndets = len(dets)

    print("# ndets, nperms = {}, {}".format(ndets, nperms))
    
    Htot = numpy.zeros((ndets*nperms, ndets*nperms))
    
    Hel = numpy.zeros((ndets,ndets))
    for i in range(ndets):
        for j in range(i,ndets):
            Hel[i,j] = get_hmatel(system, dets[i], dets[j])
            Hel[j,i] = Hel[i,j]

    Hb = numpy.zeros((nperms, nperms))

    for i in range(nperms):
        for j in range(i,nperms):
            Hb[i,j] = get_hmatboson(system, perms[i], perms[j])
            Hb[j,i] = Hb[i,j]

    for i, perm in enumerate(perms):
        offset = i * ndets
        Htot[offset:offset+ndets, offset:offset+ndets] = Hel.copy()
        Htot[offset:offset+ndets, offset:offset+ndets] += Hb[i, i]

    for ip, pi in enumerate(perms):
        offset_i = ip * ndets
        for idxdi, di in enumerate(dets):
            for jp, pj in enumerate(perms):
                offset_j = jp * ndets
                for idxdj, dj in enumerate(dets):
                    get_holstein(system, pi, pj, di, dj)

    if gen_dets:
        return scipy.linalg.eigh(Htot, lower=False), (dets,numpy.array(oa),numpy.array(ob))
    elif hamil:
        return scipy.linalg.eigh(Htot, lower=False), Htot
    else:
        return scipy.linalg.eigh(Htot, lower=False)

def to_occ (perm, nbsf):
    occ_string = numpy.zeros(nbsf)
    for i in range(nbsf):
        

def get_holstein(system, pi, pj, di, dj):
    nbsf = system.nbasis

    api = numpy.asarray(pi)
    apj = numpy.asarray(pj)

    ndiff = abs(len(api) - len(apj))

    if (ndiff == 1):
        print(list(api), list(apj))
        if (len(api) > len(apj)):
            result = all(elem in list(api) for elem in list(apj))
        else:
            result = all(elem in list(apj) for elem in list(api))
        print(result)



# only diagonal is supported for now
def get_hmatboson(system, pi, pj):
    if (pi == pj):
        p = numpy.asarray(pi)
        h = 0.0
        for i, isite in enumerate(p): # going through each boson
            h += system.hb1(isite, isite)
        return h
    else:
        return 0.0
    

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
