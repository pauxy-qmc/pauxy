import numpy
import scipy.linalg

def greens_function_qr_strat(stack, slice_ix=None):
    if slice_ix == None:
        slice_ix = stack.time_slice
    bin_ix = slice_ix // stack.stack_size
    # print(bin_ix, slice_ix, stack.nbins, stack.stack_size)
    # For final time slice want first block to be the rightmost (for energy
    # evaluation).
    if bin_ix == stack.nbins:
        bin_ix = -1

    assert stack.dtype == numpy.complex128
    G = numpy.zeros((2,stack.nbasis,stack.nbasis), stack.dtype)

    for spin in [0, 1]:
        # Need to construct the product A(l) = B_l B_{l-1}..B_L...B_{l+1} in
        # stable way. Iteratively construct column pivoted QR decompositions
        # (A = QDT) starting from the rightmost (product of) propagator(s).
        B = stack.get((bin_ix+1)%stack.nbins)

        (Q1, R1, P1) = scipy.linalg.qr(B[spin], pivoting=True,
                                       check_finite=False)
        # Form D matrices
        D1 = numpy.diag(R1.diagonal())
        D1inv = numpy.diag(1.0/R1.diagonal())
        T1 = numpy.einsum('ii,ij->ij', D1inv, R1)
        # permute them
        T1[:,P1] = T1 [:, range(stack.nbasis)]

        for i in range(2, stack.nbins+1):
            ix = (bin_ix + i) % stack.nbins
            B = stack.get(ix)
            C2 = numpy.dot(numpy.dot(B[spin], Q1), D1)
            (Q1, R1, P1) = scipy.linalg.qr(C2, pivoting=True,
                                           check_finite=False)
            # Compute D matrices
            D1inv = numpy.diag(1.0/R1.diagonal())
            D1 = numpy.diag(R1.diagonal())
            tmp = numpy.einsum('ii,ij->ij',D1inv, R1)
            tmp[:,P1] = tmp[:,range(stack.nbasis)]
            T1 = numpy.dot(tmp, T1)

        # G^{-1} = 1+A = 1+QDT = Q (Q^{-1}T^{-1}+D) T
        # Write D = Db^{-1} Ds
        # Then G^{-1} = Q Db^{-1}(Db Q^{-1}T^{-1}+Ds) T
        Db = numpy.zeros(B[spin].shape, B[spin].dtype)
        Ds = numpy.zeros(B[spin].shape, B[spin].dtype)
        for i in range(Db.shape[0]):
            absDlcr = abs(Db[i,i])
            if absDlcr > 1.0:
                Db[i,i] = 1.0 / absDlcr
                Ds[i,i] = numpy.sign(D1[i,i])
            else:
                Db[i,i] = 1.0
                Ds[i,i] = D1[i,i]

        T1inv = scipy.linalg.inv(T1, check_finite = False)
        # C = (Db Q^{-1}T^{-1}+Ds)
        C = numpy.dot(numpy.einsum('ii,ij->ij',Db, Q1.conj().T), T1inv) + Ds
        Cinv = scipy.linalg.inv(C, check_finite=False)

        # Then G = T^{-1} C^{-1} Db Q^{-1}
        # Q is unitary.
        G[spin] = numpy.dot(numpy.dot(T1inv, Cinv),
                                 numpy.einsum('ii,ij->ij', Db, Q1.conj().T))
    return G

def recompute_greens_function(fields, stack, auxf, BH1,
                              time_slice=None,
                              from_scratch=False):
    BV = numpy.zeros((2,stack.nbasis), dtype=numpy.float64)
    # print(aux_wfac_)
    if from_scratch:
        stack.reset()
        for f in fields:
            BVa = numpy.diag(numpy.array([auxf[xi,0] for xi in f]))
            BVb = numpy.diag(numpy.array([auxf[xi,1] for xi in f]))
            # Vsii Tsij
            # B = numpy.einsum('ki,kij->kij', BV, BH1)
            Ba = numpy.dot(BVa, BH1[0])
            Bb = numpy.dot(BVb, BH1[1])
            # Ba = numpy.dot(BH1[0], Ba)
            # Bb = numpy.dot(BH1[1], Bb)
            stack.update(numpy.array([Ba,Bb]))
    else:
        end = (time_slice+1)
        beg = (time_slice+1) - stack.stack_size
        block = time_slice // stack.stack_size
        stack.stack[block,0] = numpy.identity(BH1.shape[-1],
                                               dtype=stack.dtype)
        stack.stack[block,1] = numpy.identity(BH1.shape[-1],
                                               dtype=stack.dtype)
        for f in fields[beg:end]:
            BVa = numpy.diag(numpy.array([auxf[xi,0] for xi in f]))
            BVb = numpy.diag(numpy.array([auxf[xi,1] for xi in f]))
            # Vsii Tsij
            # B = numpy.einsum('ki,kij->kij', BV, BH1)
            Ba = numpy.dot(BVa, BH1[0])
            Bb = numpy.dot(BVb, BH1[1])
            # Ba = numpy.dot(BH1[0], Ba)
            # Bb = numpy.dot(BH1[1], Bb)
            stack.stack[block,0] = Ba.dot(stack.stack[block,0])
            stack.stack[block,1] = Bb.dot(stack.stack[block,1])
    return greens_function_qr_strat(stack, slice_ix=time_slice)

def propagate_greens_function(G, fields, BH1inv, BH1, auxf):
    BVa = numpy.diag([auxf[xi,0] for xi in fields])
    BVb = numpy.diag([auxf[xi,1] for xi in fields])
    B = numpy.array([numpy.dot(BVa, BH1[0]),
                     numpy.dot(BVb, BH1[1])])
    # B[0] = numpy.dot(BH1[0], B[0])
    # B[0] = numpy.dot(BH1[1], B[1])
    return numpy.array([numpy.dot(B[0], numpy.dot(G[0], scipy.linalg.inv(B[0]))),
                        numpy.dot(B[1], numpy.dot(G[1], scipy.linalg.inv(B[1])))])
    # return numpy.array([numpy.dot(scipy.linalg.inv(B[0]), numpy.dot(G[0], B[0])),
                        # numpy.dot(scipy.linalg.inv(B[1]), numpy.dot(G[1], B[1]))])

def update_greens_function(G, i, xi, delta):
    for spin in [0,1]:
        g = G[spin,:,i]
        gbar = -G[spin,i,:]
        gbar[i] += 1
        denom = 1 + (1-g[i]) * delta[xi,spin]
        G[spin] = (
            G[spin] - delta[xi,spin]*numpy.einsum('i,j->ij', g, gbar) / denom
        )
