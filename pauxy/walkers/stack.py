import numpy
import scipy.linalg
from pauxy.utils.misc import get_numeric_names
from pauxy.utils.linalg import column_pivoted_qr, column_pivoted_qr_low_rank
from pauxy.walkers.utils import get_numeric_buffer, set_numeric_buffer

class FieldConfig(object):
    """Object for managing stored auxilliary field.

    Parameters
    ----------
    nfields : int
        Number of fields to store for each back propagation step.
    nprop_tot : int
        Total number of propagators to store for back propagation + itcf.
    nbp : int
        Number of back propagation steps.
    dtype : type
        Field configuration type.
    """
    def __init__(self, nfields, nprop_tot, nbp, dtype):
        self.configs = numpy.zeros(shape=(nprop_tot, nfields), dtype=dtype)
        self.cos_fac = numpy.zeros(shape=(nprop_tot, 1), dtype=float)
        self.weight_fac = numpy.zeros(shape=(nprop_tot, 1), dtype=complex)
        self.tot_wfac = 1.0 + 0j
        self.step = 0
        # need to account for first iteration and how we iterate
        self.block = -1
        self.ib = 0
        self.nfields = nfields
        self.nbp = nbp
        self.nprop_tot = nprop_tot
        self.nblock = nprop_tot // nbp
        self.buff_names, self.buff_size = get_numeric_names(self.__dict__)

    def push(self, config):
        """Add field configuration to buffer.

        Parameters
        ----------
        config : int
            Auxilliary field configuration.
        """
        self.configs[self.step, self.ib] = config
        self.ib = (self.ib + 1) % self.nfields
        # Completed field configuration for this walker?
        if self.ib == 0:
            self.step = (self.step + 1) % self.nprop_tot
            # Completed this block of back propagation steps?
            if self.step % self.nbp == 0:
                self.block = (self.block + 1) % self.nblock

    def update(self, config, wfac):
        """Add full field configuration for walker to buffer.

        Parameters
        ----------
        config : :class:`numpy.ndarray`
            Auxilliary field configuration.
        cfac : float
            Cosine factor if using phaseless approximation.
        wfac : complex
            Weight factor to restore full walker weight following phaseless
            approximation.
        """
        self.configs[self.step] = config
        self.weight_fac[self.step] = wfac[0]
        # cosine factor is real..
        self.cos_fac[self.step] = wfac[1].real
        if abs(wfac[1]) > 1e-16:
            self.tot_wfac *= wfac[0]/wfac[1]
        else:
            self.tot_wfac = 0
        # Completed field configuration for this walker?
        self.step = self.step + 1
        # Completed this block of back propagation steps?
        if self.step % self.nbp == 0:
            self.block = (self.block + 1) % self.nblock

    def get_block(self):
        """Return a view to current block for back propagation."""
        end = self.step
        return (self.configs[:end], self.cos_fac[:end], self.weight_fac[:end])

    def get_superblock(self):
        """Return a view to current super block for ITCF."""
        end = self.nprop_tot - self.nbp
        return (self.configs[:end], self.cos_fac[:end], self.weight_fac[:end])

    def get_buffer(self):
        return get_numeric_buffer(self.__dict__,
                                  self.buff_names,
                                  self.buff_size)

    def set_buffer(self, buff):
        set_numeric_buffer(self.__dict__,
                           self.buff_names,
                           self.buff_size,
                           buff)

    def get_wfac(self):
        cfac = numpy.prod(self.cos_fac[:self.step])
        wfac = numpy.prod(self.weight_fac[:self.step])
        return cfac, wfac


    def reset(self):
        if self.step % self.nprop_tot == 0:
            self.step = 0
        self.tot_wfac = 1.0 + 0j

class PropagatorStack:

    def __init__(self, stack_size, ntime_slices, nbasis, dtype, BT=None, BTinv=None,
                 diagonal=False, averaging=False, lowrank=True, thresh=1e-6):

        self.time_slice = 0
        self.stack_size = stack_size
        self.ntime_slices = ntime_slices
        self.nbins = ntime_slices // self.stack_size
        self.diagonal_trial = diagonal
        self.averaging = averaging
        self.thresh = thresh

        self.lowrank = lowrank
        self.sgndet = [1.0, 1.0]
        self.logdet = [0.0, 0.0]

        self.reortho = 1

        if self.nbins * self.stack_size < self.ntime_slices:
            print("stack_size must divide the total path length")
            assert self.nbins * self.stack_size == self.ntime_slices

        self.nbasis = nbasis
        self.dtype = dtype
        self.BT = BT
        self.BTinv = BTinv
        self.counter = 0
        self.block = 0

        self.stack = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis),
                                 dtype=dtype)
        self.left = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis),
                                dtype=dtype)
        self.right = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis),
                                 dtype=dtype)

        self.G = numpy.asarray([numpy.eye(self.nbasis, dtype=dtype),
                                numpy.eye(self.nbasis, dtype=dtype)])

        if self.lowrank:
            if self.diagonal_trial:
                self.update_new = self.update_low_rank
            else:
                self.update_new = self.update_low_rank_non_diag
        else:
            self.update_new = self.update_full_rank

        # Global block matrix
        if self.lowrank:
            self.Ql = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)
            self.Dl = numpy.zeros(shape=(2, nbasis), dtype=dtype)
            self.Tl = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)

            self.Qr = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)
            self.Dr = numpy.zeros(shape=(2, nbasis), dtype=dtype)
            self.Tr = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)

            self.CT = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)
            self.theta = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)
            self.mT = nbasis

        self.buff_names, self.buff_size = get_numeric_names(self.__dict__)
        # set all entries to be the identity matrix
        self.reset()

    def get(self, ix):
        return self.stack[ix]

    def get_buffer(self):
        return get_numeric_buffer(self.__dict__,
                                  self.buff_names,
                                  self.buff_size)

    def set_buffer(self, buff):
        set_numeric_buffer(self.__dict__,
                           self.buff_names,
                           self.buff_size,
                           buff)

    def set_all(self, BT):
        # Diagonal = True assumes BT is diagonal and left is also diagonal
        if self.diagonal_trial:
            for i in range(0, self.ntime_slices):
                ix = i // self.stack_size # bin index
                self.left[ix,0] = numpy.diag(numpy.multiply(BT[0].diagonal(),self.left[ix,0].diagonal()))
                self.left[ix,1] = numpy.diag(numpy.multiply(BT[1].diagonal(),self.left[ix,1].diagonal()))
                self.stack[ix,0] = self.left[ix,0].copy()
                self.stack[ix,1] = self.left[ix,1].copy()
        else:
            for i in range(0, self.ntime_slices):
                ix = i // self.stack_size # bin index
                self.left[ix,0] = numpy.dot(BT[0],self.left[ix,0])
                self.left[ix,1] = numpy.dot(BT[1],self.left[ix,1])
                self.stack[ix,0] = self.left[ix,0].copy()
                self.stack[ix,1] = self.left[ix,1].copy()

        if self.lowrank:
            self.initialize_left()
            for s in [0,1]:
                self.Qr[s] = numpy.identity(self.nbasis, dtype=self.dtype)
                self.Dr[s] = numpy.ones(self.nbasis, dtype=self.dtype)
                self.Tr[s] = numpy.identity(self.nbasis, dtype=self.dtype)

    def reset(self):
        self.time_slice = 0
        self.block = 0
        for i in range(0, self.nbins):
            self.stack[i,0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.stack[i,1] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.right[i,0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.right[i,1] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.left[i,0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.left[i,1] = numpy.identity(self.nbasis, dtype=self.dtype)

        if self.lowrank:
            for s in [0,1]:
                self.Qr[s] = numpy.identity(self.nbasis, dtype=self.dtype)
                self.Dr[s] = numpy.ones(self.nbasis,dtype=self.dtype)
                self.Tr[s] = numpy.identity(self.nbasis, dtype=self.dtype)

    # Form BT product for i = 1, ..., nslices - 1 (i.e., skip i = 0)
    # \TODO add non-diagonal version of this
    def initialize_left(self):
        if self.diagonal_trial:
            for spin in [0, 1]:
                # We will assume that B matrices are all diagonal for left....
                # B = self.stack[1]
                B = self.stack[0]
                self.Dl[spin] = (B[spin].diagonal())
                self.Ql[spin] = numpy.identity(B[spin].shape[0])
                self.Tl[spin] = numpy.identity(B[spin].shape[0])

                # for ix in range(2, self.nbins):
                for ix in range(1, self.nbins):
                    B = self.stack[ix]
                    C2 = numpy.einsum('ii,i->i',B[spin],self.Dl[spin])
                    self.Dl[spin] = C2
        else:
            for spin in [0, 1]:
                B = self.stack[0][spin]
                Q, D, T = column_pivoted_qr(B)
                mR = len(D[numpy.abs(D)>self.thresh])
                # print(T, Q, D)
                for ix in range(1, self.nbins):
                    mR = len(D[numpy.abs(D)>self.thresh])
                    B = self.stack[ix][spin]
                    C = numpy.dot(B, Q[:,:mR])
                    C = numpy.einsum('ij,j->ij', C[:,:mR], D[:mR])
                    (Q, D, Tnew, mT) = column_pivoted_qr_low_rank(C, update=True, thresh=self.thresh)
                    T = numpy.dot(Tnew, T[:mR,:])
                self.Tl[spin,:mT,:] = T
                self.Ql[spin] = Q
                self.Dl[spin,:mT] = D
                self.Dl[spin,mT:] = 0.0
                mR = len(D[numpy.abs(D)>self.thresh])
                Db, Ds = split_diagonal(D, mR, dtype=B.dtype)
                sdet, logdet, TQinv, QDT = determinant_low_rank(T, Q, D,
                                                                Db, Ds,
                                                                mR, mR)
                self.sgndet[spin] = sdet
                self.logdet[spin] = logdet
                self.greens_function_low_rank(QDT, TQinv,
                                              T, Q, D,
                                              Db, mR, mR, mR, spin,
                                              dtype=B[spin].dtype)
        mL = len(self.Dl[0,numpy.abs(self.Dl[0])>self.thresh])

    def update(self, B):
        if self.counter == 0:
            self.stack[self.block,0] = numpy.identity(B.shape[-1],
                                                      dtype=B.dtype)
            self.stack[self.block,1] = numpy.identity(B.shape[-1],
                                                      dtype=B.dtype)
        self.stack[self.block,0] = B[0].dot(self.stack[self.block,0])
        self.stack[self.block,1] = B[1].dot(self.stack[self.block,1])
        self.time_slice = (self.time_slice + 1) % self.ntime_slices
        self.block = self.time_slice // self.stack_size
        self.counter = (self.counter + 1) % self.stack_size

    def update_full_rank(self, B):
        # Diagonal = True assumes BT is diagonal and left is also diagonal
        if self.counter == 0:
            self.right[self.block,0] = numpy.identity(B.shape[-1], dtype=B.dtype)
            self.right[self.block,1] = numpy.identity(B.shape[-1], dtype=B.dtype)

        if self.diagonal_trial:
            self.left[self.block,0] = numpy.diag(numpy.multiply(self.left[self.block,0].diagonal(),self.BTinv[0].diagonal()))
            self.left[self.block,1] = numpy.diag(numpy.multiply(self.left[self.block,1].diagonal(),self.BTinv[1].diagonal()))
        else:
            self.left[self.block,0] = self.left[self.block,0].dot(self.BTinv[0])
            self.left[self.block,1] = self.left[self.block,1].dot(self.BTinv[1])

        self.right[self.block,0] = B[0].dot(self.right[self.block,0])
        self.right[self.block,1] = B[1].dot(self.right[self.block,1])

        if self.diagonal_trial:
            self.stack[self.block,0] = numpy.einsum('ii,ij->ij',self.left[self.block,0],self.right[self.block,0])
            self.stack[self.block,1] = numpy.einsum('ii,ij->ij',self.left[self.block,1],self.right[self.block,1])
        else:
            self.stack[self.block,0] = self.left[self.block,0].dot(self.right[self.block,0])
            self.stack[self.block,1] = self.left[self.block,1].dot(self.right[self.block,1])

        self.time_slice = self.time_slice + 1 # Count the time slice
        self.block = self.time_slice // self.stack_size # move to the next block if necessary
        self.counter = (self.counter + 1) % self.stack_size # Counting within a stack

    def update_low_rank(self, B):
        assert not self.averaging
        # Diagonal = True assumes BT is diagonal and left is also diagonal
        assert self.diagonal_trial

        if self.counter == 0:
            self.Tl = self.left[self.block]

        mR = B.shape[-1] # initial mR
        mL = B.shape[-1] # initial mL
        mT = B.shape[-1] # initial mT
        next_block = (self.time_slice+1) // self.stack_size # move to the next block if necessary
        for s in [0,1]:
            # 1. Update Left stack by BTinv
            mR = len(self.Dr[s][numpy.abs(self.Dr[s])>self.thresh])
            self.Dl[s] = numpy.einsum("i,ii->i", self.Dl[s], self.BTinv[s])
            mL = len(self.Dl[s][numpy.abs(self.Dl[s])>self.thresh])

            # 2. Update Right stack by added propagator B
            self.Qr[s][:,:mR] = B[s].dot(self.Qr[s][:,:mR]) # N x mR
            self.Qr[s][:,mR:] = 0.0

            # 3. Compute (Qr Dr)
            Ccr = numpy.einsum('ij,j->ij',self.Qr[s][:,:mR], self.Dr[s][:mR]) # N x mR
            if next_block > self.block: # Do QR and update here?
                # If moving past block in next step need to do recompute QR
                (Qlcr, Dlcr, T) = column_pivoted_qr_low_rank(Ccr)
                self.Dr[s][:mR] = Dlcr
                self.Dr[s][mR:] = 0.0
                self.Qr[s] = Qlcr
                Tlcr = numpy.dot(T, self.Tr[s][:mR,:]) # mR x N
                self.Tr[s][:mR,:] = Tlcr
                # assume left stack is all diagonal (i.e., QDT = diagonal -> Q and T are identity)
                # 4. Compute L R.
                Clcr = numpy.einsum('i,ij->ij',
                                    self.Dl[s][:mL],
                                    numpy.einsum('ij,j->ij', Qlcr[:mL,:mR], Dlcr[:mR])) # mL x mR
            else:
                # 4. Compute L R.
                Clcr = numpy.einsum('i,ij->ij', self.Dl[s][:mL], Ccr[:mL,:mR])

            # 5. Compute A = q d t
            # a. Compute CPQR of Clcr = (QD)
            Qlcr, Dlcr, T, mT = column_pivoted_qr_low_rank(Clcr,
                                                           update=True,
                                                           thresh=self.thresh)
            # b. Compute Tlcr = (Tlcr) tr
            Tlcr = numpy.dot(T, self.Tr[s][:mR,:]) # mT x N

            # 6. Split diagonal into Db and Ds needed for stratification.
            Db, Ds = split_diagonal(Dlcr, mT, dtype=B[s].dtype)

            # 7. Compute det(1+A) = det(1+qdt) = det(1+dtq)
            sdet, logdet, TQinv, QDT = determinant_low_rank(Tlcr, Qlcr,
                                                            Dlcr,
                                                            Db, Ds,
                                                            mT, mL)
            self.sgndet[s] = sdet
            self.logdet[s] = logdet

            # 8. Update Green's function:
            self.greens_function_low_rank(QDT, TQinv,
                                          Tlcr, Qlcr, Dlcr,
                                          Db, mL, mR, mT, s,
                                          dtype=B[s].dtype)

        self.mT = mT
        self.time_slice = self.time_slice + 1 # Count the time slice
        self.block = self.time_slice // self.stack_size # move to the next block if necessary
        self.counter = (self.counter + 1) % self.stack_size # Counting within a stack

    def update_low_rank_non_diag(self, B):
        assert not self.averaging
        # Diagonal = True assumes BT is diagonal and left is also diagonal

        mR = B.shape[-1] # initial mR
        mL = B.shape[-1] # initial mL
        mT = B.shape[-1] # initial mT
        next_block = (self.time_slice+1) // self.stack_size # move to the next block if necessary
        for s in [0,1]:
            # 1. Update Left stack by BTinv
            mR = len(self.Dr[s][numpy.abs(self.Dr[s])>self.thresh])
            # print("tl: ", self.Dl[s])
            self.Tl[s] = numpy.dot(self.Tl[s], self.BTinv[s])
            mL = len(self.Dl[s][numpy.abs(self.Dl[s])>self.thresh])

            # 2. Update Right stack by added propagator B
            self.Qr[s][:,:mR] = numpy.dot(B[s], self.Qr[s][:,:mR]) # N x mR
            self.Qr[s][:,mR:] = 0.0

            # 3. Ccr = B qR dR
            Ccr = numpy.einsum('ij,j->ij', self.Qr[s][:,:mR], self.Dr[s][:mR]) # N x mR
            # 4. Clcr = dL tL Ccr
            if next_block > self.block: # Do QR and update here?
                # If moving past block in next step need to do recompute QR
                # right block
                (Qlcr, Dlcr, T) = column_pivoted_qr_low_rank(Ccr)
                self.Dr[s][:mR] = Dlcr
                self.Dr[s][mR:] = 0.0
                self.Qr[s] = Qlcr
                Tlcr = numpy.dot(T, self.Tr[s][:mR,:]) # mR x N
                self.Tr[s][:mR,:] = Tlcr
                # left block
                Clc = numpy.einsum('ij,j->ij', self.Ql[s][:,:mL], self.Dl[s][:mL]) # N x mL
                (Qlc, Dlc, T) = column_pivoted_qr_low_rank(Clc)
                self.Dl[s][:mL] = Dlc
                self.Dl[s][mL:] = 0.0
                self.Ql[s] = Qlc
                Tlcr = numpy.dot(T, self.Tl[s][:mL,:]) # mL x N
                self.Tl[s][:mL,:] = Tlcr
                A1 = numpy.einsum('i,ij->ij', self.Dl[s][:mL], self.Tl[s,:mL,:])
                A2 = numpy.einsum('ij,j->ij', Qlcr[:,:mR], Dlcr[:mR]) # mL x mR
                # print(A1.shape, A2.shape)
                Clcr = numpy.dot(A1,A2)
            else:
                Clcr = numpy.dot(numpy.einsum('i,ij->ij', self.Dl[s][:mL], self.Tl[s,:mL,:]), Ccr[:,:mR]) # mL x mR
            BB = numpy.dot(numpy.dot(self.Qr[s], numpy.diag(self.Dr[s])), self.Tr[s])
            # 5. Compute A = q d t
            # a. Compute CPQR of Clcr = (QD)
            # self.Qr[s] = numpy.dot(scipy.linalg.inv(B[s]), self.Qr[s])
            # self.Tl[s] = numpy.dot(self.Tl[s], self.BT[s])
            # Clcr = numpy.dot(numpy.einsum('i,ij->ij', self.Dl[s][:mL], self.Tl[s,:mL,:]),
                             # numpy.einsum('ij,j->ij', self.Qr[s][:mL,:mR],
                                 # self.Dr[s][:mR])) # mL x mR
            Qlcr, Dlcr, T, mT = column_pivoted_qr_low_rank(Clcr, update=True, thresh=self.thresh)
            # b. Qlcr = qL Qlcr
            Qlcr = numpy.dot(self.Ql[s,:,:mL], Qlcr) # N x mT
            # c. Compute Tlcr = (Tlcr) tr
            Tlcr = numpy.dot(T, self.Tr[s][:mR,:]) # mT x N

            # 6. Split diagonal into Db and Ds needed for stratification.
            Db, Ds = split_diagonal(Dlcr, mT, dtype=B[s].dtype)

            # 7. Compute det(1+A) = det(1+qdt) = det(1+dtq)
            sdet, logdet, TQinv, QDT = determinant_low_rank(Tlcr, Qlcr,
                                                            Dlcr,
                                                            Db, Ds,
                                                            mT, mL)
            self.sgndet[s] = sdet
            self.logdet[s] = logdet

            # 8. Update Green's function:
            self.greens_function_low_rank(QDT, TQinv,
                                          Tlcr, Qlcr, Dlcr,
                                          Db, mL, mR, mT, s,
                                          dtype=B[s].dtype)

        self.mT = mT
        self.time_slice = self.time_slice + 1 # Count the time slice
        self.block = self.time_slice // self.stack_size # move to the next block if necessary
        self.counter = (self.counter + 1) % self.stack_size # Counting within a stack

    def greens_function_low_rank(self, QDT, TQinv,
                                 Tlcr, Qlcr, Dlcr,
                                 Db, mL, mR, mT, spin,
                                 dtype=numpy.float64):
        # [QT^{-1}db + ds]^{-1}
        # print(mL, mR, mT)
        assert QDT.shape == (mT, mT)
        QDTinv = scipy.linalg.inv(QDT, check_finite=False)
        # print(QDTinv.shape)
        # Db ([QT^{-1}db + ds]^{-1} [TQ]^{-1})
        A = numpy.einsum("i,ij->ij", Db, QDTinv.dot(TQinv)) # mT x mT
        Qlcr_pad = numpy.zeros((self.nbasis, self.nbasis), dtype=dtype)
        # print(mL, mR, mT, Qlcr.shape)
        Qlcr_pad[:mL,:mT] = Qlcr[:mL,:mT]
        self.CT[spin][:,:] = 0.0
        self.CT[spin][:,:mT] = (A.dot(Tlcr)).T.conj()
        self.theta[spin][:,:] = 0.0
        self.theta[spin][:mT,:] = Qlcr_pad[:,:mT].dot(numpy.diag(Dlcr[:mT])).T
        self.G[spin] = numpy.eye(self.nbasis, dtype=dtype) - self.theta[spin][:mT,:].T.dot(self.CT[spin][:,:mT].T.conj())

def split_diagonal(Dlcr, mT, dtype=numpy.float64):
    Db = numpy.zeros(mT, dtype)
    Ds = numpy.zeros(mT, dtype)
    for i in range(mT):
        absDlcr = abs(Dlcr[i])
        if absDlcr > 1.0:
            Db[i] = 1.0 / absDlcr
            Ds[i] = numpy.sign(Dlcr[i])
        else:
            Db[i] = 1.0
            Ds[i] = Dlcr[i]
    return Db, Ds

def determinant_low_rank(Tlcr, Qlcr, Dlcr, Db, Ds, mT, mL):
    # Compute det(1+A)
    # [TQ^{-1}Db + Ds] Db^{-1} TQ
    Dbinv = 1.0 / Db
    TQ = Tlcr[:,:mL].dot(Qlcr[:mL,:mT]) # mT x mT
    TQinv = scipy.linalg.inv(TQ, check_finite=False)
    QDT = numpy.einsum('ij,j->ij', TQinv, Db) + numpy.diag(Ds) # mT x mT
    # M = 1 + A, mT x mT
    M = numpy.einsum("ij,j->ij", QDT, Dbinv).dot(TQ)
    # self.ovlp[s] = 1.0 / scipy.linalg.det(M, check_finite=False)
    # want log(det(1+A)) = log (det(M))
    sdet, logdet = numpy.linalg.slogdet(M)
    return sdet, logdet, TQinv, QDT
