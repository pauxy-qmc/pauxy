import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
from pyscf.mp import mp2
#from pyscf.mp.mp2 import make_rdm1, make_rdm2, make_rdm1_ao
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_dfmp2_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE):
    if mo_energy is None or mo_coeff is None:
        mo_coeff = mp2._mo_without_core(mp, mp.mo_coeff)
        mo_energy = mp2._mo_energy_without_core(mp, mp.mo_energy)
    else:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert(mp.frozen is 0 or mp.frozen is None)

    nocc = mp.nocc
    nvir = mp.nmo - nocc - mp.nfv
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:nocc+nvir]
    t2 = []
    emp2 = 0
    for istep, qov in enumerate(mp.loop_ao2mo(mo_coeff, nocc)):
        logger.debug(mp, 'Load cderi step %d', istep)
        for i in range(nocc):
            buf = numpy.dot(qov[:,i*nvir:(i+1)*nvir].T,
                            qov).reshape(nvir,nocc,nvir)
            gi = numpy.array(buf, copy=False)
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2) # iab
            t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
            emp2 += numpy.einsum('jab,jab', t2i, gi) * 2
            emp2 -= numpy.einsum('jab,jba', t2i, gi)
            gi = gi - gi.transpose(0,2,1) # iab
            t2 += [gi/lib.direct_sum('jb+a->jba', eia, eia[i])]

    return emp2, t2


class DFMP2(mp2.MP2):
    def __init__(self, mf, frozen=0, nfv=0, mo_coeff=None, mo_occ=None):
        self.nfv = nfv
        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if hasattr(mf, 'with_df') and mf.with_df:
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=False)
        self._keys.update(['with_df'])

    @lib.with_doc(mp2.MP2.kernel.__doc__)
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return mp2.MP2.kernel(self, mo_energy, mo_coeff, eris, with_t2, kernel)

    def loop_ao2mo(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order='F')
        nvir = mo.shape[1] - nocc - self.nfv
        mo = mo[:,:nocc+nvir]
        nmo = mo.shape[1]
        ijslice = (0, nocc, nocc, nmo)
        Lov = None
        with_df = self.with_df

        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        blksize = int(min(naux, max(with_df.blockdim,
                                    (max_memory*1e6/8-nocc*nvir**2*2)/(nocc*nvir))))
        for eri1 in with_df.loop(blksize=blksize):
            Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lov)
            yield Lov
