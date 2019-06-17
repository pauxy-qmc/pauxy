import unittest
from mpi4py import MPI
import os
from pyscf import gto, ao2mo, scf
from pauxy.qmc.afqmc import AFQMC
from pauxy.systems.generic import Generic
from pauxy.utils.from_pyscf import integrals_from_scf

class TestGeneric(unittest.TestCase):

    def test_from_pyscf(self):
        atom = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
        mf = scf.RHF(atom)
        mf.kernel()
        options = {
                'qmc': {
                        'timestep': 0.01,
                        'num_steps': 10,
                        'num_blocks': 10,
                        'rng_seed': 8,
                    },
                }
        comm = MPI.COMM_WORLD
        afqmc = AFQMC(options=options, mf=mf, verbose=0)
        afqmc.run(comm=comm, verbose=0)
        afqmc.finalise(verbose=0)

    def tearDown(self):
        cwd = os.getcwd()
        files = ['estimates.0.h5']
        for f in files:
            try:
                os.remove(cwd+'/'+f)
            except OSError:
                pass
