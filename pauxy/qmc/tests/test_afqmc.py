import unittest
from mpi4py import MPI
import os
from pyscf import gto, ao2mo, scf
from pauxy.qmc.calc import setup_calculation
from pauxy.qmc.afqmc import AFQMC
from pauxy.systems.generic import Generic
from pauxy.utils.from_pyscf import integrals_from_scf

class TestGeneric(unittest.TestCase):

    def test_from_pyscf(self):
        atom = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
        mf = scf.RHF(atom)
        ehf = mf.kernel()
        options = {
                'get_sha1': False,
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

    def test_ueg(self):
        options = {
                'verbosity': 0,
                'get_sha1': False,
                'qmc': {
                    'timestep': 0.01,
                    'num_steps': 10,
                    'num_blocks': 10,
                    'rng_seed': 8,
                },
                'model': {
                    'name': "UEG",
                    'rs': 2.44,
                    'ecut': 4,
                    'nup': 7,
                    'ndown': 7,
                },
                'trial': {
                    'name': 'hartree_fock'
                }
            }
        (afqmc, comm) = setup_calculation(options)
        afqmc.run(comm=comm, verbose=0)
        afqmc.finalise(verbose=0)
        # ref = 6.828957055614434+0.22576828445100017j
        # ref = 6.821009376769289+0.13276828693227866j
        # FDM: Update reference following merge sort update.
        ref = 6.562928368348016+0.07235261291158207j
        cur = afqmc.psi.walkers[0].phi.trace()
        self.assertAlmostEqual(cur.real, ref.real)
        self.assertAlmostEqual(cur.imag, ref.imag)

    def tearDown(self):
        cwd = os.getcwd()
        files = ['estimates.0.h5']
        for f in files:
            try:
                os.remove(cwd+'/'+f)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main()
