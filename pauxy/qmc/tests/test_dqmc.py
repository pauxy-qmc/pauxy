import pytest
from mpi4py import MPI
from pauxy.qmc.dqmc import DQMC
from pauxy.analysis.extraction import extract_mixed_estimates

@pytest.mark.driver
def test_hubbard():
    options = {
            'verbosity': 0,
            'get_sha1': False,
            'qmc': {
                'timestep': 0.05,
                'num_steps': 10,
                'beta': 1.0,
                'blocks': 5,
                'rng_seed': 7,
            },
            'model': {
                'name': "Hubbard",
                'nx': 6,
                'ny': 1,
                'nup': 3,
                "U": 4,
                "mu": 2,
                'ndown': 3,
            },
            'propagator': {
                'hubbard_stratonovich': 'discrete',
                'stack_size': 1
            }
        }
    comm = MPI.COMM_WORLD
    afqmc = DQMC(comm=comm, options=options)
    afqmc.run(comm=comm)
    data = extract_mixed_estimates('estimates.0.h5')
    ref = -1.141830530555080
    val = data['ETotal'].values[-1]
    assert abs(val-ref) == pytest.approx(0.0)

def teardown_module(self):
    cwd = os.getcwd()
    files = ['estimates.0.h5']
    for f in files:
        try:
            os.remove(cwd+'/'+f)
        except OSError:
            pass
