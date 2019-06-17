#!/usr/bin/env python
# Triplet UHF ground state of carbon atom.

import h5py
import numpy
from mpi4py import MPI
from pyscf import gto, scf

from pauxy.qmc.afqmc import AFQMC

mol = gto.Mole()
mol.basis = 'cc-pvtz'
mol.atom = (('C', 0,0,0),)
mol.spin = 2
mol.verbose = 4
mol.build()

mf = scf.UHF(mol)
mf.kernel()
# Check if UHF solution is stable.
mf.stability()

options = {
    'qmc': {
        'timestep': 0.01,
        'num_steps': 100,
        'print_freq': 10,
        'rng_seed': 8,
        'num_walkers': 10
    },
}
comm = MPI.COMM_WORLD
verbose = 1
afqmc = AFQMC(options=options, mf=mf, verbose=verbose)
print(afqmc.qmc.nmeasure)
afqmc.run(comm=comm, verbose=verbose)
afqmc.finalise(verbose=verbose)
