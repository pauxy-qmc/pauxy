from mpi4py import MPI
from pauxy.qmc.thermal_afqmc import ThermalAFQMC
import numpy

comm = MPI.COMM_WORLD

system = {
    "name": "Hubbard",
    "nup": 7,
    "ndown": 7,
    "symmetric": True,
    "pinning_fields": True,
    "nx": 4,
    "ny": 4,
    "U": 4,
    "mu": 1
}

qmc = {
    "dt": 0.05,
    "nsteps": 10,
    "nwalkers": 288,
    "beta": 5
}

mu = numpy.linspace(0.8,1,5)
for i, b in enumerate([2, 5, 10, 20]):
    estim = {
        "overwrite": False,
        "basename": "estimates_beta_{}".format(i)
        }
    qmc["beta"] = b
    # Scan over chemical potential values
    for m in mu:
        system['mu'] = m
        afqmc = ThermalAFQMC(comm, system, qmc, estimates=estim,
                             walker_opts=walker_opts,
                             verbose=(True and comm.rank==0))
        afqmc.run(comm=comm, verbose=True)
