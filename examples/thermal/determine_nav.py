#!/usr/bin/env python

import glob
import numpy
import matplotlib.pyplot as pl
from pauxy.trial_density_matrices.onebody import OneBody
from pauxy.qmc.calc import init_communicator, setup_calculation
from pauxy.qmc.thermal_afqmc import ThermalAFQMC
from pauxy.propagation.utils import get_propagator
from pauxy.analysis.thermal import analyse_energy
from pauxy.utils.io import to_json

sys = {
    "name": "Hubbard",
    "t": 1.0,
    "U": 4,
    "nx": 4,
    "ny": 4,
    "nup": 7,
    "ndown": 7
}
qmc = {
    "dt": 0.005,
    "nsteps": 10,
    "nmeasure": 1,
    "nwalkers": 36,
    "npop_control": 10,
    "nstabilise": 10,
    "beta": 1,
    "rng_seed": 7
}
estimates = {'mixed': {'thermal': True} }
options = {'model': sys, 'qmc_options': qmc, 'estimates': estimates}
(afqmc, comm) = setup_calculation(options)
# afqmc.run(comm=comm)
# mu0 = afqmc.trial.mu
scan = numpy.linspace(0.5, 1.5, 10)
for mu in scan:
    afqmc.trial = OneBody({'mu': mu}, afqmc.system, afqmc.qmc.beta, afqmc.qmc.dt)
    afqmc.propagators = get_propagator({}, afqmc.qmc, afqmc.system, afqmc.trial)
    afqmc.psi.reset(afqmc.trial)
    afqmc.estimators.json_string = to_json(afqmc)
    afqmc.estimators.reset(comm.rank==0)
    afqmc.run(comm=comm)

# if comm.rank == 0:
    # files = glob.glob('*.h5')
    # data = analyse_energy(files)
    # print (data.to_string())
