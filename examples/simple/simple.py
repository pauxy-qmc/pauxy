from pauxy.qmc.afqmc import AFQMC
from pauxy.qmc.calc import AFQMC, init_communicator

system = {
    "name": "Generic",
    "atom": "Neon",
    "nup": 5,
    "ndown": 5,
    "integrals": "fcidump.ascii"
}
qmc_options = {
    "dt": 0.05,
    "nsteps": 2000,
    "nmeasure": 10,
    "nwalkers": 1,
    "npop_control": 1,
    "nstabilise": 1,
    "rng_seed": 7
}
trial_wavefunction = {
    "name": "hartree_fock"
}
propagator = {
    "hubbard_stratonovich": "continuous",
    "expansion_order": 6,
    "free_projection": False
}

comm = init_communicator()
afqmc = AFQMC(system, qmc_options, {}, trial_wavefunction, propagator)
afqmc.run(comm=comm)
(energy, error) = afqmc.get_energy()
print (energy, error)
