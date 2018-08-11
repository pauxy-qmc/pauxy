# Use pauxy to compute energy and RDM of Neon atom in cc-PVDZ basis.
# Integral file generated using PYSCF.
import numpy
from pauxy.qmc.afqmc import AFQMC
from pauxy.qmc.calc import init_communicator

system = {
    "name": "Generic", # Descriptive
    "atom": "Neon", # Descriptive
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
estimators = {
    "back_propagated": {
        "nback_prop": 40,
        "rdm": True
    }
}

comm = init_communicator()
afqmc = AFQMC(system, qmc_options, estimators, trial_wavefunction, propagator)
afqmc.run(comm=comm, verbose=False)
(energy, error) = afqmc.get_energy()
print ("Mixed estimate for the energy: %f +/- %f"%(energy, error))
(one_rdm, one_rdm_error) = afqmc.get_one_rdm()
print ("Total number of electrons: %f"%numpy.einsum('kii->', one_rdm).real)
