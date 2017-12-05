import numpy
import afqmcpy.trail_wave_function as tw

# 1. Perform initial CPMC calculation
table = {
    "model": {
        "name": "Hubbard",
        "t": 1.0,
        "U": 4,
        "nx": 4,
        "ny": 4,
        "ktwist": [0],
        "nup": 3,
        "ndown": 3,
    },
    "qmc_options": {
        "method": "CPMC",
        "dt": 0.05,
        "nsteps": 2,
        "nmeasure": 10,
        "nwalkers": 100,
        "npop_control": 10000,
        "temperature": 0.0,
        "hubbard_stratonovich": "discrete",
        "importance_sampling": True,
        "rng_seed": 7,
        "ueff": 4,
        "trial_wavefunction": "free_electron",
    }
}
