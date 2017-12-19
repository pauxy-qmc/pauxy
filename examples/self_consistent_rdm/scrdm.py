import numpy
import afqmcpy.initialise
import analysis.extraction
import sys
import h5py

# 1. Perform initial CPMC calculation
input_file = 'stable.json'
(state, psi, comm) = afqmcpy.initialise.initialise(input_file)
afqmcpy.initialise.finalise(state, 0)

data = h5py.File('estimates.0.h5', 'r')
print (data.items())
# 2. Extract initial 1RDM
rdm, err = analysis.blocking.average_rdm('estimates.0.h5')
# check quality.
mean_err = err.diag.mean()
if (mean_err > rdm_delta):
    warnings.warn("Error too large in CPMC rdm: %f. Exiting"%mean_err)
    sys.exit()

# 3.
