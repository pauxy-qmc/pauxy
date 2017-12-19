import numpy
import warnings
import sys
import h5py
import time
import json
import copy
from mpi4py import MPI
import matplotlib.pyplot as pl
import afqmcpy.cpmc
import analysis.extraction

def generate_qmc_rdm(state, options, comm, rdm_delta, index=0):
    # 1. Generate psi
    trial.
    estimators = afqmcpy.estimators.Estimators(options.get('estimates'),
                                               state.root,
                                               state.uuid,
                                               state.qmc,
                                               state.system.nbasis,
                                               state.json_string,
                                               state.trial.type=='GHF')
    psi = afqmcpy.cpmc.run(state, psi0, estimators, comm)
    afqmcpy.cpmc.finalise(state, estimators, start)
    # 2. Extract initial 1RDM
    data = 'estimates.'+str(index)+'.h5'
    rdm, err = analysis.blocking.average_rdm(data, skip=2)
    bp_av, norm_av = analysis.blocking.analyse_estimates(data, 100)
    # check quality.
    mean_err = err.diagonal().mean()
    print ("# Mean error in CPMC RDM: %f"%mean_err)
    print ("# CPMC energy: %f +/- %f"%(norm_av.E, norm_av.E_err))
    if (mean_err > rdm_delta):
        warnings.warn("Error too large in CPMC rdm: %f."%mean_err)
        warnings.warn("Need to run for roughly %d times longer."%(mean_err/rdm_delta)**2.0)
        warnings.warn("Exiting now.")
        sys.exit()
    else:
        return (rdm, err, norm_av.E, norm_av.E_err)


def find_uopt(system, trial, mmin, mmax):
    ueff = numpy.linspace(mmin, mmax, 20)
    cost = numpy.zeros(len(ueff))
    psis = numpy.zeros(len(ueff), system.nbasis, system.nel)
    for (i, u) in enumerate(ueff):
        print ("##########################")
        print ("# Scan %d of %d. Ueff : %f"%(i, len(ueff), u))
        trial['ueff'] = u
        uhf = afqmcpy.trial_wavefunction.UHF(system, False, trial)
        psis[i] = copy.deepcopy(uhf.trial.psi)
        cost[i] = (numpy.sum((uhf.nav[0]-rdm[0].diagonal())**2.0))**0.5/len(uhf.nav[0])
        cost[i] += (numpy.sum((uhf.nav[1]-rdm[1].diagonal())**2.0)**0.5)/len(uhf.nav[1])
        print ("##########################")

    pl.plot(ueff, cost, marker='o')
    pl.xlabel(r"$U_{\mathrm{eff}}$")
    pl.ylabel(r"Cost Function")

    imin = numpy.argmin(cost)
    uopt = ueff[imin]
    pl.axvline(uopt, color='red', linestyle=':')
    print ("# Optimal Ueff : %f"%uopt)
    pl.show()
    return (uopt, psis[imin])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# 1. Perform initial CPMC calculation
start = time.time()
input_file = 'stable.json'
if rank == 0:
    print('# Initialising AFQMCPY simulation from %s'%input_file)
    with open(input_file) as inp:
        options = json.load(inp)
    inp.close()
    # sometimes python is beautiful
    print('# Running on %s core%s'%(nprocs, 's' if nprocs > 1 else ''))
else:
    options = None
if comm is not None:
    options = comm.bcast(options, root=0)
(state, psi0, comm) = afqmcpy.cpmc.setup(options, comm)

# Options
# -------
# stochastic error bar in density
rdm_delta = 5e-3
nself_consist = 5

energies = numpy.zeros(nself_consist)
errors = numpy.zeros(nself_consist)
uopt = numpy.zeros(nself_consist)
system = state.system
uhf_input = {
    "name": "UHF",
    "ueff": 0.5,
    "ninitial": 10,
    "nconv": 1000,
    "deps": 1e-6,
    "verbose": False,
}
# Initial step starting from FE trial wavefunction to construct constraint.
print ("# Self consistency cycle %d of %d"%(0, nself_consist))
(rdm, rdm_err, energies[isc], errors[isc]) = (
    generate_qmc_rdm(state, options, comm, rdm_delta, isc)
)
(uopt[isc], psi_opt) = find_uopt(system, uhf_input, 0.1, 3)
wfn_file = 'uopt_trial_wfn.'+str(isc)'.npy'
numpy.save(wfn_file, psi_opt)
state.trial.psi = numpy.load(wfn_file)
for isc in range(1, nself_consist):
    print ("# Self consistency cycle %d of %d"%(isc, nself_consist))
    (rdm, rdm_err, energies[isc], errors[isc]) = (
        generate_qmc_rdm(state, options, comm, rdm_delta, isc)
    )
    (uopt[isc], psi_opt) = find_uopt(system, uhf_input, 0.1, 3)
    # write psi to file.
    wfn_file = 'uopt_trial_wfn.'+str(isc)'.npy'
    numpy.save(wfn_file, psi_opt)
    state.trial.psi = numpy.load(wfn_file)
