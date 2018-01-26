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

def generate_qmc_rdm(cpmc, options, comm, rdm_delta, index=0):
    # 1. Generate psi
    data = 'estimates.' + str(index) + '.h5'
    estimate_opts = options.get('estimates')
    estimate_opts['filename'] = data
    if index != 0:
        estimators = afqmcpy.estimators.Estimators(estimate_opts,
                                                   cpmc.root,
                                                   cpmc.qmc,
                                                   cpmc.system,
                                                   cpmc.trial,
                                                   cpmc.propagators.BT_BP)
        cpmc.estimators = estimators
    psi0 = afqmcpy.walker.Walkers(cpmc.system, cpmc.trial,
                                  cpmc.qmc.nwalkers,
                                  cpmc.estimators.nprop_tot,
                                  cpmc.estimators.nbp)
    cpmc.run(psi=psi0, comm=comm)
    cpmc.finalise()
    # 2. Extract initial 1RDM
    if cpmc.root:
        start = int(4.0/(cpmc.estimators.nbp*cpmc.qmc.dt))
        rdm, err = analysis.blocking.average_rdm(data, 'back_prop', skip=start)
        bp_av, norm_av = analysis.blocking.analyse_estimates([data], start_time=4)
        # check quality.
        mean_err = err.diagonal().mean()
        print ("# Mean error in CPMC RDM: %f"%mean_err)
        print ("# CPMC energy: %f +/- %f"%(norm_av.E, norm_av.E_error))
    else:
        mean_err = None
    mean_err = comm.bcast(mean_err, root=0)
    if (mean_err > rdm_delta):
        warnings.warn("Error too large in CPMC rdm: %f."%mean_err)
        warnings.warn("Need to run for roughly %d times longer."%(mean_err/rdm_delta)**2.0)
        warnings.warn("Exiting now.")
        sys.exit()
    else:
        if cpmc.root:
            return (rdm, err, norm_av.E, norm_av.E_error)
        else:
            return (None, None, None, None)


def find_uopt(rdm, system, trial, mmin, mmax, index=0, dtype=numpy.float64):
    ueff = numpy.linspace(mmin, mmax, 200)
    cost = numpy.zeros(len(ueff))
    psis = numpy.zeros((len(ueff), system.nbasis, system.ne), dtype=dtype)
    uhf = afqmcpy.trial_wavefunction.UHF(system, system.ktwist.all() is not None, trial)
    for (i, u) in enumerate(ueff):
        (niup, nidown, e_up, e_down) = uhf.diagonalise_mean_field(system, u,
                                                                  rdm[0].diagonal(),
                                                                  rdm[1].diagonal())
        psis[i] = copy.deepcopy(uhf.trial)
        cost[i] = numpy.sum((niup-rdm[0].diagonal())**2.0)
        cost[i] += numpy.sum((nidown-rdm[1].diagonal())**2.0)
        cost[i] = numpy.sqrt(cost[i]) / system.nbasis

    pl.plot(ueff, cost, marker='o')
    pl.xlabel(r"$U_{\mathrm{eff}}$")
    pl.ylabel(r"Cost Function")

    imin = numpy.argmin(cost)
    uopt = ueff[imin]
    pl.axvline(uopt, color='red', linestyle=':')
    print ("# Optimal Ueff : %f"%uopt)
    pl.savefig('uopt.'+str(index)+'.pdf', fmt='pdf', bbox_inches='tight')
    pl.cla()
    return (uopt, psis[imin])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# 1. Perform initial CPMC calculation
start = time.time()
input_file = 'stable.json'

(options, comm) = afqmcpy.calc.init(input_file)
if comm is not None:
    cpmc = afqmcpy.calc.setup_parallel(options, comm)
else:
    cpmc = afqmcpy.cpmc.CPMC(options.get('model'),
                             options.get('qmc_options'),
                             options.get('estimates'),
                             options.get('trial_wavefunction'))
# Options
# -------
# stochastic error bar in density
rdm_delta = 4e-2
nself_consist = 10

energies = numpy.zeros(nself_consist)
errors = numpy.zeros(nself_consist)
uopt = numpy.zeros(nself_consist)
system = cpmc.system
uhf_input = {
    "name": "UHF",
    "ueff": 0.5,
    "ninitial": 10,
    "nconv": 1000,
    "deps": 1e-6,
    "verbose": False,
}
# Initial step starting from FE trial wavefunction to construct constraint.
if rank == 0:
    print ("# Self consistency cycle %d of %d"%(0, nself_consist))
(rdm, rdm_err, energies[0], errors[0]) = (
    generate_qmc_rdm(cpmc, options, comm, rdm_delta, 0)
)
wfn_file = 'uopt_trial_wfn.0.npy'
if rank == 0:
    start = int(4.0/(cpmc.estimators.nbp*cpmc.qmc.dt))
    rdm, err = analysis.blocking.average_rdm('estimates.0.h5', 'back_prop', skip=start)
    (uopt[0], psi_opt) = find_uopt(rdm, system, uhf_input, 0.01, 10, index=0,
                                   dtype=cpmc.psi.walkers[0].phi.dtype)
    # print (psi_opt)
    numpy.save(wfn_file, psi_opt)
comm.Barrier()
cpmc.trial.psi = numpy.load(wfn_file)
for isc in range(1, nself_consist):
    if rank == 0:
        print ("# Self consistency cycle %d of %d"%(isc+1, nself_consist))
    (rdm, rdm_err, energies[isc], errors[isc]) = (
        generate_qmc_rdm(cpmc, options, comm, rdm_delta, isc)
    )
    if rank == 0:
        (uopt[isc], psi_opt) = find_uopt(rdm, system, uhf_input, 0.01, 10,
                                         index=isc, dtype=cpmc.psi.walkers[0].phi.dtype)
        # write psi to file.
        wfn_file = 'uopt_trial_wfn.'+str(isc)+'.npy'
        # overkill
        numpy.save(wfn_file, psi_opt)
    comm.Barrier()
    cpmc.trial.psi = numpy.load(wfn_file)

if rank == 0:
    pl.errorbar(range(1,nself_consist+1), energies, yerr=errors, fmt='o')
    pl.xlabel(r'iteration')
    pl.xticks(range(1, nself_consist+1), [str(x)[:5] for x in uopt], rotation=45)
    pl.xlim([0, nself_consist+1])
    pl.ylabel(r'$E_{\mathrm{CPMC}}$')
    pl.savefig('self_consist_conv.pdf', fmt='pdf', bbox_inches='tight')
